"""
A script to collect a batch of human demonstrations that can be used
to generate a learning curriculum (see `demo_learning_curriculum.py`).

The demonstrations can be played back using the `playback_demonstrations_from_pkl.py`
script.
"""

import os
import shutil
import time
import argparse
import datetime
import h5py
from glob import glob
import numpy as np
import json

import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers import DataCollectionWrapper
from robomimic.utils.env_utils import create_env_from_metadata
from robomimic.utils.file_utils import get_env_metadata_from_dataset


def sample_actions(time_steps, a_dim, bias, beta=0.5, std=1):
    action_bias_trunc = bias[:a_dim]
    noise_samples = [(np.random.randn(a_dim) * std) + action_bias_trunc]
    for _ in range(1, time_steps):
        noise_samp = beta * noise_samples[-1] + (1-beta) * ((np.random.randn(a_dim) * std) + action_bias_trunc)
        noise_samples.append(noise_samp)
    noise_samples = np.stack(noise_samples, axis=0)
    return noise_samples


def normalize(v):
    return v / np.linalg.norm(v)


def within_bounds(v):
    return -0.4 < v[0] < 0.4 and -0.4 < v[1] < 0.4


def collect_random_trajectory(env, max_steps, arm, env_configuration, filter_condition, target_object):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
    """

    env.reset()

    # ID = 2 always corresponds to agentview
    #env.render()

    is_first = True

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal

    # Loop until we get a reset from the input or the task completes
    step_num = 0
    action_dim = env.action_dim
    low, high = env.action_spec

    obj_positions = [env.get_object_positions()]

    # target_object = np.random.randint(0, 4)
    print(f'target object {target_object}')
    # Generate a random vector uniformly on the half circle with positive x values
    angle = np.random.rand() * np.pi
    direction_vector = np.array([np.sin(angle), np.cos(angle)])
    # direction_vector[0] = np.abs(direction_vector[0])
    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    target_object_position = obj_positions[-1][target_object][:2] + 0.3 * direction_vector
    num_times_checked = 1
    # while not within_bounds(target_object_position):
    #     if num_times_checked > 50:
    #         target_object = np.random.randint(0, 4)
    #         direction_vector = np.random.rand(2)
    #         direction_vector[0] = np.abs(direction_vector[0])
    #         direction_vector = direction_vector / np.linalg.norm(direction_vector)
    #         target_object_position = obj_positions[-1][target_object][:2] + 0 * direction_vector
    #         break
    #     else:
    #         num_times_checked += 1
    #         target_object = np.random.randint(0, 4)
    #         direction_vector = np.random.rand(2)
    #         direction_vector[0] = np.abs(direction_vector[0])
    #         direction_vector = direction_vector / np.linalg.norm(direction_vector)
    #         target_object_position = obj_positions[-1][target_object][:2] + 0.3 * direction_vector
    initial_location = obj_positions[-1][target_object][:2]
    print(f'initial location {obj_positions[-1][target_object][:2]}')
    print(f'target location {target_object_position}')

    action_queue = []
    img_observations = []

    while step_num < max_steps:
        step_num += 1
        # Set active robot
        active_robot = env.robots[0] if env_configuration == "bimanual" else env.robots[arm == "left"]

        # Get the newest action
        if len(action_queue) == 0:
            bias = np.zeros(action_dim)
            # if step_num == 1:
            #     bias[2] = -0.5
            # print(bias)
            std = np.array([0.05, 0.05, 0.05, 0])
            control_freq = env.env.control_freq
            if step_num == 1:
                plan_len = 6 if control_freq == 5 else 3
            else:
                plan_len = 12 if control_freq == 5 else 8
            gain = 7
            print(f'---- step num {step_num} -----')
            action_noise = 0.05
            # if 0 < step_num < 10:
            if 0 < step_num < 10:
                current_object_position = obj_positions[-1][target_object][:2]
                print('current object position', current_object_position)
                target_arm_position = current_object_position + 0.175 * normalize(current_object_position - target_object_position)
                print('target arm position', target_arm_position)
                current_arm_position = env.env.get_gripper_pos()[:2]
                print('current arm position', current_arm_position)
                bias[:2] = (target_arm_position - current_arm_position) * gain
                print('bias', bias[:2])
                # action_queue = sample_actions(1, action_dim, bias, std=np.array([0.0, 0.0, 0.0, 0]))
                action_queue = sample_actions(1, action_dim, bias, std=np.array([action_noise, action_noise, action_noise, 0]))
                action_queue = action_queue + np.random.randn(*action_queue.shape) * std
                action_queue = np.clip(action_queue, low, high)
            elif step_num > 1:
                current_arm_position = env.env.get_gripper_pos()[:2]
                print('current arm position', current_arm_position)
                current_object_position = obj_positions[-1][target_object][:2]
                print('current object position', current_object_position)
                print('target object position', target_object_position)
                bias[:2] = (target_object_position - current_object_position) * 5
                # bias[:2] = normalize((current_object_position - current_arm_position)) * 2
                print('bias', bias[:2])
                # action_queue = sample_actions(1, action_dim, bias, std=np.array([0.00, 0.00, 0.0, 0]))
                action_queue = sample_actions(1, action_dim, bias, std=np.array([action_noise, action_noise, action_noise, 0]))
                action_queue = action_queue + np.random.randn(*action_queue.shape) * std
                action_queue = np.clip(action_queue, low, high)
            if 1 < step_num < 3 or 10 < step_num < 16:
                # action_queue = sample_actions(plan_len, action_dim, bias, std=std)
                # print(action_queue)
                current_arm_position = env.env.get_gripper_pos()
                if current_arm_position[2] > 0.83:
                    print(current_arm_position[2])
                    # action_queue[:, 2] = -1 + np.random.randn(1) * 0.005
                    # action_queue[:, 0] = np.random.randn(1) * 0.05
                    # action_queue[:, 1] = np.random.randn(1) * 0.05
                    action_queue[:, 2] = -1 + np.random.randn(1) * action_noise
                    action_queue[:, 0] = np.random.randn(1) * action_noise
                    action_queue[:, 1] = np.random.randn(1) * action_noise
                action_queue = np.clip(action_queue, low, high)

            action_queue = list(action_queue)

        action = action_queue.pop(0)
        # print(action)
        
        # If action is none, then this a reset so we should break
        if action is None:
            break

        # Run environment step
        obs, rew, _, _ = env.step(action)
        # img_observations.append(obs['agentview_image'][::-1])
        #env.render()
        obj_positions.append(env.get_object_positions())
        print(obj_positions[-1])

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # # state machine to check for having a success for 10 consecutive timesteps
        # if env._check_success():
        #     if task_completion_hold_count > 0:
        #         task_completion_hold_count -= 1  # latched state, decrement count
        #     else:
        #         task_completion_hold_count = 10  # reset count on first success timestep
        # else:
        #     task_completion_hold_count = -1  # null the counter if there's no success

    print(f'final location {obj_positions[-1][target_object][:2]}')
    print(f'DISTANCE {np.linalg.norm(initial_location - obj_positions[-1][target_object][:2])}')
    # cleanup for end of data collection episodes

    take_trajectory = filter_condition(obj_positions, target_object)
    # take_trajectory = take_trajectory and manual_inspection(img_observations)
    if not take_trajectory:
        env.ep_directory = os.path.join("/viscam/u/stian/tmp/bad_trajs", "bad_traj")
    env.close()
    return take_trajectory


def manual_inspection(img_observations):
    from fitvid.utils import save_moviepy_gif
    save_moviepy_gif(img_observations, 'img_observations')
    if input('okay?').lower() == 'y':
        return True
    else:
        return False


def filter_object_motion(obj_positions, target_object):
    init_obj_positions = obj_positions[0]
    final_obj_positions = obj_positions[-1]
    # iterate over initial and final position for each object
    object_distances = []
    y_motion = []
    for i, (init_p, final_p) in enumerate(zip(init_obj_positions, final_obj_positions)):
        # make sure it didn't fall off the table
        if np.abs(init_p[2] - final_p[2]) > 0.5:
            return False
        if i != 3 or target_object == 3:
            object_distances.append(np.linalg.norm(init_p[:2] - final_p[:2]))
            y_motion.append(np.abs(init_p[1] - final_p[1]))
    print(np.max(object_distances))
    if np.max(object_distances) > 0.08:
        return True
    return False

def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file. 
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
                
        if len(states) == 0:
            continue

        # Delete the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action,
        # so we end up with an extra state at the end.
        del states[-1]
        assert len(states) == len(actions)

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # store model xml as an attribute
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()
        ep_data_grp.attrs["model_file"] = xml_str

        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(suite.models.assets_root, "demonstrations"),
    )
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument("--config", type=str, default="single-arm-opposed",
                        help="Specified environment configuration if necessary")
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--camera", type=str, default="agentview", help="Which camera to use for collecting demos")
    parser.add_argument("--out-dir", type=str, default="", help="Out dir to save demo.hdf5 file in")
    parser.add_argument("--env-like", type=str,
                        help="Dataset file to create env based on")

    parser.add_argument("--controller", type=str, default="OSC_POSE",
                        help="Choice of controller.'")
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--num-trajs", type=int, default=100)
    parser.add_argument("--control-freq", type=float, default=20)
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    parser.add_argument("--filter-object-motion", action='store_true', help="use filter obj motion")
    args = parser.parse_args()

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config

    # Create environment
    if args.env_like == 'None':
        config.update({
            'control_freq': args.control_freq,
            'controller_configs': load_controller_config(default_controller=args.controller)
        })
        env = suite.make(
            **config,
            has_renderer=False,
            # has_offscreen_renderer=True,
            has_offscreen_renderer=False,
            render_camera=args.camera,
            ignore_done=True,
            use_camera_obs=False,
            reward_shaping=True,
            #control_freq=args.control_freq,
            #controller_configs=load_controller_config(default_controller=args.controller)
        )
        env_info = json.dumps(config)
    else:
        env_meta = get_env_metadata_from_dataset(args.env_like)
        if args.controller:
            print(f'!!! Overriding controller to {args.controller}!')
            env_meta['env_kwargs']['controller_configs']['type'] = args.controller
            if args.controller == 'OSC_POSITION':
                env_meta['env_kwargs']['controller_configs']['output_max'] = env_meta['env_kwargs']['controller_configs']['output_max'][:3]
                env_meta['env_kwargs']['controller_configs']['output_min'] = env_meta['env_kwargs']['controller_configs']['output_min'][:3]
        env_meta['env_kwargs']['control_freq'] = args.control_freq
        print(env_meta)
        env = create_env_from_metadata(env_meta).env
        env_info = json.dumps(env_meta['env_kwargs'])

    # Wrap this with visualization wrapper
    #env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string

    # wrap the environment with data collection wrapper
    tmp_directory = "/viscam/u/stian/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    # make a new timestamped directory
    if args.out_dir:
        new_dir = os.path.join(args.directory, args.out_dir)
    else:
        t1, t2 = str(time.time()).split(".")
        new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))

    os.makedirs(new_dir, exist_ok=True)
    # collect demonstrations
    successful_trajectories = 0
    object_target = 0

    if args.filter_object_motion:
        filter = filter_object_motion
    else:
        # No filter, all trajectories are success
        filter = lambda x, y: True

    while successful_trajectories < args.num_trajs:
        take_trajectory = collect_random_trajectory(env, args.max_steps, args.arm, args.config, filter, object_target)
        if take_trajectory:
            successful_trajectories += 1
            object_target = (object_target + 1) % len(env.get_object_positions())
        print(successful_trajectories)
        if successful_trajectories % 500 == 1 or successful_trajectories == args.num_trajs-1:
            gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)
