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


def collect_random_trajectory(env, max_steps, arm, env_configuration):
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

    action_queue = [] 

    while step_num < max_steps:
        step_num += 1
        # Set active robot
        active_robot = env.robots[0] if env_configuration == "bimanual" else env.robots[arm == "left"]

        # Get the newest action
        if len(action_queue) == 0:
            print(step_num)
            bias = np.zeros(action_dim)
            # if step_num == 1:
            #     bias[2] = -0.5
            print(bias)
            std = np.array([1, 1, 0.2, 1])
            if step_num == 1:
                plan_len = 6
            else:
                plan_len = 12
            action_queue = sample_actions(plan_len, action_dim, bias, std=std)
            action_queue = np.clip(action_queue, low, high)
            if step_num == 1:
                action_queue[:, 2] = -0.7
            action_queue = list(action_queue)

        action = action_queue.pop(0)
        print(action)
        
        # If action is none, then this a reset so we should break
        if action is None:
            break

        # Run environment step
        env.step(action)
        #env.render()

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

    # cleanup for end of data collection episodes
    env.close()


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
    parser.add_argument("--env-like", type=str, 
                        help="Dataset file to create env based on")

    parser.add_argument("--controller", type=str, default="OSC_POSE",
                        help="Choice of controller.'")
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--num-trajs", type=int, default=100)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
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
            has_offscreen_renderer=True,
            render_camera=args.camera,
            ignore_done=True,
            use_camera_obs=False,
            reward_shaping=True,
            #control_freq=args.control_freq,
            #controller_configs=load_controller_config(default_controller=args.controller)
        )
        env_info = json.dumps(config)
    else:

        # env = suite.make(
        #     **config,
        #     has_renderer=False,
        #     has_offscreen_renderer=False,
        #     render_camera=args.camera,
        #     ignore_done=True,
        #     use_camera_obs=False,
        #     reward_shaping=True,
        #     control_freq=20,
        #     controller_configs=load_controller_config(default_controller="OSC_POSITION")
        # )
        # env_info = json.dumps(config)

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
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(new_dir)
    # collect demonstrations
    for i in range(args.num_trajs):
        collect_random_trajectory(env, args.max_steps, args.arm, args.config)
        if i % 500 == 0 or i == args.num_trajs-1:
            gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)
