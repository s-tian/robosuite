from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CanObject, BallObject, CylinderObject, CapsuleObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial, ALL_TEXTURES
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat, mat2euler


class PushCenterMulti(SingleArmEnv):
    """
    This class corresponds to the lifting task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        reward_function='push_center',
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        self.reward_function = reward_function
        print(f'Using reward function {self.reward_function}!')

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.25 is provided if the cube is lifted

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 1], to encourage the arm to reach the cube
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            - Lifting: in {0, 1}, non-zero if arm has lifted the cube

        The sparse reward only consists of the lifting component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.25 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0

        if self.reward_function == 'push_center':
            # sparse completion reward
            if self._check_success():
                reward = 4.0
            # use a shaping reward
            elif self.reward_shaping:
                for obj_id in self.object_body_ids:

                    # reaching reward
                    # cube_pos = self.sim.data.body_xpos[obj_id]
                    # gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
                    # dist = np.linalg.norm(gripper_site_pos - cube_pos)
                    # reaching_reward = 1 - np.tanh(10.0 * dist)
                    # reward += reaching_reward

                    # xy distance reward
                    cube_xy = self.sim.data.body_xpos[obj_id][:2]
                    table_xy = self.model.mujoco_arena.table_offset[:2]
                    target = table_xy.copy()
                    target[0] += 0.2
                    # cube is higher than the table top above a margin
                    center_dist = np.linalg.norm(cube_xy - target)
                    reward += 1 - np.tanh(10.0 * center_dist)

            # Scale reward if requested
            if self.reward_scale is not None:
                reward *= self.reward_scale / 4.0

        elif self.reward_function == 'tip_cylinder':
            if self._check_success():
                reward = 2.0
            elif self.reward_shaping:
                reward += 1 - self.get_cylinder_verticality()
                cylinder_id = self.sim.model.body_name2id(self.objects[2].root_body)
                cylinder_pos = self.sim.data.body_xpos[cylinder_id]
                gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
                dist = np.linalg.norm(cylinder_pos - gripper_site_pos)
                reaching_reward = 1 - np.tanh(10.0 * dist)
                reward += reaching_reward
            # print(z_value)
            # if self.reward_shaping:
            if self.reward_scale is not None:
                reward *= self.reward_scale / 2.0
        else:
            raise NotImplementedError
        return reward

    def get_object_positions(self):
        positions = []
        for obj_id in self.object_body_ids:
            positions.append(np.copy(self.sim.data.body_xpos[obj_id][:3]))
        return positions

    def get_gripper_pos(self):
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        return np.copy(gripper_site_pos)

    def get_cylinder_verticality(self):
        # cylinder is object index 2 in self.objects
        cylinder_id = self.sim.model.body_name2id(self.objects[2].root_body)
        # print(self.sim.data.body_xmat[cylinder_id])
        rot_mat = self.sim.data.body_xmat[cylinder_id]
        # the result of z^TRz, where z = [0, 0, 1], represents how aligned the normal from the circular face
        # is with the unit vector in the z direction. The closer to 0 it is, the flatter the cylinder is.
        zRz = rot_mat[-1]
        return np.abs(zRz)

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }

        random_shininess = np.random.uniform()

        texture_list = list(ALL_TEXTURES)
        materials = []
        for i in range(4):
            texture = None
            while not texture or 'Cereal' in texture:
                texture_rnd_idx = np.random.choice(len(texture_list))
                texture = texture_list[texture_rnd_idx]
            mat_attrib = {
                "texrepeat": "1 1",
                "specular": f"{random_shininess}",
                "shininess": f"{random_shininess}",
                "reflectance": f"{random_shininess}",
            }
            mat = CustomMaterial(
                texture=texture,
                tex_name="block_tex",
                mat_name="block_tex_mat",
                tex_attrib=tex_attrib,
                mat_attrib=mat_attrib,
            )
            materials.append(mat)

        self.box_object = BoxObject(
            name="cube",
            size=[0.06, 0.06, 0.06],
            # density=((0.04/0.04)**3),
            density=1000,
            #density=0.5,
            #size_min=[0.020, 0.020, 0.020],  # [0.015, 0.015, 0.015],
            #size_max=[0.022, 0.022, 0.022],  # [0.018, 0.018, 0.018])
            rgba=[1, 0, 0, 0],
            friction=[0.7, 0.005, 0.0001],
            # solimp=[0.97, 0.990, 0.001, 0.5, 1],
            # solref=[0.02, 10],
            # solref=[-5000.0, -100.0],
            material=materials[0],
        )

        self.box_object2 = BoxObject(
            name="cube2",
            size=[0.04, 0.04, 0.04],
            # density=((0.02/0.04)**3),
            density=1000,
            #density=0.5,
            #size_min=[0.020, 0.020, 0.020],  # [0.015, 0.015, 0.015],
            #size_max=[0.022, 0.022, 0.022],  # [0.018, 0.018, 0.018])
            rgba=[1, 0, 0, 0],
            # solref=[-5000.0, -100.0],
            material=materials[3],
        )

        self.ball_object = BallObject(
            name="ball",
            size=[0.07],
            friction=[0.5, 0.005, 0.0001],
            # density=((0.02 / 0.06) ** 3),
            density=1000,
            # density=0.5,
            # size_min=[0.020, 0.020, 0.020],  # [0.015, 0.015, 0.015],
            # size_max=[0.022, 0.022, 0.022],  # [0.018, 0.018, 0.018])
            rgba=[1, 0, 0, 0],
            # solref=[-5000.0, -100.0],
            material=materials[1],
        )

        self.cylinder_object = CylinderObject(
            name="cylinder",
            size=[0.06, 0.06],
            # density=((0.04 / 0.06) ** 3),
            density=1000,
            # density=0.5,
            # size_min=[0.020, 0.020, 0.020],  # [0.015, 0.015, 0.015],
            # size_max=[0.022, 0.022, 0.022],  # [0.018, 0.018, 0.018])
            friction=[0.7, 0.005, 0.0001],
            rgba=[1, 0, 0, 0],
            # solref=[-5000.0, -100.0],
            material=materials[2],
        )

        self.objects = [self.box_object, self.box_object2, self.cylinder_object, self.ball_object]

        # Create placement initializer
        if self.placement_initializer is not None:
            #print('Placement initializer specified')
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.objects)
        else:
            print('Using default placement initializer')
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.objects,
                x_range=[-0.1, 0.3],
                y_range=[-0.15, 0.15],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        # self.object_body_id = self.sim.model.body_name2id(self.object.root_body)
        self.object_body_ids = [self.sim.model.body_name2id(obj.root_body) for obj in self.objects]

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # cube-related observables
            @sensor(modality=modality)
            def cube_pos(obs_cache):
                return np.concatenate([np.array(self.sim.data.body_xpos[id]) for id in self.object_body_ids])

            @sensor(modality=modality)
            def cube_quat(obs_cache):
                return np.concatenate([convert_quat(np.array(self.sim.data.body_xquat[id]), to="xyzw") for id in self.object_body_ids])

            # @sensor(modality=modality)
            # def gripper_to_cube_pos(obs_cache):
            #     return (
            #         obs_cache[f"{pf}eef_pos"] - obs_cache["cube_pos"]
            #         if f"{pf}eef_pos" in obs_cache and "cube_pos" in obs_cache
            #         else np.zeros(3)
            #     )

            sensors = [cube_pos, cube_quat]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.objects[0])

    def _check_success(self):
        """
        Check if cube has been pushed to center

        Returns:
            bool: True if cube has been lifted
        """
        if self.reward_function == 'push_center':
            object_distances = []
            for obj_id in self.object_body_ids:
                cube_xy = self.sim.data.body_xpos[obj_id][:2]
                table_xy = self.model.mujoco_arena.table_offset[:2]
                target = table_xy.copy()
                target[0] += 0.2
                # cube is higher than the table top above a margin
                center_dist = np.linalg.norm(cube_xy - target)
                object_distances.append(center_dist)
            object_distances = np.array(object_distances)
            return object_distances.min() < 0.05
        elif self.reward_function == 'tip_cylinder':
            return self.get_cylinder_verticality() < 0.05
