from pxr import  Gf, PhysxSchema,UsdGeom
from omni.isaac.core.prims import RigidPrimView, XFormPrimView,GeometryPrimView,XFormPrim
from omni.isaac.core.utils.stage import add_reference_to_stage,get_current_stage
from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.utils.torch.rotations import *
from omniisaacgymenvs.tasks.base.rl_task import RLTask

import numpy as np
import torch
from omni.isaac.core.objects import cuboid, DynamicCuboid, VisualCuboid
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from Utils.TaskUtils import get_robot,get_door,get_world_point
from Utils.FrankaView import FrankaView


class DragTask( RLTask):
    def __init__(
            self,
            name,
            sim_config,
            env,
            offset=None
    )-> None:

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._dt = self._task_cfg["sim"]["dt"]

        self._max_episode_length = self._task_cfg["env"]["episodeLength"]
        self.robot_name = 'franka_lift'
        self.reduce = self.reduce = self._task_cfg["env"]["reduce"]

        #RL params

        self._num_actions = 10
        if self.reduce:
            self._num_observations = 36
        else:
            self._num_observations = 51

        self.robot_position = Gf.Vec3d(0, 0, 0.0)

        self.belt_tar = np.array([1.5, -0.3, 0.8])
        self.robot_tar = np.array([0.5, -0.75, 0.1])


        RLTask.__init__(self, name, env)

        return

    def set_up_scene(self, scene) -> None:
        self._stage = get_current_stage()
        get_robot(self.robot_name, self.default_zero_env_path + "/Robot",
                  self.robot_position)  # add reference
        self.add_target()
        replicate_physics = False
        super().set_up_scene(scene, replicate_physics)

        self._robot = FrankaView(prim_paths_expr="/World/envs/.*/Robot",
                                 #name='franka_view'
                                 )
        scene.add(self._robot)

        self._robot_target = XFormPrimView(prim_paths_expr="/World/envs/.*/Belt_Target",
                                           name="target_robot_view",
                                           reset_xform_properties=False
                                           )
        scene.add(self._robot_target)

        self._belt_target = XFormPrimView(prim_paths_expr="/World/envs/.*/Robot_Target",
                                          name="target_belt_view",
                                          reset_xform_properties=False
                                          )
        scene.add(self._belt_target)
        self.set_initial_camera_params(camera_position=[9.7,-8.1,2.3])

        return

    def add_target(self):
        target = VisualCuboid(prim_path=self.default_zero_env_path + "/Belt_Target",
                                     position=self.belt_tar,
                                     color=np.array([0., 1.0, 0]),
                                     size=0.1)

        target = cuboid.VisualCuboid(prim_path=self.default_zero_env_path + "/Robot_Target",
                                     position=self.robot_tar,
                                     color=np.array([1., 0, 0]),
                                     size=0.2)

    #In this method, we can implement logic that gets executed once the scene is constructed and simulation starts running.
    def post_reset(self) -> None:
        #randomize all envs, maybe train several env one time
        if self.robot_name == 'franka_lift':
            self.robot_default_dof = torch.tensor(
                [0.0, 0.0, 0.0,
                 1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469,
                 ], device=self._device
            )

        dof_limits = self._robot.get_dof_limits()
        self.robot_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)  # dof limitation list,(10,)
        self.robot_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_upper_limits) * 2
        self.robot_dof_speed_scales[:2] = 1

        self.robot_dof_targets = torch.zeros(
            (self._num_envs, self._robot.num_dof), dtype=torch.float, device=self._device
        )
        self.robot_dof_pos = torch.zeros(
            (self.num_envs, self._robot.num_dof), device=self._device)

        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)

        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    # reset the environment and set a random joint action,set our environment into an initial state for starting a new training episode.
    # should be random initial pose, however all zero now
    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        pos = tensor_clamp(
            self.robot_default_dof.unsqueeze(0),
            # + 0.25 * (torch.rand((len(env_ids), self._robot.num_dof), device=self._device) - 0.5),
            self.robot_dof_lower_limits,
            self.robot_dof_upper_limits,
        )
        dof_pos = torch.zeros((num_indices, self._robot.num_dof), device=self._device)
        dof_vel = torch.zeros((num_indices, self._robot.num_dof), device=self._device)
        dof_pos[:, :] = pos
        self.robot_dof_targets[env_ids, :] = pos
        self.robot_dof_pos[env_ids, :] = pos

        self._robot.set_joint_positions(dof_pos, indices=indices)  # set to default
        self._robot.set_joint_velocities(dof_vel, indices=indices)  # set to zero
        self._robot.set_joint_position_targets(self.robot_dof_targets[env_ids], indices=indices)

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0



    # deal with action and also
    #This method will be called from VecEnvBase before each simulation step
    def pre_physics_step(self, actions)-> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.actions = actions.clone().to(self._device)
        # targets = self.robot_dof_targets + self.robot_dof_speed_scales * self._dt * self.actions * self.action_scale#### maybe need to change
        # targets = self.robot_dof_speed_scales * self.actions
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self._dt * self.actions
        self.robot_dof_targets[:] = tensor_clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

        env_ids_int32 = torch.arange(self._robot.count, dtype=torch.int32, device=self._device)
        self._robot.set_joint_position_targets(self.robot_dof_targets, indices=env_ids_int32)

    # collect observation info
    def get_observations(self):
        # related element node position, 44 points


        self.belt_target, _ = self._belt_target.get_world_poses()
        self.robot_target, _ = self._robot_target.get_world_poses()

        robot_position, robot_rot = self._robot._base.get_world_poses()
        belt_position, _ = self._robot._def.get_world_poses()

        robot_dof_pos = self._robot.get_joint_positions(clone=False)
        robot_dof_poses = robot_dof_pos.reshape(self._num_envs, -1).to(dtype=torch.float)

        robot_dof_vel = self._robot.get_joint_velocities(clone=False)
        robot_dof_vels = robot_dof_vel.reshape(self._num_envs, -1).to(dtype=torch.float)

        if self.reduce:
            ele_pos = belt_position
        else:
            belt_ele_pos = []
            for prim in self._robot._def.prims:
                ele, _ = get_world_point(prim)  # ele,24 pionts
                belt_ele_pos.append(ele[::4])
            belt_ele_pos = np.array(belt_ele_pos).reshape(self._num_envs, -1)
            ele_pos = torch.tensor(belt_ele_pos, dtype=torch.float)  # num*72


        self.obs_buf = torch.cat(
            (
                ele_pos,
                robot_position,
                robot_rot,
                self.belt_target,
                self.robot_target,
                robot_dof_poses,
                robot_dof_vels,
            ),
            axis=-1)

        #####must return the obs or there would be error
        observations = {
            'PlaceTask': {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def calculate_metrics(self) -> None:
        robot_target = self.robot_target
        belt_target = self.belt_target

        belt_position, _ = self._robot._def.get_world_poses()
        belt_dis = torch.sqrt(torch.sum((belt_position - belt_target) ** 2, dim=-1))

        robot_position, _ = self._robot._base.get_world_poses()
        robot_dis = torch.sqrt(torch.sum((robot_position - robot_target) ** 2, dim=-1))

        dis  = robot_dis + belt_dis
        rewards = -dis
        self.flag = (0.4 > dis)
        if self.flag.any():
            print('done')
        rewards = torch.where(self.flag, rewards + 20, rewards)

        self.rew_buf[:] = rewards

    # know if it finish or not
    def is_done(self) -> bool:
        resets = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf),
                             self.reset_buf)
        resets = torch.where(self.flag, torch.ones_like(self.reset_buf), resets)
        # or reward >12
        self.reset_buf[:] = resets