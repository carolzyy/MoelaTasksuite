

from pxr import  Gf, PhysxSchema,UsdGeom
from omni.isaac.core.prims import RigidPrimView, XFormPrimView,GeometryPrimView,XFormPrim
from omni.isaac.core.utils.stage import add_reference_to_stage,get_current_stage
from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.utils.torch.rotations import *
from omniisaacgymenvs.tasks.base.rl_task import RLTask

import numpy as np
import torch


from Utils.TaskUtils import get_robot,get_table,get_world_point
from Utils.FrankaView import FrankaView

#from omni.isaac.debug_draw import _debug_draw
#draw = _debug_draw.acquire_debug_draw_interface()
import hydra
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
class PlaceTask( RLTask):
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
        #self.action_scale = self._task_cfg["env"]["actionScale"]

        self.robot_position = Gf.Vec3d(0,0,0.0)
        self.table_height = 0.83

        self._num_observations = 95
        self._num_actions = 10   #with the fingers


        self.table_position = Gf.Vec3d(1.5, 1.5, 0.0)
        self.robot_name = name

        RLTask.__init__(self, name, env)

        return

    def set_up_scene(self, scene) -> None:
        self._stage = get_current_stage()
        get_robot(self.robot_name,self.default_zero_env_path + "/Robot",
                  self.robot_position)#add reference
        get_table(table_position = self.table_position,
                  prim = self.default_zero_env_path+ "/Table")
        replicate_physics = False
        super().set_up_scene(scene, replicate_physics)

        self._robot = FrankaView(prim_paths_expr="/World/envs/.*/Robot")
        scene.add(self._robot)
        #scene.add(self._robot._ee)
        scene.add(self._robot._def)

        self._table = XFormPrimView(prim_paths_expr="/World/envs/.*/Table",
                                    name="table_view",
                                    reset_xform_properties=False
                                )
        scene.add(self._table)

        return

    #In this method, we can implement logic that gets executed once the scene is constructed and simulation starts running.
    def post_reset(self) -> None:
        #randomize all envs, maybe train several env one time
        # env data initial
        name = 'Franka'
        if name == 'Franka':
            self.robot_default_dof = torch.tensor(
                [0.0, 0.0, 0.0,
                 1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469,
                ], device=self._device
            )

        dof_limits=self._robot.get_dof_limits() #[env_num,dof_num]
        self.robot_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)# dof limitation list,(12,)
        self.robot_dof_lower_limits[:2] = 0
        self.robot_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.robot_dof_upper_limits[:2] = 2
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_upper_limits)*2
        self.robot_dof_speed_scales[:3] = 1


        self.robot_dof_targets = torch.zeros(
            (self._num_envs, self._robot.num_dof), dtype=torch.float, device=self._device
        )
        self.robot_dof_pos = torch.zeros(
            (self.num_envs, self._robot.num_dof), device=self._device)
        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)
        table_pos,_ = self._table.get_world_poses()
        table_pos[:,-1] = self.table_height
        self.targets = table_pos

        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    # reset the environment and set a random joint action,set our environment into an initial state for starting a new training episode.
    # should be random initial pose, however all zero now
    #need to debug
    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        pos = tensor_clamp(
            self.robot_default_dof.unsqueeze(0),
            #+ 0.25 * (torch.rand((len(env_ids), self._robot.num_dof), device=self._device) - 0.5),
            self.robot_dof_lower_limits,
            self.robot_dof_upper_limits,
        )
        dof_pos = torch.zeros((num_indices, self._robot.num_dof), device=self._device)
        dof_vel = torch.zeros((num_indices, self._robot.num_dof), device=self._device)
        dof_pos[:, :] = pos
        self.robot_dof_targets[env_ids, :] = pos
        self.robot_dof_pos[env_ids, :] = pos


        self._robot.set_joint_positions(dof_pos, indices=indices) # set to default
        self._robot.set_joint_velocities(dof_vel, indices=indices) # set to zero
        self._robot.set_joint_position_targets(self.robot_dof_targets[env_ids], indices=indices)


        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0


    def pre_physics_step(self, actions)-> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.actions = actions.clone().to(self._device)
        #targets = self.robot_dof_targets + self.robot_dof_speed_scales * self._dt * self.actions * self.action_scale#### maybe need to change
        #targets = self.robot_dof_speed_scales * self.actions
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self._dt * self.actions
        self.robot_dof_targets[:] = tensor_clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

        env_ids_int32 = torch.arange(self._robot.count, dtype=torch.int32, device=self._device)
        self._robot.set_joint_position_targets(self.robot_dof_targets, indices=env_ids_int32)

    # collect observation info
    # need to debug
    def get_observations(self, ):

        robot_dof_pos = self._robot.get_joint_positions(clone=False)
        robot_dof_poses =  robot_dof_pos.reshape(self._num_envs, -1).to(dtype=torch.float)

        robot_dof_vel = self._robot.get_joint_velocities(clone=False)
        robot_dof_vels = robot_dof_vel.reshape(self._num_envs, -1).to(dtype=torch.float)

        rod_ele_pos=[]
        rod_ee_pos=[]
        for prim in self._robot._def.prims:
            ele,ee = get_world_point(prim)
            rod_ele_pos.append(ele)
            rod_ee_pos.append(ee)
        rod_ele_pos = np.array(rod_ele_pos).reshape(self._num_envs,-1)
        ele_pos = torch.tensor(rod_ele_pos,dtype=torch.float)#num*72

        rod_ee_pos = np.array(rod_ee_pos).reshape(self._num_envs, -1)
        ee_pos = torch.tensor(rod_ee_pos,dtype=torch.float)#2*3
        self.ee_pos = ee_pos


        self.obs_buf = torch.cat(
                                 (
                                   ele_pos,
                                   self.targets,
                                   robot_dof_poses,
                                   robot_dof_vels,
                                 ),
                                   axis = -1)

#####must return the obs or there would be error
        observations = {
            'PlaceTask': {
                "obs_buf": self.obs_buf
            }
        }
        return observations


    def calculate_metrics(self) -> None:

        rod_mid_position,_rot = self._robot._def.get_world_poses()
        rod_mid_table_dis = torch.sqrt(torch.sum((rod_mid_position-self.targets)**2, dim=-1))

        rod_ee_position = self.ee_pos
        rod_ee_table_dis = torch.sqrt(torch.sum((rod_ee_position - self.targets) ** 2, dim=-1))

        robot_ee_position,_rot = self._robot._ee.get_world_poses()
        robot_ee_table_dis = torch.sqrt(torch.sum((robot_ee_position - self.targets) ** 2, dim=-1))

        rewards = - rod_ee_table_dis * 2
        rewards = torch.where( robot_ee_table_dis > rod_ee_table_dis, rewards + (robot_ee_table_dis-rod_ee_table_dis), rewards )
        rewards =torch.where(
                rod_ee_position[:, -1] > 0.8,
            torch.where( rod_mid_position [:, -1] > 0.8, rewards + (rod_mid_table_dis-rod_ee_table_dis)*2, rewards )
            , rewards)

        action_penalty = 0.01*torch.sum(self.actions **2, dim = -1)
        rewards -= action_penalty

        self.flag = self.on_table()
        rewards = torch.where( self.flag, rewards + 20, rewards ) #  not sure, need to test change on 902

        #rewards = torch.where(ee_position[:, -1] > 0.8, rewards + 0.25, rewards)
        #rewards = torch.where(robot_ee_position[:, -1] > 0.8, rewards + 0.25, rewards)
        self.rew_buf[:] = rewards


    # know if it finish or not
    def is_done(self) -> None:

        resets = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)
        resets = torch.where(self.flag, torch.ones_like(self.reset_buf), resets)
        # or reward >12
        self.reset_buf[:] = resets

    def on_table(self):
        table_pos = self.targets
        ee_pos = self.ee_pos

        table_limit_x_low = table_pos[:,0]-0.56
        table_limit_x_upp= table_pos[:,0]+0.56
        x_flag = (ee_pos[:,0]>table_limit_x_low) & (table_limit_x_upp>ee_pos[:,0])

        table_limit_y_low = table_pos[:, 1] - 0.4
        table_limit_y_upp = table_pos[:, 1] + 0.4
        y_flag = (ee_pos[:,1]>table_limit_y_low) & (table_limit_y_upp>ee_pos[:,1])

        table_limit_z_low = table_pos[:, -1]
        table_limit_z_upp = table_pos[:, -1] + 0.15
        z_flag = (ee_pos[:,-1]>table_limit_z_low) & (table_limit_z_upp>ee_pos[:,-1])

        flag = x_flag & y_flag & z_flag
        if flag.any() == True:
            print('done')

        return flag
