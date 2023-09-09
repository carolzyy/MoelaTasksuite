from abc import abstractmethod, ABC
from typing import Optional

from omni.isaac.kit import SimulationApp

#simulation_app = SimulationApp({"headless": False})

from omni.isaac.gym.vec_env import VecEnvBase
#env = VecEnvBase(headless=False)


import omni.usd as usd
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.stage import add_reference_to_stage,get_current_stage,open_stage
import omni.physx as _physx
from omni.physx import get_physx_scene_query_interface,get_physx_interface, get_physx_simulation_interface
from omni.isaac.core import World
from omni.isaac.core.scenes.scene import Scene
from pxr import UsdGeom, UsdLux, Sdf, Gf, Vt, Usd, UsdPhysics,PhysxSchema,PhysicsSchemaTools
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.objects import cuboid

import numpy as np
import torch
from Utils.RidgeFranka import RidgeFranka
from omni.physx.scripts import physicsUtils, deformableUtils
from gym import spaces
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from Utils.LegSpot import LegSpot

class PullTask( RLTask):
    def __init__(
            self,
            name,
            #env_num=1,
            #repeat_act=3,
            #reduce=False,
            sim_config,
            env,
            offset=None
    )-> None:

        _physx.get_physx_interface().overwrite_gpu_setting(1)
        self._device = torch.cuda.get_device_name()
        print('cude device is:', self._device)
        # task specific
        self._robot = None
        self._stage = None
        self._cube_prim = None
        self.reduce = reduce
        self.repeat_act = repeat_act

        #RL params
        self._num_observations = 98
        if self.reduce:
            self._num_observations = 50
        self._num_actions = 10
        self.num_envs = env_num
        self._deform_api = None

        self.robot_position = Gf.Vec3d(0, 0, 0.0)
        self.rod_ee = torch.tensor([[0, 0, 0.0]])

        self.rod_tar = np.array([1.75, 0.5, 0.8])
        self.robot_tar = np.array([0.1, 0.8, 0.0])

        self.obs = np.zeros((self.num_envs, self._num_observations))
        self.resets = torch.zeros((self.num_envs, 1))

        # set the action and observation space for RL
        self.action_space = spaces.Box(np.ones(self._num_actions) * -1,
                                       np.ones(self._num_actions) * 1
                                       )

        self.observation_space = spaces.Box(np.ones(self._num_observations) * -np.Inf,
                                            np.ones(self._num_observations) * np.Inf)

        RLTask.__init__(self, name, env)

        return
#TDL: get ee position and set to rod
    def set_up_scene(self, scene: Scene) -> None:
        super().set_up_scene(scene)
        self._stage = get_current_stage()
        scene.add_default_ground_plane()

        self.robot = scene.add(RidgeFranka(position=self.robot_position))
        self._rod_mesh = self.set_rod()
        self.set_cube()

        self.set_initial_camera_params(camera_position=[3,3,3])

        return

    def set_initial_camera_params(self, camera_position=[1.5, 1.5, 1.5], camera_target=[0, 0, 0]):# defaul params
        set_camera_view(eye=camera_position, target=camera_target, camera_prim_path="/OmniverseKit_Persp")


    def set_rod(self):
        rod_path = '/World/env/rod'
        deform_mater_path = '/World/Physics_Materials/deform_material'
        robot_ee_path = "/World/env/robot/panda_rightfinger/collisions"
        cube_path = '/World/env/cube'

        rod_mesh = physicsUtils.create_mesh_cube(self._stage,  rod_path, 0.05, 1.5)
        rod_mesh.AddTranslateOp().Set(Gf.Vec3f(0.95, 0.051, 0.8))


        deform_api = deformableUtils.add_physx_deformable_body(self._stage,
                                                               rod_mesh.GetPath(),
                                                               collision_simplification=True,
                                                               simulation_hexahedral_resolution=5,
                                                               self_collision=False,)


        deformableUtils.add_deformable_body_material(self._stage,
                                                     deform_mater_path,
                                                     youngs_modulus=3.0e12,
                                                     )
        physicsUtils.add_physics_material_to_prim(self._stage, rod_mesh.GetPrim(), deform_mater_path)


        #set robot attachment, target 1 is rod, target 2 is arm_ee
        attachment_robot_path = rod_mesh.GetPath().AppendElementString('attach_robot')
        attachment_robot = PhysxSchema.PhysxPhysicsAttachment.Define(self._stage, attachment_robot_path)
        attachment_robot.GetActor0Rel().SetTargets([rod_path])
        attachment_robot.GetActor1Rel().SetTargets([robot_ee_path])
        attach_api_robot = PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment_robot.GetPrim())
        attach_api_robot.CreateEnableRigidSurfaceAttachmentsAttr(True)

        # set cube attachment, target 1 is rod, target 2 is arm_ee
        attachment_cube_path = rod_mesh.GetPath().AppendElementString('attach_cube')
        attachment_cube = PhysxSchema.PhysxPhysicsAttachment.Define(self._stage, attachment_cube_path)
        attachment_cube.GetActor0Rel().SetTargets([rod_path])
        attachment_cube.GetActor1Rel().SetTargets([cube_path])
        attach_api_cube = PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment_cube.GetPrim())
        attach_api_cube.CreateEnableRigidSurfaceAttachmentsAttr(True)

        self._deform_api = deform_api

        return rod_mesh



    def set_cube(self, cube_position=Gf.Vec3f(2.4, 0.051, 0.83),obstacle_position =Gf.Vec3f(1.75, 0.2, 0.8)):
        self.cube_prim = physicsUtils.add_rigid_box(self._stage,
                                                    '/World/env/cube',
                                                    size=Gf.Vec3f(0.15),
                                                    position=cube_position,
                                                    density=0.0)

        self.cylin_prim = physicsUtils.add_rigid_cylinder(
                                                    self._stage,
                                                    '/World/env/cylin',
                                                    radius=0.1,
                                                    height=0.4,
                                                    position=obstacle_position,
                                                    density=0.0)

        cuboid.VisualCuboid("/World/env/rod_target",
                            position=self.rod_tar,
                            color=np.array([0, 1.0, 0]), size=0.1)

        cuboid.VisualCuboid("/World/env/robot_target",
                            position=self.robot_tar + np.array([0, 0, 0.15]),
                            color=np.array([1., 0, 0]), size=0.2)



    #In this method, we can implement logic that gets executed once the scene is constructed and simulation starts running.
    def post_reset(self) -> None:
        #randomize all envs, maybe train several env one time
        indices = torch.arange(self.num_envs)
        self.reset(indices)

    # reset the environment and set a random joint action,set our environment into an initial state for starting a new training episode.
    # should be random initial pose, however all zero now
    def reset(self,env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)
        # num_resets = len(env_ids)
        self.robot.initialize()

        self.resets[env_ids] = 0


    # deal with action and also
    #This method will be called from VecEnvBase before each simulation step
    def pre_physics_step(self, actions)-> None:
        reset_env_ids = self.resets.nonzero().squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset(reset_env_ids)
        self.robot.apply_dofaction(actions, self.repeat_act)

    # collect observation info
    def get_observations(self):
        # related element node position, 44 points
        rod_elem_pos = self._deform_api.GetSimulationPointsAttr().Get()
        if self.reduce == True:
            rod_elem_pos = rod_elem_pos[::3]

        rod_pos = usd.utils.get_world_transform_matrix(self._rod_mesh.GetPrim())
        rod_tran = Gf.Vec3f(rod_pos.ExtractTranslation())
        self.rod_position = rod_tran



        self.rod_ee = torch.tensor(np.sort(np.array(rod_elem_pos), axis=0)[-4:] + rod_tran).mean(0, True)
        self.robot.update()

        self.obs = np.concatenate(
            (np.array(rod_elem_pos).reshape(self.num_envs, -1),
             self.rod_tar.reshape(self.num_envs, -1),
             self.robot_tar.reshape(self.num_envs, -1),
             #self.robot._state.baseframe_pos.reshape(self.num_envs, -1),
             self.robot._state.joint_pos[:self._num_actions].reshape(self.num_envs, -1),
             self.robot._state.joint_vel[:self._num_actions].reshape(self.num_envs, -1)),
            axis=-1)
        #####must return the obs or there would be error
        return self.obs

    
    def calculate_metrics(self) -> None:
        rod_pos = torch.tensor(self.rod_position)
        rod_target = torch.tensor(self.rod_tar)
        rod_dis = torch.sqrt(torch.sum((rod_pos - rod_target) ** 2, dim=-1, keepdim=True)).unsqueeze(-1)

        robot_pos = torch.tensor(self.robot._state.baseframe_pos)
        robot_target = torch.tensor(self.robot_tar)
        robot_dis = torch.sqrt(torch.sum((robot_pos - robot_target) ** 2, dim=-1, keepdim=True)).unsqueeze(-1)

        self.reward = torch.where(0.2 > rod_dis, 5, -rod_dis)
        self.reward = torch.where(0.2 > robot_dis, self.reward + 5, self.reward - robot_dis)

        return self.reward.item()

    # know if it finish or not
    def is_done(self) -> bool:
        resets = torch.where(self.reward > 0, 1, 0)
        self.resets = resets

        return resets.item()

'''

def main():
    rod_task = RidgeSpot_Pull()
    world = World(stage_units_in_meters=1.0)
    world.add_task(rod_task)
    world.reset()

    #draw = _debug_draw.acquire_debug_draw_interface()


#getBypassRenderSkelMeshProcessing
    while simulation_app.is_running():
        world.step(render=True)
        #i = i+1
        if world.is_playing():
            if world.current_time_step_index == 0:
                world.reset()


    simulation_app.close()


if __name__ == "__main__":
    main()



from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

task = RidgeSpot_Pull()

env.set_task(task,
             backend="torch",
             init_sim=True,
             physics_dt=1.0 / 60.0,
             rendering_dt=1.0 / 60.0,
             )
log_dir = "./FrankaPull"


#checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=log_dir, name_prefix="Ridgefranka_ppo_checkpoint")
# create agent from stable baselines
model = PPO(
        "MlpPolicy",
        env,
        n_steps=4096,
        batch_size=128,
        n_epochs=20,
        learning_rate=0.0005,
        gamma=0.9,
        device="cuda:0",
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=1.0,
        verbose=1,
        tensorboard_log=log_dir
)
model.learn(total_timesteps=40000)
model.save(log_dir+"/ppo_franka0")
print('learning finish')

env.close()
'''