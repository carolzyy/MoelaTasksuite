from abc import abstractmethod, ABC
from typing import Optional

from omni.isaac.kit import SimulationApp

#simulation_app = SimulationApp({"headless": False})

from omni.isaac.gym.vec_env import VecEnvBase
env = VecEnvBase(headless=False)

import carb
import omni
import omni.usd as usd
import gym
from omni.isaac.core.tasks import BaseTask
from omni.isaac.manipulators import SingleManipulator
#from omni.isaac.manipulators.grippers import SingleGripper
from omni.isaac.core.utils.stage import add_reference_to_stage,get_current_stage,open_stage
import omni.physx as _physx
from omni.physx import get_physx_scene_query_interface,get_physx_interface, get_physx_simulation_interface
from omni.isaac.core import World
from omni.isaac.core.scenes.scene import Scene
from pxr import UsdGeom, UsdLux, Sdf, Gf, Vt, Usd, UsdPhysics,PhysxSchema,PhysicsSchemaTools
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.motion_generation.lula import RmpFlow
from omni.isaac.motion_generation import ArticulationMotionPolicy
from omni.isaac.core.objects import cuboid, DynamicCuboid, VisualCuboid
from omni.isaac.core.prims import GeometryPrim, XFormPrim
from omni.isaac.core.utils.types import ArticulationAction
import numpy as np
import torch
from omni.physx.scripts import physicsUtils, deformableUtils
from gym import spaces

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from Utils.LegSpot import LegSpot
from Utils.RidgeFranka import RidgeFranka
class LiftTask( RLTask):
    def __init__(
            self,
            name,
            #env_num=1,
            #repeat_act=3,
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
        self.repeat_act = repeat_act

        #RL params
        self._num_observations = 32
        self._num_actions = 10
        self.num_envs = env_num
        self._deform_api = None

        self.robot_position = Gf.Vec3d(0, 0, 0.0)
        self.rod_ee = torch.tensor([[0, 0, 0.0]])

        self.rod_tar = np.array([1.75, 0.5, 0.8])
        self.robot_tar = np.array([0, 1, 0.0])

        self.obs = np.zeros((self.num_envs, self._num_observations))
        self.resets = torch.zeros((self.num_envs, 1))
        self.reward = torch.ones((self.num_envs, 1))

        # set the action and observation space for RL
        self.action_space = spaces.Box(np.ones(self._num_actions) * -1,
                                       np.ones(self._num_actions) * 1
                                       )

        self.observation_space = spaces.Box(np.ones(self._num_observations) * -np.Inf,
                                            np.ones(self._num_observations) * np.Inf)

        RLTask.__init__(self, name=name, env)

        return

    def set_up_scene(self, scene: Scene) -> None:
        super().set_up_scene(scene)
        self._stage = get_current_stage()
        scene.add_default_ground_plane()

        self.robot = scene.add(RidgeFranka(position=self.robot_position))
        self._rod_mesh = self.set_rod()

        self.set_initial_camera_params(camera_position=[6,0,1])

        return

    def set_initial_camera_params(self, camera_position=[1.5, 1.5, 1.5], camera_target=[0, 0, 0]):# defaul params
        set_camera_view(eye=camera_position, target=camera_target, camera_prim_path="/OmniverseKit_Persp")


    def set_rod(self,left_position = Gf.Vec3f(2.0, -0.6, 0.8), right_position=Gf.Vec3f(2.0, 0.6, 0.8)):
        env_path = '/World/env/'
        deform_mater_path = '/World/Physics_Materials/deform_material'

        rod_mesh = physicsUtils.create_mesh_cube(self._stage, env_path + 'rod', 0.05, 1.2)
        rod_mesh.AddTranslateOp().Set(Gf.Vec3f(2, -0.6, 0.6))
        rod_mesh.AddRotateZOp().Set(90)

        cuboid.VisualCuboid("/World/env/rod_target", position=np.array([2, 0, 0.8]), color=np.array([0, 0, 1.0]),size=0.1)
        cuboid.VisualCuboid("/World/env/robot_target", position=np.array([4, 0, 0.1]), color=np.array([0., 1.0, 0]), size=0.2)


        cube_left_prim = physicsUtils.add_rigid_box(self._stage,
                                                    env_path + 'cube_left',
                                                    size=Gf.Vec3f(3, 0.1, 1.6),
                                                    position=left_position,
                                                    density=0.0)

        cube_right_prim = physicsUtils.add_rigid_box(self._stage,
                                                     env_path + 'cube_right',
                                                     size=Gf.Vec3f(3, 0.1, 1.6),
                                                     position=right_position,
                                                     density=0.0)


        deformableUtils.add_physx_deformable_body(self._stage,
                                                               rod_mesh.GetPath(),
                                                               collision_simplification=True,
                                                               simulation_hexahedral_resolution=5,
                                                               self_collision=False,)


        deformableUtils.add_deformable_body_material(self._stage,
                                                     deform_mater_path,
                                                     youngs_modulus=3.0e12,
                                                     )

        physicsUtils.add_physics_material_to_prim(self._stage, rod_mesh.GetPrim(), deform_mater_path)

        # set robot attachment, target 1 is rod, target 2 is arm_ee
        attachment_cube_right_path = rod_mesh.GetPath().AppendElementString('attach_cube_right')
        attachment_cube_right = PhysxSchema.PhysxPhysicsAttachment.Define(self._stage, attachment_cube_right_path)
        attachment_cube_right.GetActor0Rel().SetTargets([rod_mesh.GetPath()])
        attachment_cube_right.GetActor1Rel().SetTargets([cube_right_prim.GetPath()])
        attach_api_cube_right = PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment_cube_right.GetPrim())
        attach_api_cube_right.CreateEnableRigidSurfaceAttachmentsAttr(True)

        # set cube attachment, target 1 is rod, target 2 is arm_ee
        attachment_cube_left_path = rod_mesh.GetPath().AppendElementString('attach_cube_left')
        attachment_cube_left = PhysxSchema.PhysxPhysicsAttachment.Define(self._stage, attachment_cube_left_path)
        attachment_cube_left.GetActor0Rel().SetTargets([rod_mesh.GetPath()])
        attachment_cube_left.GetActor1Rel().SetTargets([cube_left_prim.GetPath()])
        attach_api_cube_left = PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment_cube_left.GetPrim())
        attach_api_cube_left.CreateEnableRigidSurfaceAttachmentsAttr(True)

        return rod_mesh


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
        num_resets = len(env_ids)
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
        #rod_elem_pos = self._deform_api.GetSimulationPointsAttr().Get()
        # rod_elem_pos1 = self._deform_api.GetCollisionPointsAttr().Get()

        rod_pos = usd.utils.get_world_transform_matrix(self._rod_mesh.GetPrim())
        rod_tran = Gf.Vec3f(rod_pos.ExtractTranslation())
        self.rod_position = np.array(rod_tran)


        self.robot.update()

        self.obs = np.concatenate(
            (self.rod_position.reshape(self.num_envs, -1),
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


        self.reward = torch.where(0.3> rod_dis, 300, -rod_dis)
        self.reward = torch.where(0.2> robot_dis, self.reward +300, self.reward-robot_dis)


        return self.reward.item()

    # know if it finish or not
    def is_done(self) -> bool:
        resets = torch.where(self.reward > 0, 1, 0)
        self.resets = resets

        return resets.item()

'''

def main():
    rod_task = RidgeSpot_Lift()
    world = World(stage_units_in_meters=1.0)
    world.add_task(rod_task)
    world.reset()




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



'''
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

task = RidgeSpot_Lift()

env.set_task(task,
             backend="torch",
             init_sim=True,
             physics_dt=1.0 / 60.0,
             rendering_dt=1.0 / 60.0,
             )
log_dir = "./FrankaLift"


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
