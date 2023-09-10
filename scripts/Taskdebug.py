#from omni.isaac.kit import SimulationApp

#simulation_app = SimulationApp({"headless": False})
#from omni.isaac.core import World

from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
env = VecEnvRLGames(headless=False, #sim_device=cfg.device_id,
                        )
import omni

from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.utils.torch.rotations import *

import numpy as np

from omni.isaac.debug_draw import _debug_draw
draw = _debug_draw.acquire_debug_draw_interface()
import hydra
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
from tasks.BendTask import BendTask
from tasks.PlaceTask import PlaceTask
from tasks.PullTask import PullTask
from tasks.LiftTask import LiftTask
from tasks.TransportTask import TransportTask


def get_world_point(prim,):
    pose = omni.usd.utils.get_world_transform_matrix(prim)
    rot = pose.ExtractRotationMatrix()
    trans = np.array(pose.ExtractTranslation())

    points = np.array(prim.GetAttribute('physxDeformable:simulationPoints').Get())@rot + trans
    ee = np.array(points[-4:]).mean(axis=0)


    return points,ee




@hydra.main(config_name="config", config_path="../cfg")
def main(cfg: DictConfig):
    cfg_dict = omegaconf_to_dict(cfg)

    headless = cfg.headless
    render = not headless
    #env = VecEnvRLGames(headless=headless, sim_device=cfg.device_id,
    #                    )
    sim_config = SimConfig(cfg_dict)
    task = LiftTask(name='spot',sim_config=sim_config,env=env)

    env.set_task(task=task, sim_params=sim_config.get_physics_params(), backend="torch", init_sim=True)

#getBypassRenderSkelMeshProcessing
    while env._simulation_app.is_running():
        if env._world.is_playing():
            if env._world.current_time_step_index == 0:
                env._world.reset(soft=True)
            env._world.step(render=render)
            env.sim_frame_count += 1
            env._task.post_physics_step()
            '''
            ee_list=[]


            draw.clear_points()
            prims =task._robot._def.prims


            point_list,ee1 = get_world_point(prims[0])
            po_list, ee2 = get_world_point(prims[1])
            ee_list.append(ee1)
            ee_list.append(ee2)

            #task.deformAPI_List[0].GetSimulationPointsAttr().Get())
            point = np.array(task.deformAPI_List[0].GetSimulationPointsAttr().Get())@rot + pos

            pos1,rot =task._robot._rod.get_world_poses() # almost same

            #po_list.append(np.array(pos1[0]))

            pos2, rot = task._robot._def.get_world_poses() # middle point

            po_list.append(point[-1])
            po_list.append(point[-2])
            po_list.append(point[-3])
            po_list.append(point[-4])
            po_list.append(np.array(point_list[-4:]).mean(axis=0))

            point3 = po_list
            color3 = [(0, 0, 1, 1) for _ in range(len(point3))]
            sizes3 = [5 for _ in range(len(point3))]
            draw.draw_points(point3, color3, sizes3)

            draw.clear_points()
            point1 = task.point1[0]
            color1 = [(1, 0, 0, 1) for _ in range(len(point1))]
            sizes1 = [5 for _ in range(len(point1))]
            draw.draw_points(point1, color1, sizes1)

            point2 = task.point1[1]
            color2 = [(1, 1, 0, 1) for _ in range(len(point2))]
            sizes2 = [5 for _ in range(len(point2))]
            draw.draw_points(point2, color2, sizes2)
            
            '''


    env._simulation_app.close()



if __name__ == "__main__":
 main()