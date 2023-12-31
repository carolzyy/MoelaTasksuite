from omni.isaac.core.utils.stage import add_reference_to_stage,get_current_stage
from Utils.RidgeFranka import RidgeFranka
from Utils.Spot import Spot
import os
import omni
import math
import numpy as np
from omni.isaac.core.prims import RigidPrimView, XFormPrim
from pxr import UsdGeom, UsdLux, Sdf, Gf, Vt, Usd, UsdPhysics,PhysxSchema,PhysicsSchemaTools
root_path=os.getcwd()

def get_robot(robot_name,path,robot_position,max_velocity=None):
    if robot_name == 'franka_rod':
        robot = RidgeFranka(path, position=robot_position, usd_path=root_path + '/Asset/RidgebackFranka/ridgeback_franka_rod.usd')
    elif robot_name == 'franka':
        robot = RidgeFranka(path, position=robot_position,
                            usd_path=root_path + '/Asset/ridgeback_franka.usd')
    elif robot_name == 'franka_long':
        max_velocity = [4000, 4000] + [math.degrees(x)/3 for x in [2.175, 2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]]
        robot = RidgeFranka(path, position=robot_position, usd_path=root_path + '/Asset/RidgebackFranka/ridgeback_franka_rod_long.usd',max_velocity=max_velocity)
    elif robot_name == 'franka_drag':
        robot = RidgeFranka(path, position=robot_position, usd_path=root_path + '/Asset/franka_drag.usd')
    elif robot_name == 'spot_rod':
        robot = Spot(path , position=robot_position, usd_path=root_path + '/Asset/spot_rod.usd')
    elif robot_name == 'spot_drag':
        robot = Spot(path, position=robot_position, usd_path=root_path + '/Asset/spot_drag.usd')
    elif robot_name == 'spot':
        robot = Spot(path, position=robot_position, usd_path=root_path + '/Asset/spot.usd')


def get_table(table_position,prim):
    table_usd_path = root_path+'/Asset/Table.usd'
    add_reference_to_stage(usd_path=table_usd_path, prim_path=prim)

    tables = XFormPrim(
            prim_path=prim,
            name="table",
            translation=table_position,
            scale=[0.01],# maybe need trible
        )

def get_wall(wall_position,prim):
    wall_usd_path = root_path + '/Asset/L-walls.usd'
    add_reference_to_stage(usd_path=wall_usd_path, prim_path=prim)

    walls = XFormPrim(
        prim_path=prim,
        name="L-wall",
        translation=wall_position,
        #scale=[0.1],  # maybe need trible
    )


def get_obstacle(ob_position,prim):
    wall_usd_path = root_path + '/Asset/Obstacle_ob.usd'
    add_reference_to_stage(usd_path=wall_usd_path, prim_path=prim)

    obstacle = XFormPrim(
        prim_path=prim,
        name="obstacle",
        translation=ob_position,
        scale=Gf.Vec3d(0.9, 0.9, 1.5)
    )

def get_door(position,prim):
    usd_path = root_path + '/Asset/Door.usd'
    add_reference_to_stage(usd_path=usd_path, prim_path=prim)

    door = XFormPrim(
        prim_path=prim,
        name="door",
        translation=position,
    )


def initialize_task(config, env, init_sim=True):
    from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
    sim_config = SimConfig(config)

    from tasks.PlaceTask import PlaceTask
    from tasks.BendTask import BendTask
    from tasks.TransportTask import TransportTask
    from tasks.LiftTask import LiftTask
    from tasks.DragTask import DragTask

    # Mappings from strings to environments
    task_map = {
        "Place": PlaceTask,
        "Bend": BendTask,
        "Lift": LiftTask,
        "Drag": DragTask,
        "Transport": TransportTask,
    }

    cfg = sim_config.config
    task = task_map[cfg["task_name"]](
        name=cfg["task_name"], sim_config=sim_config, env=env
    )

    env.set_task(task=task, sim_params=sim_config.get_physics_params(), backend="torch", init_sim=init_sim)

    return task

def get_world_point(prim,):
    pose = omni.usd.utils.get_world_transform_matrix(prim)
    rot = pose.ExtractRotationMatrix()
    trans = np.array(pose.ExtractTranslation())

    points = np.array(prim.GetAttribute('physxDeformable:simulationPoints').Get())@rot + trans
    ee = np.array(points[-4:]).mean(axis=0)


    return points,ee

