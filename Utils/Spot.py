import torch
import os
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.prims import get_prim_at_path
from typing import Optional
from omni.isaac.core.utils.stage import add_reference_to_stage
import numpy as np
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive
import carb
import math
from pxr import PhysxSchema
#/home/carol/.local/share/ov/pkg/isaac_sim-2022.1.1/OmniIsaacGymEnvs/omniisaacgymenvs/robots/articulations



class Spot(Robot):
    def __init__(
            self,
            prim_path:str,
            name: str = 'spot',
            usd_path: Optional[str] = None,
            position: Optional[torch.tensor] = None,
            orientation: Optional[torch.tensor] = None,
            max_velocity: [] = None,
    ) -> None:
        """initialize robot, set up sensors and controller

        Args:
            prim_path {str} -- prim path of the robot on the stage
            name {str} -- name of the quadruped
            position {np.ndarray} -- position of the robot
            orientation {np.ndarray} -- orientation of the robot

        """
        self._usd_path = usd_path
        self._name = name

        self._position = torch.tensor([0.0, 0.0, 0.0]) if position is None else position
        self._orientation = torch.tensor([1.0, 0.0, 0.0, 0.0]) if orientation is None else orientation



        if self._usd_path is None:
            root_path = os.getcwd()
            self._usd_path = root_path + '/Asset/Spot_rod.usd'

        carb.log_warn("asset path is: " + self._usd_path)
        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            position=self._position,
            orientation=self._orientation,
            articulation_controller=None
        )
        #self.num_dof = 19


        dof_paths = [
            'body/fl_hx',#revolute
            'body/fr_hx',#revolute
            'body/hl_hx',#revolute
            'body/hr_hx',#revolute
            'base_link/arm0_sh0',#revolute
            'arm0_link_sh0/arm0_sh1',#revolute 1.7
            'arm0_link_sh1/arm0_hr0',#revolute
            "arm0_link_hr0/arm0_el0",#revolute
            "arm0_link_el0/arm0_el1",#revolute 1.7
            "arm0_link_el1/arm0_wr0",#revolute
            "arm0_link_wr0/arm0_wr1",#revolute
            #"arm0_link_wr1/arm0_f1x",#revolute 0.75
            "front_left_hip/fl_hy",#revolute
            "front_left_upper_leg/fl_kn",#revolute
            "front_right_hip/fr_hy",#revolute
            "front_right_upper_leg/fr_kn",#revolute
            "rear_left_hip/hl_hy", #revolute
            "rear_left_upper_leg/hl_kn",#revolute
            "rear_right_hip/hr_hy", #revolute
            "rear_right_upper_leg/hr_kn",#revolute

        ]
        num_dof = len(dof_paths)

        drive_type = ["angular"] * num_dof
        default_dof_pos = [math.degrees(x) for x in [0.0,]*num_dof]
        stiffness = [3000 * np.pi / 180] * num_dof  #
        damping = [5 * np.pi / 180] * num_dof
        max_force = [200, ] * num_dof
        if max_velocity == None:
            max_velocity = [math.degrees(x)/3 for x in [2.175,] * num_dof]

        for i, dof in enumerate(dof_paths):
            set_drive(
                prim_path=f"{self.prim_path}/{dof}",
                drive_type=drive_type[i],
                target_type="position",
                target_value=default_dof_pos[i],
                stiffness=stiffness[i],
                damping=damping[i],
                max_force=max_force[i]
            )

            PhysxSchema.PhysxJointAPI(get_prim_at_path(f"{self.prim_path}/{dof}")).CreateMaxJointVelocityAttr().Set(
                max_velocity[i])