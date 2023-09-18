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



class RidgeFranka(Robot):
    def __init__(
            self,
            prim_path:str,
            name: str = 'RidgeFranka',
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

        self._position = torch.tensor([1.0, 0.0, 0.0]) if position is None else position
        self._orientation = torch.tensor([1.0, 0.0, 0.0, 0.0]) if orientation is None else orientation



        if self._usd_path is None:
            root_path = os.getcwd()
            self._usd_path = root_path + '/Asset/ridgeback_franka.usd'

        carb.log_warn("asset path is: " + self._usd_path)
        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            position=self._position,
            orientation=self._orientation,
            articulation_controller=None
        )


        dof_paths = [
            'world/dummy_base_prismatic_x_joint',
            'dummy_base_x/dummy_base_prismatic_y_joint',
            'dummy_base_y/dummy_base_revolute_z_joint',
            "panda_link0/panda_joint1",
            "panda_link1/panda_joint2",
            "panda_link2/panda_joint3",
            "panda_link3/panda_joint4",
            "panda_link4/panda_joint5",
            "panda_link5/panda_joint6",
            "panda_link6/panda_joint7"
            #"panda_hand/panda_finger_joint1",
            #"panda_hand/panda_finger_joint2"
        ]

        drive_type = ["linear"] * 2 + ["angular"] * 8
        default_dof_pos = [0.0, 0.0]+[math.degrees(x) for x in [0.0,0.0, -1.0, 0.0, -2.2, 0.0, 2.4, 0.8]]
        stiffness = [1000] * 2+[400 * np.pi / 180] * 8  #
        damping = [100] * 2+[80 * np.pi / 180] * 8
        max_force = [200,200,87,87, 87, 87, 87, 12, 12, 12,]
        if max_velocity == None:
            max_velocity = [0.2, 0.2]+[math.degrees(x)/3 for x in [2.175,2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]]

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