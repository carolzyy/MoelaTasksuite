from omni.isaac.core.robots.robot import Robot
import omni.isaac.dynamic_control._dynamic_control as omni_dc
from omni.isaac.core.utils.stage import get_current_stage,get_stage_units
from omni.isaac.core.utils.prims import get_prim_at_path,define_prim
from omni.isaac.core.utils.rotations import quat_to_euler_angles#, quat_to_rot_matrix, euler_to_rot_matrix
from typing import Optional,List
import numpy as np
import carb
from dataclasses import field, dataclass

DOF_DRIVE_MODE = {
    "force": int(omni_dc.DriveMode.DRIVE_FORCE),
    "acceleration": int(omni_dc.DriveMode.DRIVE_ACCELERATION),
}
"""Mapping from drive mode names to  drive mode in DC Toolbox type."""

DOF_CONTROL_MODE = {"position": 0, "velocity": 1, "effort": 2}
"""Mapping between control modes to integers."""

@dataclass
class LegSpotState():
    # ['fl_hx', 'fr_hx', 'hl_hx', 'hr_hx', 'arm0_sh0',
    # 'fl_hy', 'fr_hy', 'hl_hy', 'hr_hy', 'arm0_sh1',
    # 'fl_kn', 'fr_kn', 'hl_kn', 'hr_kn',
    # 'arm0_hr0', 'arm0_el0', 'arm0_el1', 'arm0_wr0', 'arm0_wr1', 'arm0_f1x']
    # 20 dof
    """The kinematic state of the articulated robot."""

    joint_pos: np.ndarray = field(default_factory=lambda: np.zeros(20))
    """Joint positions with shape: (20,)"""

    joint_vel: np.ndarray = field(default_factory=lambda: np.zeros(20))
    """Joint positions with shape: (20,)"""

    baseframe_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    baseframe_quat: np.ndarray = field(default_factory=lambda: np.zeros(4))
    # be careful may be get quaternion
    # x,y,z,rotation

    base_joint_pos: np.ndarray = field(default_factory=lambda: np.zeros(12))
    base_joint_vel: np.ndarray = field(default_factory=lambda: np.zeros(12))

    arm_joint_pos: np.ndarray = field(default_factory=lambda: np.zeros(7))
    arm_joint_vel: np.ndarray = field(default_factory=lambda: np.zeros(7))

    tool_joint_pos: np.ndarray = field(default_factory=lambda: np.zeros(1))
    tool_joint_vel: np.ndarray = field(default_factory=lambda: np.zeros(1))


class LegSpot(Robot):
    def __init__(
            self,
            prim_path:str= "/World/env/spot",
            name: str = 'LegSpot',
            usd_path: Optional[str] = None,
            position: Optional[np.ndarray] = None,
            orientation: Optional[np.ndarray] = None,
    ) -> None:
        """initialize robot, set up sensors and controller

        Args:
            prim_path {str} -- prim path of the robot on the stage
            name {str} -- name of the quadruped
            position {np.ndarray} -- position of the robot
            orientation {np.ndarray} -- orientation of the robot

        """
        self._stage = get_current_stage()
        self._prim_path = prim_path
        prim = get_prim_at_path(self._prim_path)
        self.meters_per_unit = get_stage_units()

        if not prim.IsValid():
            prim = define_prim(self._prim_path, "Xform")
            if usd_path:
                prim.GetReferences().AddReference(usd_path)
            else:
                asset_path = '/home/carol/Project/benchmark/Issac_test/Asset/Legspot.usd'

                carb.log_warn("asset path is: " + asset_path)
                prim.GetReferences().AddReference(asset_path)



        super().__init__(
            prim_path=prim_path, name=name, position=position, orientation=orientation, articulation_controller=None
        )

        self.dof_limit = None
        self.dof_vel_limit = 100.0
        self.dof_max = None


        self.num_dof_base = 12
        self.num_dof_arm = 7
        self.num_dof_tool = 1

        self._dof_control_mode = 'position'
        self._dof_control_modes: List[int] = list()
        self._state = LegSpotState()
        self._default_state = LegSpotState()

        self._default_state.baseframe_pos = np.array([0.0, 0.0, 0.0])
        self._default_state.baseframe_orien = np.array([0.0, 0.0, 0.0, 1.0])

        self._default_state.joint_pos = np.zeros(20)
        self._default_state.joint_vel = np.zeros(20)
        #self._default_state.joint_vel = np.array([0,0,0,0,0,-2,-2,-2,-2,0,24,23,25,24,0,0,0,0,0,0])



    def check_dc_interface(self) -> None:
        """[summary]

        Checks the DC interface handle of the robot

        Raises:
            RuntimeError: When the DC Toolbox interface has not been configured.
        """
        if self._handle == omni_dc.INVALID_HANDLE or self._handle is None:
            raise RuntimeError(f"Failed to obtain articulation handle at: '{self._prim_path}'")
        return True

    def set_dof_drive_mode(self, drive) -> None:
        """[summary]

        Set drive mode of the quadruped to force or acceleration

        Args:
            drive {List[str]} -- drive mode of the robot, can be either "force" or "acceleration"

        """
        self.check_dc_interface()
        dof_props = self._dc_interface.get_articulation_dof_properties(self._handle)
        if not isinstance(drive, list):
            drive = [drive] * self.num_dof
        if not len(drive) == self.num_dof:
            msg = f"Insufficient number of DOF drive modes specified. Expected: {self.num_dof}. Received: {len(drive)}."
            carb.log_error(msg)
        for index, drive_mode in enumerate(drive):
            # set drive mode
            try:
                dof_props["driveMode"][index] = DOF_DRIVE_MODE[drive_mode]
            except AttributeError:
                msg = f"Invalid articulation drive mode '{drive_mode}'. Supported drive types: {DOF_DRIVE_MODE.keys()}"
                raise ValueError(msg)
        # Set the properties into simulator
        self._dc_interface.set_articulation_dof_properties(self._handle, dof_props)

    def set_dof_control(self, control, kp, kd, drive) -> None:
        """[summary]

        Set dof control to position, velocity or effort

        Args:
            control {int or List[int]}: DOF control mode, can be  {"position": 0, "velocity": 1, "effort": 2}
            kp {float or List[float]}: proportional constant
            kd {float or List[float]}: derivative constant
            drive {int or List[int]}: DOF drive mode, can be "force": int(omni_dc.DriveMode.DRIVE_FORCE) or "acceleration": int(omni_dc.DriveMode.DRIVE_ACCELERATION)
        """
        self.check_dc_interface()
        # Extend to list if values provided
        if not isinstance(control, list):
            control = [control] * self.num_dof
        if not isinstance(kp, list):
            kp = [kp] * self.num_dof
        if not isinstance(kd, list):
            kd = [kd] * self.num_dof

        # Check that lists are of the correct size
        if not len(control) == self.num_dof:
            msg = f"Insufficient number of DOF control modes specified. Expected: {self.num_dof}. Received: {len(control)}."
            raise ValueError(msg)
        if not len(kp) == self.num_dof:
            msg = f"Insufficient number of DOF stiffness specified. Expected: {self.num_dof}. Received: {len(kp)}."
            raise ValueError(msg)
        if not len(kd) == self.num_dof:
            msg = f"Insufficient number of DOF damping specified. Expected: {self.num_dof}. Received: {len(kd)}."
            raise ValueError(msg)

        dof_props = self._dc_interface.get_articulation_dof_properties(self._handle)
        for index, (control_mode, stiffness, damping) in enumerate(zip(control, kp, kd)):
            # set control mode
            try:
                control_value = DOF_CONTROL_MODE[control_mode]
                self._dof_control_modes.append(control_value)
            except AttributeError:
                msg = f"Invalid articulation control mode '{control_mode}'. Supported control types: {DOF_CONTROL_MODE.keys()}"
                raise ValueError(msg)

            # set drive mode
            dof_props["driveMode"][index] = DOF_DRIVE_MODE[drive]
            # set the gains
            if stiffness is not None:
                dof_props["stiffness"][index] = stiffness
            if damping is not None:
                dof_props["damping"][index] = damping

        # Set the properties into simulator
        self._dc_interface.set_articulation_dof_properties(self._handle, dof_props)
        return

    def update(self) -> None:
        """[summary]



        Raises:
            NotImplementedError if not implemented
        """
        joint_state = super().get_joints_state()
        if joint_state == None:
            return True

        self._state.joint_pos = joint_state.positions
        self._state.joint_vel = joint_state.velocities

        if self._root_handle == omni_dc.INVALID_HANDLE:
            raise RuntimeError(f"Failed to obtain articulation handle at: '{self._prim_path}'")



        #self._state.base_joint_pos = joint_state.positions[0:3, ]
        #self._state.base_joint_vel = joint_state.velocities[0:3, ]

        #self._state.arm_joint_pos = joint_state.positions[3:10, ]
        #self._state.arm_joint_vel = joint_state.velocities[3:10, ]

        #self._state.tool_joint_pos = joint_state.positions[10:, ]
        #self._state.tool_joint_vel = joint_state.velocities[10:, ]

    def advance(self) -> None:
        """[summary]

        Compute torque applied on each joint

        Raises:
            NotImplementedError if not implemented
        """
        raise NotImplementedError

    def initialize(self, physics_sim_view=None) -> None:
        """[summary]

        initialize dc interface, set up drive mode
        """
        super().initialize(physics_sim_view=physics_sim_view)
        self.set_dof_drive_mode(drive="force")
        # be careful about the dof control mode selection
        #self.set_dof_control(control="position", kp=0.0, kd=0.0, drive="force")
        self.set_state(self._default_state)
        self.dof_limit = self.get_articulation_controller().get_joint_limits()

    def post_reset(self) -> None:
        """[summary]

        post reset articulation
        """
        super().post_reset()

    def get_state(self):
        joint_state = self.get_joints_state()
        self._state.joint_pos = joint_state.positions
        self._state.joint_vel = joint_state.velocities
        '''
        base_frame_pose = self._dc_interface.get_rigid_body_pose(self._root_handle)
        self._state.base_frame_pos[0:3] = base_frame_pose.p
        self._state.base_frame_pos[-1] = quat_to_euler_angles(base_frame_pose.r)

        self._state.base_frame_vel[0:3]= np.asarray(self._dc_interface.get_rigid_body_linear_velocity(self._root_handle)) * self.meters_per_unit
        self._state.base_frame_vel[3:] = np.asarray(self._dc_interface.get_rigid_body_angular_velocity(self._root_handle)) * self.meters_per_unit

        self._state.arm_joint_pos = joint_state.positions[3:,]
        self._state.arm_joint_vel = joint_state.velocities[3:, ]
        self._state.base_joint_pos = joint_state.positions[0:3]
        self._state.base_joint_vel = joint_state.velocities[0:3]
        '''

    def set_state(self, state: LegSpotState):
        self.check_dc_interface()

        dof_state = self._dc_interface.get_articulation_dof_states(self._handle, omni_dc.STATE_ALL)

        dof_state["pos"] = np.asarray(np.array(state.joint_pos.flat), dtype=np.float32)
        dof_state["vel"] = np.asarray(np.array(state.joint_vel.flat), dtype=np.float32)
        dof_state["effort"] = 0.0
        # set joint state
        status = self._dc_interface.set_articulation_dof_states(self._handle, dof_state, omni_dc.STATE_ALL)
        if not status:
            raise RuntimeError("Unable to set the DOF state properly.")

    def apply_dofaction(self, act, repeat_time = 5) -> None:
        act[-1] = 0 # keep gripper close
        actions = act
        self._dc_interface.wake_up_articulation(self._handle)
        for _ in range(repeat_time):
            if self._dof_control_mode == 'position':
                self._dc_interface.set_articulation_dof_position_targets(self._handle, np.asarray(actions, dtype=np.float32))
            if self._dof_control_mode == 'velocity':
                self._dc_interface.set_articulation_dof_velocity_targets(self._handle, np.asarray(actions, dtype=np.float32))
            if self._dof_control_mode == 'effort':
                self._dc_interface.set_articulation_dof_efforts(self._handle, np.asarray(actions, dtype=np.float32))


