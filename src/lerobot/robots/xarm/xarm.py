#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The connection sequence and gripper normalization follow UFACTORY's `lerobot_robot_ufactory`
# (https://github.com/xArm-Developer/lerobot_robot_ufactory, Copyright 2025 UFACTORY Inc., Apache-2.0),
# reimplemented on the public xArm SDK API.

import logging
import time
from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from lerobot.cameras import DepthCamera, make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import _xarm_available, require_package
from lerobot.utils.rotation import Rotation

from ..robot import Robot
from .config_xarm import XArmConfig

if TYPE_CHECKING or _xarm_available:
    from xarm.core.config.x_code import ControllerError
    from xarm.wrapper import XArmAPI

logger = logging.getLogger(__name__)

EE_POSITION_KEYS = ("ee.x", "ee.y", "ee.z")
# xArm roll / pitch / yaw in radians: fixed-axis X, Y, Z rotations, i.e. scipy's extrinsic "xyz" Euler angles.
EE_ORIENTATION_KEYS = ("ee.rx", "ee.ry", "ee.rz")
EE_POSE_KEYS = EE_POSITION_KEYS + EE_ORIENTATION_KEYS
# TCP twist in the base frame (m/s, rad/s), as `tcp_vel` in rail-berkeley/hil-serl.
EE_VELOCITY_KEYS = ("ee.vx", "ee.vy", "ee.vz", "ee.wx", "ee.wy", "ee.wz")
# External force (N) and torque (Nm) on the TCP in the base frame, as `tcp_force` / `tcp_torque`.
FORCE_KEYS = ("ft.fx", "ft.fy", "ft.fz")
TORQUE_KEYS = ("ft.tx", "ft.ty", "ft.tz")
GRIPPER_KEY = "gripper.pos"

# xArm controller modes and states, see the xArm SDK documentation.
POSITION_MODE = 0
ONLINE_CARTESIAN_MODE = 7
READY_STATE = 0
FT_NON_FORCE_MODE = 0
FT_ADMITTANCE_MODE = 1

MM_PER_M = 1000.0
_MODE_SWITCH_TIMEOUT_S = 2.0
_RECOVERY_ATTEMPTS = 3
_RECOVERY_WAIT_S = 1.0
# Settling times from the UFACTORY F/T sensor examples.
_FT_ENABLE_WAIT_S = 0.5
_FT_ZERO_WAIT_S = 0.2


class XArm(Robot):
    """
    UFACTORY xArm with an xArm Gripper, controlled in Cartesian end-effector space.

    Observations are the measured TCP pose in the robot base frame (meters, xArm roll / pitch / yaw in
    radians), the TCP twist (finite difference of consecutive readings), optionally the external force and
    torque from the Six-axis F/T sensor (base frame), and the normalized gripper opening (0 = open, 1 =
    closed). Actions are absolute TCP pose targets plus a gripper target; the controller's online trajectory
    planning mode (mode 7) does the inverse kinematics.

    With `use_admittance`, the controller's F/T admittance control makes the arm compliant during episodes,
    like the Cartesian impedance controller of rail-berkeley/hil-serl. It is switched off while resetting.

    Controller-side safety (safety boundary, collision sensitivity, self-collision detection, TCP offset and
    payload) is left to UFACTORY Studio. Controller errors are reported through `has_fault`; `reset_to_home`
    clears them.
    """

    config_class = XArmConfig
    name = "xarm"

    def __init__(self, config: XArmConfig):
        require_package("xarm-python-sdk", extra="xarm", import_name="xarm")
        super().__init__(config)
        self.config = config
        self.arm: XArmAPI | None = None
        self.cameras = make_cameras_from_configs(config.cameras)
        self._rng = np.random.default_rng()
        # Previous (time, position, orientation) reading, for the finite-difference TCP twist.
        self._last_pose_reading: tuple[float, np.ndarray, Rotation] | None = None
        self._last_gripper_target: float | None = None
        self._last_gripper_obs = 0.0

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        features: dict[str, tuple] = {}
        for cam_key, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                features[cam_key] = (cam.height, cam.width, 3)
            if isinstance(cam, DepthCamera) and cam.use_depth:
                features[f"{cam_key}_depth"] = (cam.height, cam.width, 1)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        state_keys = [*EE_POSE_KEYS, *EE_VELOCITY_KEYS]
        if self.config.use_ft_sensor:
            state_keys += [*FORCE_KEYS, *TORQUE_KEYS]
        return {**dict.fromkeys(state_keys, float), GRIPPER_KEY: float, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        # The gripper stays last: the discrete gripper critic and the gripper penalty expect it there.
        return {**dict.fromkeys(EE_POSE_KEYS, float), GRIPPER_KEY: float}

    @property
    def is_connected(self) -> bool:
        return (
            self.arm is not None
            and self.arm.connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def _api(self) -> "XArmAPI":
        """The xArm SDK handle, only available while connected."""
        if self.arm is None:
            raise DeviceNotConnectedError(f"{self} is not connected. Run `.connect()` first.")
        return self.arm

    @property
    def has_fault(self) -> bool:
        """Whether the controller reports an error. Motion commands are ignored until `reset_to_home`."""
        return self.arm is not None and self.arm.error_code != 0

    @property
    def fault_description(self) -> str:
        if not self.has_fault:
            return ""
        code = self._api.error_code
        error = ControllerError(code)
        return f"xArm error {code}: {error.title['en']}. {error.description['en']}"

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        arm = XArmAPI(self.config.ip, is_radian=False)
        if not arm.connected:
            raise ConnectionError(f"Could not connect to the xArm at {self.config.ip}.")
        self.arm = arm

        self._enable_motion()
        self._check(self._api.set_gripper_enable(True), "set_gripper_enable")
        self._check(self._api.set_gripper_mode(0), "set_gripper_mode")
        self._check(self._api.set_gripper_speed(self.config.gripper_speed), "set_gripper_speed")
        if self.config.use_ft_sensor:
            self._start_ft_sensor()
        if self.config.use_admittance:
            self._configure_admittance()
        # Stiff until the first `reset_to_home`, which switches compliance on at the start pose.
        self._set_mode(ONLINE_CARTESIAN_MODE)

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def reset_to_home(self, joints_deg: Sequence[float] | None = None) -> None:
        """
        Clear any controller error and move (blocking) to the home joint configuration, then open the gripper.

        A joint-space home keeps the redundant joints of the arm from drifting across episodes. When
        `random_xy_range_m` / `random_rz_range_rad` are set, a uniform random xy / yaw offset is then added to
        the home TCP pose. As `go_to_reset` in rail-berkeley/hil-serl, the arm is stiff while resetting; the
        F/T sensor is zeroed at the start pose and admittance control is switched back on afterwards.

        Args:
            joints_deg: Home joint angles in degrees. If `None`, the arm stays where it is.
        """
        # Right after a collision the controller can raise the error again on the first move, so retry.
        for attempt in range(1, _RECOVERY_ATTEMPTS + 1):
            code, api = self._try_reset_to_home(joints_deg)
            if code == 0 and not self.has_fault:
                break
            if attempt == _RECOVERY_ATTEMPTS:
                self._check(code or -1, api)
            logger.warning(
                f"{self} homing failed (`{api}` code {code}, attempt {attempt}/{_RECOVERY_ATTEMPTS}), "
                f"retrying. {self.fault_description}"
            )
            time.sleep(_RECOVERY_WAIT_S)

        self._last_gripper_target = 0.0
        self._last_pose_reading = None
        # F/T sensor commands leave the controller paused (state 5), so they come before `_set_mode`, which sets
        # it ready again. Otherwise motion commands are ignored (only the gripper, on its own bus, moves).
        if self.config.use_ft_sensor:
            self._zero_ft_sensor()
        if self.config.use_admittance:
            self._set_ft_mode(FT_ADMITTANCE_MODE)
        self._set_mode(ONLINE_CARTESIAN_MODE)

    def _try_reset_to_home(self, joints_deg: Sequence[float] | None) -> tuple[int, str]:
        """One homing attempt. Returns the first non-zero SDK code (or 0) and the API that returned it."""
        self._enable_motion()
        if self.config.use_admittance:
            self._set_ft_mode(FT_NON_FORCE_MODE)
        self._set_mode(POSITION_MODE)
        if joints_deg is not None:
            code = self._api.set_servo_angle(
                angle=list(joints_deg),
                speed=self.config.home_joint_speed_deg_s,
                is_radian=False,
                wait=True,
            )
            if code != 0:
                return code, "set_servo_angle"

        if self.config.random_xy_range_m > 0 or self.config.random_rz_range_rad > 0:
            dx, dy = self._rng.uniform(-self.config.random_xy_range_m, self.config.random_xy_range_m, size=2)
            dyaw = self._rng.uniform(-self.config.random_rz_range_rad, self.config.random_rz_range_rad)
            # A relative yaw is a rotation about the base Z axis, as hil-serl's `euler_random[-1] += ...`.
            code = self._api.set_position(
                x=dx * MM_PER_M,
                y=dy * MM_PER_M,
                z=0,
                roll=0,
                pitch=0,
                yaw=np.degrees(dyaw),
                relative=True,
                speed=self.config.max_linear_speed_mm_s,
                is_radian=False,
                wait=True,
            )
            if code != 0:
                return code, "set_position"

        # The moves above already waited for the arm. By default the SDK also waits for the arm before moving
        # the gripper, and that wait returns -9 (`EMERGENCY_STOP`) on a controller left stopped by a fault.
        code = self._api.set_gripper_position(self.config.gripper_open_pos, wait=True, wait_motion=False)
        return code, "set_gripper_position"

    @check_if_not_connected
    def get_joint_positions(self) -> list[float]:
        """Current joint angles in degrees, e.g. to save as the home configuration."""
        code, angles = self._api.get_servo_angle(is_radian=False)
        self._check(code, "get_servo_angle")
        return [float(a) for a in angles[: self._api.axis]]

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()
        code, pose = self._api.get_position(is_radian=True)
        self._check(code, "get_position")
        position = np.asarray(pose[:3], dtype=float) / MM_PER_M
        rpy = np.asarray(pose[3:6], dtype=float)
        orientation = Rotation.from_euler("xyz", rpy)

        obs_dict: RobotObservation = {
            **dict(zip(EE_POSITION_KEYS, position.tolist(), strict=True)),
            **dict(zip(EE_ORIENTATION_KEYS, rpy.tolist(), strict=True)),
            **dict(
                zip(EE_VELOCITY_KEYS, self._tcp_twist(start, position, orientation).tolist(), strict=True)
            ),
        }
        if self.config.use_ft_sensor:
            wrench = np.asarray(self._api.ft_ext_force, dtype=float)
            force, torque = wrench[:3], wrench[3:6]
            if self.config.ft_sensor_frame == "base":
                force, torque = orientation.apply(force), orientation.apply(torque)
            obs_dict.update(zip(FORCE_KEYS, force.tolist(), strict=True))
            obs_dict.update(zip(TORQUE_KEYS, torque.tolist(), strict=True))
        obs_dict[GRIPPER_KEY] = self._read_gripper()
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        for cam_key, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                obs_dict[cam_key] = cam.read_latest()
            if isinstance(cam, DepthCamera) and cam.use_depth:
                obs_dict[f"{cam_key}_depth"] = cam.read_latest_depth()

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """
        Command an absolute TCP pose target (meters and xArm roll / pitch / yaw in radians, base frame) and a
        gripper target in [0, 1].

        Nothing is sent while the controller reports an error. The gripper is only commanded when its target
        changes, so the arm streaming is not slowed down by the gripper's Modbus link.

        Returns:
            RobotAction: the action actually sent (gripper clipped to [0, 1]).
        """
        target = [float(action[key]) for key in EE_POSE_KEYS]
        gripper = float(np.clip(action[GRIPPER_KEY], 0.0, 1.0))
        sent_action: RobotAction = {
            **dict(zip(EE_POSE_KEYS, target, strict=True)),
            GRIPPER_KEY: gripper,
        }

        if self.has_fault:
            logger.debug(f"{self} ignores the action: {self.fault_description}")
            return sent_action

        code = self._api.set_position(
            *(t * MM_PER_M for t in target[:3]),
            *target[3:],
            speed=self.config.max_linear_speed_mm_s,
            mvacc=self.config.max_linear_acc_mm_s2,
            is_radian=True,
            wait=False,
        )
        if code != 0:
            logger.warning(f"{self} set_position returned code {code}")

        if self._last_gripper_target is None or abs(gripper - self._last_gripper_target) > 1e-3:
            # `wait_motion=False`: by default the SDK waits for the arm to stop before moving the gripper.
            code = self._api.set_gripper_position(
                self._normalized_to_gripper(gripper), wait=False, wait_motion=False
            )
            if code == 0:
                self._last_gripper_target = gripper
            else:
                logger.warning(f"{self} set_gripper_position returned code {code}")

        return sent_action

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.config.use_admittance:
            self._api.set_ft_sensor_mode(FT_NON_FORCE_MODE)
        if self.config.use_ft_sensor:
            self._api.set_ft_sensor_enable(0)
        self._api.set_mode(POSITION_MODE)
        self._api.set_state(READY_STATE)
        self._api.disconnect()
        self.arm = None
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")

    def _check(self, code: int, api: str) -> None:
        if code != 0:
            raise RuntimeError(f"{self}: xArm `{api}` failed with code {code}. {self.fault_description}")

    def _enable_motion(self) -> None:
        self._api.clean_error()
        self._api.clean_warn()
        # The error is cleared asynchronously by the controller.
        deadline = time.perf_counter() + _MODE_SWITCH_TIMEOUT_S
        while self.has_fault and time.perf_counter() < deadline:
            time.sleep(0.05)
        self._check(self._api.motion_enable(enable=True), "motion_enable")

    def _set_mode(self, mode: int) -> None:
        self._check(self._api.set_mode(mode), "set_mode")
        self._check(self._api.set_state(READY_STATE), "set_state")
        # The new mode is only active once the controller reports it.
        deadline = time.perf_counter() + _MODE_SWITCH_TIMEOUT_S
        while self._api.mode != mode:
            if time.perf_counter() > deadline:
                raise RuntimeError(f"{self}: xArm did not switch to mode {mode}. {self.fault_description}")
            time.sleep(0.01)

    def _tcp_twist(self, now: float, position: np.ndarray, orientation: Rotation) -> np.ndarray:
        """Base-frame TCP twist from the previous pose reading, zero on the first reading after a reset."""
        last = self._last_pose_reading
        self._last_pose_reading = (now, position, orientation)
        if last is None or now - last[0] <= 0.0:
            return np.zeros(6)
        dt = now - last[0]
        linear = (position - last[1]) / dt
        angular = (orientation * last[2].inv()).as_rotvec() / dt
        return np.concatenate([linear, angular])

    def _start_ft_sensor(self) -> None:
        self._check(self._api.set_ft_sensor_enable(1), "set_ft_sensor_enable")
        time.sleep(_FT_ENABLE_WAIT_S)
        self._zero_ft_sensor()

    def _zero_ft_sensor(self) -> None:
        self._check(self._api.set_ft_sensor_zero(), "set_ft_sensor_zero")
        time.sleep(_FT_ZERO_WAIT_S)

    def _configure_admittance(self) -> None:
        # Damping is reserved: the controller uses 2 * sqrt(M * K).
        code = self._api.set_ft_sensor_admittance_parameters(
            M=list(self.config.admittance_mass), K=list(self.config.admittance_stiffness), B=[0.0] * 6
        )
        self._check(code, "set_ft_sensor_admittance_parameters")
        code = self._api.set_ft_sensor_admittance_parameters(
            coord=self.config.admittance_frame, c_axis=list(self.config.admittance_axes)
        )
        self._check(code, "set_ft_sensor_admittance_parameters")

    def _set_ft_mode(self, mode: int) -> None:
        self._check(self._api.set_ft_sensor_mode(mode), "set_ft_sensor_mode")
        # The force mode takes effect once the controller is set ready again.
        self._check(self._api.set_state(READY_STATE), "set_state")

    def _read_gripper(self) -> float:
        ret = self._api.get_gripper_position()
        # The SDK returns a bare error code instead of a `(code, position)` tuple when the gripper has a fault.
        code, position = ret if isinstance(ret, tuple) else (ret, None)
        if code != 0 or position is None:
            logger.warning(f"{self} could not read the gripper (code {code}), reusing the last reading.")
            return self._last_gripper_obs
        self._last_gripper_obs = self._gripper_to_normalized(position)
        return self._last_gripper_obs

    def _gripper_to_normalized(self, position: float) -> float:
        open_pos, close_pos = self.config.gripper_open_pos, self.config.gripper_close_pos
        return float(np.clip((position - open_pos) / (close_pos - open_pos), 0.0, 1.0))

    def _normalized_to_gripper(self, value: float) -> float:
        open_pos, close_pos = self.config.gripper_open_pos, self.config.gripper_close_pos
        return open_pos + value * (close_pos - open_pos)
