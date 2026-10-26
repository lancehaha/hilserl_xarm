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

import math
from unittest.mock import MagicMock, call, patch

import pytest

pytest.importorskip("xarm", reason="xarm-python-sdk is required (install lerobot[xarm])")

from lerobot.robots.xarm import XArm, XArmConfig

_MODULE = "lerobot.robots.xarm.xarm"
HOME_JOINTS = [0.0, 0.0, 0.0, 90.0, 0.0, 90.0, 0.0]
POSE_DEG = [300.0, -50.0, 200.0, 180.0, 0.0, 90.0]
STATE_KEYS = [
    *("ee.x", "ee.y", "ee.z", "ee.rx", "ee.ry", "ee.rz"),
    *("ee.vx", "ee.vy", "ee.vz", "ee.wx", "ee.wy", "ee.wz"),
]
FT_KEYS = ["ft.fx", "ft.fy", "ft.fz", "ft.tx", "ft.ty", "ft.tz"]


def _make_arm_mock() -> MagicMock:
    """Return an `XArmAPI` mock with the attributes and return codes used by the robot."""
    arm = MagicMock(name="XArmAPIMock")
    arm.connected = True
    arm.error_code = 0
    arm.axis = 7
    arm.mode = 0

    def _set_mode(mode):
        arm.mode = mode
        return 0

    arm.set_mode.side_effect = _set_mode
    for api in [
        "set_state",
        "clean_error",
        "clean_warn",
        "motion_enable",
        "set_gripper_enable",
        "set_gripper_mode",
        "set_gripper_speed",
        "set_position",
        "set_servo_angle",
        "set_gripper_position",
        "set_ft_sensor_enable",
        "set_ft_sensor_zero",
        "set_ft_sensor_mode",
        "set_ft_sensor_admittance_parameters",
    ]:
        getattr(arm, api).return_value = 0
    arm.pose_deg = list(POSE_DEG)

    def _get_position(is_radian=False):
        pose = list(arm.pose_deg)
        if is_radian:
            pose[3:] = [math.radians(angle) for angle in pose[3:]]
        return 0, pose

    arm.get_position.side_effect = _get_position
    arm.ft_ext_force = [0.0] * 6
    arm.get_gripper_position.return_value = (0, 800)
    arm.get_servo_angle.return_value = (0, [*HOME_JOINTS, 0.0])
    return arm


@pytest.fixture
def arm():
    return _make_arm_mock()


@pytest.fixture
def make_robot(arm, tmp_path):
    robots = []

    def _make(**config_kwargs) -> XArm:
        robot = XArm(XArmConfig(calibration_dir=tmp_path, **config_kwargs))
        robots.append(robot)
        return robot

    with patch(f"{_MODULE}.XArmAPI", return_value=arm):
        yield _make
    for robot in robots:
        if robot.is_connected:
            robot.disconnect()


@pytest.fixture
def robot(make_robot):
    robot = make_robot()
    robot.connect()
    return robot


def test_features():
    robot = XArm(XArmConfig())
    pose = dict.fromkeys(("ee.x", "ee.y", "ee.z", "ee.rx", "ee.ry", "ee.rz"), float)
    assert robot.action_features == {**pose, "gripper.pos": float}
    assert list(robot.action_features)[-1] == "gripper.pos"
    assert list(robot.observation_features) == [*STATE_KEYS, "gripper.pos"]

    robot = XArm(XArmConfig(use_ft_sensor=True))
    assert list(robot.observation_features) == [*STATE_KEYS, *FT_KEYS, "gripper.pos"]


def test_admittance_requires_ft_sensor():
    with pytest.raises(ValueError, match="use_ft_sensor"):
        XArmConfig(use_admittance=True)


def test_connect_streams_in_cartesian_mode(robot, arm):
    assert robot.is_connected
    assert arm.mode == 7
    arm.set_gripper_enable.assert_called_once_with(True)


def test_connect_failure(make_robot, arm):
    arm.connected = False
    robot = make_robot()
    with pytest.raises(ConnectionError):
        robot.connect()
    assert not robot.is_connected


def test_get_observation_in_meters_and_normalized_gripper(robot, arm):
    obs = robot.get_observation()
    assert obs["ee.x"] == pytest.approx(0.3)
    assert obs["ee.y"] == pytest.approx(-0.05)
    assert obs["ee.z"] == pytest.approx(0.2)
    assert [obs["ee.rx"], obs["ee.ry"], obs["ee.rz"]] == pytest.approx([math.pi, 0.0, math.pi / 2])
    assert obs["gripper.pos"] == pytest.approx(0.0)  # 800 = open

    arm.get_gripper_position.return_value = (0, 0)
    assert robot.get_observation()["gripper.pos"] == pytest.approx(1.0)  # 0 = closed


def test_gripper_fault_reuses_last_reading(robot, arm):
    arm.get_gripper_position.return_value = (0, 400)
    assert robot.get_observation()["gripper.pos"] == pytest.approx(0.5)
    # The SDK returns a bare error code when the gripper has a fault.
    arm.get_gripper_position.return_value = 23
    assert robot.get_observation()["gripper.pos"] == pytest.approx(0.5)


def test_twist_is_finite_difference(robot, arm):
    with patch(f"{_MODULE}.time.perf_counter", side_effect=[10.0, 10.0, 10.1, 10.1]):
        first = robot.get_observation()
        arm.pose_deg = [310.0, -50.0, 200.0, 180.0, 0.0, 95.0]
        second = robot.get_observation()
    assert [first[key] for key in STATE_KEYS[6:]] == pytest.approx([0.0] * 6)
    # +10 mm in x and +5 deg of yaw (about base z) in 0.1 s.
    expected = [0.1, 0.0, 0.0, 0.0, 0.0, math.radians(5.0) / 0.1]
    assert [second[key] for key in STATE_KEYS[6:]] == pytest.approx(expected, abs=1e-9)


def test_force_rotated_into_base_frame(make_robot, arm):
    robot = make_robot(use_ft_sensor=True)
    robot.connect()
    # Gripper pointing down (roll = pi, yaw = 0): tool z is base -z, tool y is base -y.
    arm.pose_deg = [300.0, 0.0, 200.0, 180.0, 0.0, 0.0]
    arm.ft_ext_force = [1.0, 2.0, -30.0, 0.1, 0.2, 0.3]
    obs = robot.get_observation()
    assert [obs[key] for key in FT_KEYS] == pytest.approx([1.0, -2.0, 30.0, 0.1, -0.2, -0.3])

    robot.config.ft_sensor_frame = "base"
    obs = robot.get_observation()
    assert [obs[key] for key in FT_KEYS] == pytest.approx(arm.ft_ext_force)


def test_send_action_pose_target(robot, arm):
    action = {"ee.x": 0.31, "ee.y": -0.05, "ee.z": 0.25, "ee.rx": 3.0, "ee.ry": 0.1, "ee.rz": 1.5}
    sent = robot.send_action({**action, "gripper.pos": 0.0})

    args, kwargs = arm.set_position.call_args
    assert args == pytest.approx((310.0, -50.0, 250.0, 3.0, 0.1, 1.5))
    assert kwargs == {"speed": 200.0, "mvacc": 2000.0, "is_radian": True, "wait": False}
    assert sent == {**action, "gripper.pos": 0.0}


ACTION = {"ee.x": 0.3, "ee.y": 0.0, "ee.z": 0.2, "ee.rx": math.pi, "ee.ry": 0.0, "ee.rz": 0.0}


def test_gripper_only_commanded_on_change(robot, arm):
    action = {**ACTION, "gripper.pos": 1.0}
    robot.send_action(action)
    robot.send_action(action)
    arm.set_gripper_position.assert_called_once_with(0.0, wait=False, wait_motion=False)

    robot.send_action({**action, "gripper.pos": 0.0})
    assert arm.set_gripper_position.call_args == call(800.0, wait=False, wait_motion=False)


def test_fault_blocks_motion(robot, arm):
    arm.error_code = 31
    assert robot.has_fault
    assert "31" in robot.fault_description

    robot.send_action({**ACTION, "gripper.pos": 1.0})
    arm.set_position.assert_not_called()
    arm.set_gripper_position.assert_not_called()


def test_reset_to_home(make_robot, arm):
    robot = make_robot(random_xy_range_m=0.02, random_rz_range_rad=0.1)
    robot.connect()
    arm.reset_mock()
    arm.mode = 7

    robot.reset_to_home(HOME_JOINTS)

    names = [name for name, _, _ in arm.method_calls]
    order = ["clean_error", "motion_enable", "set_servo_angle", "set_position", "set_gripper_position"]
    assert [name for name in names if name in order] == order
    assert arm.set_mode.call_args_list == [call(0), call(7)]

    _, kwargs = arm.set_servo_angle.call_args
    assert kwargs["angle"] == HOME_JOINTS
    assert kwargs["wait"] is True

    _, kwargs = arm.set_position.call_args
    assert kwargs["relative"] is True and kwargs["wait"] is True
    assert abs(kwargs["x"]) <= 20.0 and abs(kwargs["y"]) <= 20.0
    assert kwargs["z"] == 0 and kwargs["roll"] == 0 and kwargs["pitch"] == 0
    assert abs(kwargs["yaw"]) <= math.degrees(0.1) and kwargs["is_radian"] is False

    arm.set_gripper_position.assert_called_once_with(800, wait=True, wait_motion=False)


def test_reset_to_home_retries_gripper_failure(robot, arm):
    arm.set_gripper_position.side_effect = [-9, 0]
    with patch(f"{_MODULE}.time.sleep"):
        robot.reset_to_home(HOME_JOINTS)
    assert arm.set_servo_angle.call_count == 2
    assert arm.set_gripper_position.call_count == 2
    assert arm.mode == 7


def test_reset_to_home_raises_when_gripper_keeps_failing(robot, arm):
    arm.set_gripper_position.return_value = -9
    with patch(f"{_MODULE}.time.sleep"), pytest.raises(RuntimeError, match="set_gripper_position"):
        robot.reset_to_home(HOME_JOINTS)
    assert arm.set_gripper_position.call_count == 3


def test_reset_to_home_retries_after_collision(robot, arm):
    arm.set_servo_angle.side_effect = [1, 0]
    with patch(f"{_MODULE}.time.sleep"):
        robot.reset_to_home(HOME_JOINTS)
    assert arm.set_servo_angle.call_count == 2
    assert arm.mode == 7


def test_reset_to_home_raises_when_recovery_fails(robot, arm):
    arm.set_servo_angle.return_value = 1
    with patch(f"{_MODULE}.time.sleep"), pytest.raises(RuntimeError, match="set_servo_angle"):
        robot.reset_to_home(HOME_JOINTS)
    assert arm.set_servo_angle.call_count == 3


def test_reset_to_home_without_joints_stays_in_place(robot, arm):
    robot.reset_to_home(None)
    arm.set_servo_angle.assert_not_called()
    arm.set_position.assert_not_called()
    assert arm.mode == 7


def test_get_joint_positions(robot):
    assert robot.get_joint_positions() == HOME_JOINTS


def test_admittance_off_while_resetting(make_robot, arm):
    robot = make_robot(use_ft_sensor=True, use_admittance=True)
    robot.connect()
    arm.set_ft_sensor_enable.assert_called_once_with(1)
    assert arm.set_ft_sensor_admittance_parameters.call_args_list == [
        call(
            M=[0.06, 0.06, 0.06, 0.0006, 0.0006, 0.0006],
            K=[2000.0, 2000.0, 2000.0, 20.0, 20.0, 20.0],
            B=[0.0] * 6,
        ),
        call(coord=0, c_axis=[1, 1, 1, 1, 1, 1]),
    ]
    # Stiff until the first reset.
    arm.set_ft_sensor_mode.assert_not_called()

    arm.reset_mock()
    with patch(f"{_MODULE}.time.sleep"):
        robot.reset_to_home(HOME_JOINTS)
    names = [name for name, _, _ in arm.method_calls]
    order = ["set_ft_sensor_mode", "set_servo_angle", "set_ft_sensor_zero", "set_ft_sensor_mode"]
    assert [name for name in names if name in order] == order
    assert arm.set_ft_sensor_mode.call_args_list == [call(0), call(1)]
    # F/T sensor commands pause the controller (state 5): the arm must be set ready after the last of them,
    # or the streamed targets are ignored.
    last_ft_call = max(i for i, name in enumerate(names) if name.startswith("set_ft_sensor"))
    assert "set_state" in names[last_ft_call + 1 :]
    assert names[-2:] == ["set_mode", "set_state"]

    robot.disconnect()
    assert arm.set_ft_sensor_mode.call_args == call(0)
    assert arm.set_ft_sensor_enable.call_args == call(0)
