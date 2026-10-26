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

"""`gym_manipulator` with a robot that has no motor bus and takes Cartesian targets (e.g. the xArm)."""

import math
import time
from unittest.mock import patch

import numpy as np
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.envs.configs import (
    GripperConfig,
    HILSerlProcessorConfig,
    HILSerlRobotEnvConfig,
    InverseKinematicsConfig,
    ObservationConfig,
    ResetConfig,
)
from lerobot.processor import TransitionKey
from lerobot.rl.gym_manipulator import (
    RobotEnv,
    make_processors,
    reset_and_build_transition,
    step_env_and_process_transition,
)
from lerobot.teleoperators.utils import TeleopEvents
from tests.mocks.mock_robot import MockRobot, MockRobotConfig

POSE_KEYS = ["ee.x", "ee.y", "ee.z", "ee.rx", "ee.ry", "ee.rz"]
VELOCITY_KEYS = ["ee.vx", "ee.vy", "ee.vz", "ee.wx", "ee.wy", "ee.wz"]
STATE_KEYS = [*POSE_KEYS, *VELOCITY_KEYS, "gripper.pos"]
ACTION_KEYS = [*POSE_KEYS, "gripper.pos"]
# Gripper pointing down.
START_STATE = {
    **dict(zip(POSE_KEYS, [0.3, 0.0, 0.2, math.pi, 0.0, 0.0], strict=True)),
    **dict.fromkeys(VELOCITY_KEYS, 0.0),
    "gripper.pos": 0.0,
}
DOWN = {"ee.rx": math.pi, "ee.ry": 0.0, "ee.rz": 0.0}
HOME_JOINTS = [0.0, 0.0, 0.0, 90.0, 0.0, 90.0, 0.0]


class FakeCartesianRobot:
    """Bus-less robot with the interface of `XArm`: Cartesian state/targets, `reset_to_home`, `has_fault`."""

    name = "fake_cartesian"

    def __init__(self):
        self.is_connected = True
        self.cameras = {"wrist": None}
        self.observation_features = {**dict.fromkeys(STATE_KEYS, float), "wrist": (8, 8, 3)}
        self.action_features = dict.fromkeys(ACTION_KEYS, float)
        self.state = dict(START_STATE)
        self.has_fault = False
        self.fault_description = ""
        self.sent: list[dict] = []
        self.home_calls: list = []

    def get_observation(self):
        return {**self.state, "wrist": np.zeros((8, 8, 3), dtype=np.uint8)}

    def send_action(self, action):
        action = {key: float(value) for key, value in action.items()}
        self.sent.append(action)
        return action

    def reset_to_home(self, joints_deg):
        self.home_calls.append(joints_deg)

    def disconnect(self):
        self.is_connected = False


class FakeTeleop:
    """SpaceMouse-like teleop: delta action + HIL events."""

    def __init__(self):
        self.action = {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0, "gripper": 1}
        self.intervention = False

    def get_action(self):
        return dict(self.action)

    def get_teleop_events(self):
        return {
            TeleopEvents.IS_INTERVENTION: self.intervention,
            TeleopEvents.TERMINATE_EPISODE: False,
            TeleopEvents.SUCCESS: False,
            TeleopEvents.RERECORD_EPISODE: False,
        }


@pytest.fixture(autouse=True)
def _no_sounds():
    with patch("lerobot.rl.gym_manipulator.log_say"):
        yield


@pytest.fixture
def robot():
    return FakeCartesianRobot()


@pytest.fixture
def env(robot):
    return RobotEnv(robot=robot, use_gripper=True, reset_pose=HOME_JOINTS, reset_time_s=0.0)


def test_env_state_from_robot_features(env, robot):
    obs, info = env.reset()
    assert robot.home_calls == [HOME_JOINTS]
    np.testing.assert_allclose(obs["agent_pos"], list(START_STATE.values()))
    assert list(obs["pixels"]) == ["wrist"]
    assert list(env.get_raw_joint_positions()) == STATE_KEYS
    assert env.observation_space["observation.state"].shape == (13,)
    assert env.action_space.shape == (4,)
    assert RobotEnv(
        robot=robot, use_gripper=True, reset_time_s=0.0, use_rotation=True
    ).action_space.shape == (7,)
    assert info == {TeleopEvents.IS_INTERVENTION: False}


def test_env_step_sends_action_by_feature_order(env, robot):
    env.reset()
    _, _, terminated, truncated, info = env.step(torch.tensor([0.31, 0.0, 0.2, 3.0, 0.1, 0.2, 1.0]))
    expected = dict(zip(ACTION_KEYS, [0.31, 0.0, 0.2, 3.0, 0.1, 0.2, 1.0], strict=True))
    assert robot.sent[-1] == pytest.approx(expected)
    assert not terminated and not truncated
    assert "robot_fault" not in info


def test_robot_fault_fails_episode(env, robot):
    env.reset()
    robot.has_fault = True
    _, _, terminated, truncated, info = env.step(torch.tensor([0.3, 0.0, 0.2, math.pi, 0.0, 0.0, 0.0]))
    assert terminated and not truncated
    assert info["robot_fault"] is True


def test_step_observes_after_control_period(robot):
    """The observation is taken a control period after the action is sent, not right after `send_action`."""
    period = 0.05
    env = RobotEnv(robot=robot, use_gripper=True, reset_time_s=0.0, control_period_s=period)
    events = []
    send_action, get_observation = robot.send_action, robot.get_observation
    robot.send_action = lambda action: events.append(("send", time.perf_counter())) or send_action(action)
    robot.get_observation = lambda: events.append(("obs", time.perf_counter())) or get_observation()

    env.reset()
    for _ in range(3):
        events.clear()
        env.step(torch.tensor([0.3, 0.0, 0.2, math.pi, 0.0, 0.0, 0.0]))
        (send, send_time), (obs, obs_time) = events
        assert (send, obs) == ("send", "obs")
        assert obs_time - send_time >= period - 1e-3


def test_motor_bus_robot_unchanged():
    robot = MockRobot(MockRobotConfig(n_motors=3, random_values=False, static_values=[1.0, 2.0, 3.0]))
    env = RobotEnv(robot=robot, use_gripper=False, reset_time_s=0.0)
    obs, _ = env.reset()
    np.testing.assert_allclose(obs["agent_pos"], [1.0, 2.0, 3.0])
    assert list(env.get_raw_joint_positions()) == ["motor_1.pos", "motor_2.pos", "motor_3.pos"]
    with patch.object(robot, "send_action", wraps=robot.send_action) as send_action:
        env.step(torch.tensor([4.0, 5.0, 6.0]))
    sent = send_action.call_args.args[0]
    assert list(sent) == ["motor_1.pos", "motor_2.pos", "motor_3.pos"]
    assert [float(v) for v in sent.values()] == [4.0, 5.0, 6.0]


def _cartesian_env_cfg(delta_reference: str = "measured", six_dof: bool = False) -> HILSerlRobotEnvConfig:
    extra = {}
    if six_dof:
        extra = {"use_rotation": True, "end_effector_rot_step": 0.1, "action_frame": "ee"}
    return HILSerlRobotEnvConfig(
        fps=10,
        processor=HILSerlProcessorConfig(
            observation=ObservationConfig(relative_ee_frame=six_dof),
            gripper=GripperConfig(use_gripper=True, gripper_penalty=-0.02),
            reset=ResetConfig(reset_time_s=0.0, control_time_s=1.0),
            inverse_kinematics=InverseKinematicsConfig(
                end_effector_step_sizes={"x": 0.01, "y": 0.01, "z": 0.01},
                end_effector_bounds={"min": [0.2, -0.2, 0.1], "max": [0.4, 0.2, 0.3]},
                delta_reference=delta_reference,
                **extra,
            ),
            max_gripper_pos=1.0,
        ),
    )


def test_cartesian_pipeline_policy_and_intervention(env, robot):
    teleop = FakeTeleop()
    env_processor, action_processor = make_processors(env, teleop, _cartesian_env_cfg(), device="cpu")
    transition = reset_and_build_transition(env, env_processor, action_processor)
    assert transition[TransitionKey.OBSERVATION]["observation.state"].shape == (1, 13)

    # Policy acts: +x at full scale, gripper "stay".
    transition = step_env_and_process_transition(
        env, transition, torch.tensor([1.0, 0.0, 0.0, 1.0]), env_processor, action_processor
    )
    assert robot.sent[-1] == pytest.approx(
        {"ee.x": 0.31, "ee.y": 0.0, "ee.z": 0.2, **DOWN, "gripper.pos": 0.0}
    )
    assert not transition[TransitionKey.INFO][TeleopEvents.IS_INTERVENTION]

    # Human intervenes: +y and close the gripper. The executed (teleop) delta is what HIL-SERL stores.
    teleop.intervention = True
    teleop.action = {"delta_x": 0.0, "delta_y": 1.0, "delta_z": 0.0, "gripper": 0}
    transition = step_env_and_process_transition(
        env, transition, torch.tensor([1.0, 0.0, 0.0, 1.0]), env_processor, action_processor
    )
    assert robot.sent[-1] == pytest.approx(
        {"ee.x": 0.3, "ee.y": 0.01, "ee.z": 0.2, **DOWN, "gripper.pos": 1.0}
    )
    assert transition[TransitionKey.INFO][TeleopEvents.IS_INTERVENTION]
    torch.testing.assert_close(
        transition[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"], torch.tensor([0.0, 1.0, 0.0, 0.0])
    )
    # Closing an open gripper costs the grasp penalty.
    assert transition[TransitionKey.COMPLEMENTARY_DATA]["discrete_penalty"] == pytest.approx(-0.02)


def test_six_dof_ee_frame_pipeline(robot):
    env = RobotEnv(robot=robot, use_gripper=True, reset_pose=HOME_JOINTS, reset_time_s=0.0, use_rotation=True)
    teleop = FakeTeleop()
    teleop.action = {
        "delta_x": 0.0,
        "delta_y": 0.0,
        "delta_z": 0.0,
        "delta_rx": 0.0,
        "delta_ry": 0.0,
        "delta_rz": 0.0,
        "gripper": 1,
    }
    env_processor, action_processor = make_processors(
        env, teleop, _cartesian_env_cfg(six_dof=True), device="cpu"
    )
    transition = reset_and_build_transition(env, env_processor, action_processor)
    # Relative to the reset pose.
    state = transition[TransitionKey.OBSERVATION]["observation.state"]
    torch.testing.assert_close(state, torch.zeros(1, 13))

    # Policy: +z in the end-effector frame of a gripper pointing down is base -z.
    step_env_and_process_transition(
        env, transition, torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]), env_processor, action_processor
    )
    assert robot.sent[-1] == pytest.approx(
        {"ee.x": 0.3, "ee.y": 0.0, "ee.z": 0.19, **DOWN, "gripper.pos": 0.0}
    )

    # The arm moved down 1 cm: the relative state shows +z in the reset frame.
    robot.state["ee.z"] = 0.19
    transition = step_env_and_process_transition(
        env, transition, torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), env_processor, action_processor
    )
    assert transition[TransitionKey.OBSERVATION]["observation.state"][0, :3].tolist() == pytest.approx(
        [0.0, 0.0, 0.01], abs=1e-6
    )

    # Intervention: the operator pushes base +y; it is stored in the end-effector frame (-y) and executed
    # as base +y.
    teleop.intervention = True
    teleop.action = {**teleop.action, "delta_y": 1.0}
    transition = step_env_and_process_transition(
        env, transition, torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), env_processor, action_processor
    )
    torch.testing.assert_close(
        transition[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"],
        torch.tensor([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    )
    assert robot.sent[-1]["ee.y"] == pytest.approx(0.01)


def test_cartesian_pipeline_time_limit(env, robot):
    env_processor, action_processor = make_processors(env, FakeTeleop(), _cartesian_env_cfg(), device="cpu")
    transition = reset_and_build_transition(env, env_processor, action_processor)
    for _ in range(10):  # control_time_s * fps
        transition = step_env_and_process_transition(
            env, transition, torch.tensor([0.0, 0.0, 0.0, 1.0]), env_processor, action_processor
        )
    assert transition[TransitionKey.TRUNCATED]


def test_example_configs_parse_through_entry_points():
    """The xArm example configs must parse with only the RL entry modules imported (camera registration)."""
    import subprocess
    import sys
    from pathlib import Path

    configs = Path(__file__).resolve().parents[2] / "examples" / "xarm_hilserl" / "configs"
    script = f"""
import draccus
import lerobot.rl.gym_manipulator as gym_manipulator
import lerobot.rl.learner  # noqa: F401
from lerobot.rl.train_rl import TrainRLServerPipelineConfig

env = draccus.parse(gym_manipulator.GymManipulatorConfig, config_path={str(configs / "env.json")!r}, args=[])
train = draccus.parse(TrainRLServerPipelineConfig, config_path={str(configs / "train.json")!r}, args=[])

from lerobot.cameras.realsense import RealSenseCameraConfig
assert isinstance(env.env.robot.cameras["wrist"], RealSenseCameraConfig)
assert isinstance(train.env.robot.cameras["wrist"], RealSenseCameraConfig)
assert train.policy.input_features["observation.state"].shape == (19,)
assert train.policy.output_features["action"].shape == (6,)
assert env.env.processor.inverse_kinematics.use_rotation and env.env.teleop.use_rotation
"""
    # The camera config is imported only after parsing, in a fresh interpreter, so nothing pre-registers it.
    result = subprocess.run([sys.executable, "-W", "ignore", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
