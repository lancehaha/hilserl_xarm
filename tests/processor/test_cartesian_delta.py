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

import numpy as np
import pytest
import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    MapDeltaActionToRobotActionStep,
    RobotProcessorPipeline,
    TransitionKey,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from lerobot.robots.xarm.robot_cartesian_processor import (
    CartesianDeltaToTargetStep,
    EERelativeObservationStep,
    RobotActionToTensorStep,
    TeleopDeltaToEEFrameStep,
    clip_rotation_to_box,
)
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.rotation import Rotation

STEP = 0.01
ROT_STEP = 0.1
BOUNDS = {"min": [0.2, -0.2, 0.1], "max": [0.4, 0.2, 0.3]}
CLOSE, STAY, OPEN = 0, 1, 2
# Gripper pointing down: roll = pi.
DOWN = (math.pi, 0.0, 0.0)


def _make_pipeline(delta_reference: str = "measured", **step_kwargs):
    step_kwargs.setdefault("end_effector_bounds", BOUNDS)
    return RobotProcessorPipeline(
        steps=[
            MapDeltaActionToRobotActionStep(),
            CartesianDeltaToTargetStep(
                end_effector_step_sizes={"x": STEP, "y": STEP, "z": STEP},
                delta_reference=delta_reference,
                **step_kwargs,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )


def _obs(x=0.3, y=0.0, z=0.2, gripper=0.0, rpy=DOWN):
    return {
        "ee.x": x,
        "ee.y": y,
        "ee.z": z,
        **dict(zip(("ee.rx", "ee.ry", "ee.rz"), rpy, strict=True)),
        "gripper.pos": gripper,
    }


def _delta(dx=0.0, dy=0.0, dz=0.0, gripper=STAY, rot=None):
    delta = {"delta_x": dx, "delta_y": dy, "delta_z": dz, "gripper": gripper}
    if rot is not None:
        delta.update(zip(("delta_rx", "delta_ry", "delta_rz"), rot, strict=True))
    return delta


def _xyz(action):
    return [action["ee.x"], action["ee.y"], action["ee.z"]]


def _rotation(action):
    return Rotation.from_euler("xyz", [action["ee.rx"], action["ee.ry"], action["ee.rz"]])


def _same_rotation(a: Rotation, b: Rotation) -> bool:
    return np.allclose(a.as_matrix(), b.as_matrix(), atol=1e-9)


def test_measured_reference_follows_the_arm():
    pipeline = _make_pipeline("measured")
    assert _xyz(pipeline((_delta(dx=1.0), _obs()))) == pytest.approx([0.31, 0.0, 0.2])
    # The arm has not moved yet: the target stays one step ahead of the measured pose.
    assert _xyz(pipeline((_delta(dx=1.0), _obs()))) == pytest.approx([0.31, 0.0, 0.2])
    assert _xyz(pipeline((_delta(dy=-0.5), _obs(x=0.305)))) == pytest.approx([0.305, -0.005, 0.2])


def test_commanded_reference_accumulates_targets():
    pipeline = _make_pipeline("commanded")
    assert _xyz(pipeline((_delta(dx=1.0), _obs()))) == pytest.approx([0.31, 0.0, 0.2])
    assert _xyz(pipeline((_delta(dx=1.0), _obs()))) == pytest.approx([0.32, 0.0, 0.2])


@pytest.mark.parametrize("delta_reference", ["measured", "commanded"])
def test_no_input_holds_last_target(delta_reference):
    pipeline = _make_pipeline(delta_reference)
    pipeline((_delta(dz=1.0), _obs()))
    # No input while the arm still moves: the last target is kept.
    assert _xyz(pipeline((_delta(), _obs(z=0.205)))) == pytest.approx([0.3, 0.0, 0.21])


def test_first_step_without_input_holds_measured_pose():
    pipeline = _make_pipeline()
    assert _xyz(pipeline((_delta(), _obs(x=0.25, y=0.1, z=0.15)))) == pytest.approx([0.25, 0.1, 0.15])


def test_target_is_clipped_to_bounds():
    pipeline = _make_pipeline("commanded")
    for _ in range(20):
        action = pipeline((_delta(dx=1.0, dz=-1.0), _obs(x=0.39, z=0.11)))
    assert _xyz(action) == pytest.approx([0.4, 0.0, 0.1])


def test_discrete_gripper_is_latched():
    pipeline = _make_pipeline()
    assert pipeline((_delta(gripper=STAY), _obs(gripper=0.02)))["gripper.pos"] == pytest.approx(0.02)
    assert pipeline((_delta(gripper=CLOSE), _obs()))["gripper.pos"] == 1.0
    # "Stay" keeps squeezing even though the fingers stopped half-way on an object.
    assert pipeline((_delta(gripper=STAY), _obs(gripper=0.6)))["gripper.pos"] == 1.0
    assert pipeline((_delta(gripper=OPEN), _obs(gripper=0.6)))["gripper.pos"] == 0.0


def test_reset_restarts_from_measured_pose():
    pipeline = _make_pipeline("commanded")
    pipeline((_delta(dx=1.0, gripper=CLOSE), _obs()))
    pipeline.reset()
    action = pipeline((_delta(), _obs(x=0.25, gripper=0.0)))
    assert _xyz(action) == pytest.approx([0.25, 0.0, 0.2])
    assert action["gripper.pos"] == 0.0


def test_invalid_delta_reference():
    with pytest.raises(ValueError):
        CartesianDeltaToTargetStep(
            end_effector_step_sizes={"x": STEP, "y": STEP, "z": STEP}, delta_reference="x"
        )


def test_missing_measured_pose_raises():
    with pytest.raises(ValueError):
        _make_pipeline()((_delta(), {"gripper.pos": 0.0}))


def test_three_dof_holds_orientation_measured_after_reset():
    pipeline = _make_pipeline()
    first = pipeline((_delta(dx=1.0, rot=(1.0, 0.0, 0.0)), _obs(rpy=(math.pi, 0.0, 0.2))))
    # Rotation deltas are ignored and the arm drifting in orientation does not move the held orientation.
    second = pipeline((_delta(dx=1.0), _obs(rpy=(math.pi, 0.05, 0.25))))
    for action in (first, second):
        assert _same_rotation(_rotation(action), Rotation.from_euler("xyz", [math.pi, 0.0, 0.2]))


def test_rotation_is_applied_in_base_frame():
    pipeline = _make_pipeline(use_rotation=True, end_effector_rot_step=ROT_STEP)
    action = pipeline((_delta(rot=(0.0, 0.0, 1.0)), _obs(rpy=DOWN)))
    # hil-serl: Rotation.from_rotvec(delta * scale) * current.
    expected = Rotation.from_rotvec(np.array([0.0, 0.0, ROT_STEP])) * Rotation.from_euler("xyz", DOWN)
    assert _same_rotation(_rotation(action), expected)
    assert _xyz(action) == pytest.approx([0.3, 0.0, 0.2])


def test_ee_frame_action_is_rotated_into_base_frame():
    pipeline = _make_pipeline(use_rotation=True, end_effector_rot_step=ROT_STEP, action_frame="ee")
    # Gripper pointing down: EE +z is base -z, EE +y is base -y.
    action = pipeline((_delta(dy=1.0, dz=1.0, rot=(0.0, 0.0, 1.0)), _obs(rpy=DOWN)))
    assert _xyz(action) == pytest.approx([0.3, -STEP, 0.2 - STEP])
    expected = Rotation.from_rotvec(np.array([0.0, 0.0, -ROT_STEP])) * Rotation.from_euler("xyz", DOWN)
    assert _same_rotation(_rotation(action), expected)


def test_rotation_bounds_clip_roll_by_magnitude():
    # |roll| in [pi - 0.1, pi], pitch in [-0.1, 0.1], yaw in [-0.3, 0.3].
    bounds = {"min": [*BOUNDS["min"], math.pi - 0.1, -0.1, -0.3], "max": [*BOUNDS["max"], math.pi, 0.1, 0.3]}
    pipeline = _make_pipeline(
        "commanded", use_rotation=True, end_effector_rot_step=ROT_STEP, end_effector_bounds=bounds
    )
    for _ in range(10):
        action = pipeline((_delta(rot=(1.0, 1.0, 1.0)), _obs(rpy=DOWN)))
    roll, pitch, yaw = _rotation(action).as_euler("xyz")
    assert abs(roll) == pytest.approx(math.pi - 0.1)
    assert abs(pitch) <= 0.1 + 1e-9 and abs(yaw) <= 0.3 + 1e-9


def test_clip_rotation_to_box_keeps_roll_sign():
    low, high = np.array([math.pi - 0.1, -0.1, -0.1]), np.array([math.pi, 0.1, 0.1])
    for roll in (math.pi - 0.05, -(math.pi - 0.05)):
        rotation = Rotation.from_euler("xyz", [roll, 0.0, 0.0])
        assert _same_rotation(clip_rotation_to_box(rotation, low, high), rotation)
    clipped = clip_rotation_to_box(Rotation.from_euler("xyz", [-(math.pi - 0.3), 0.0, 0.0]), low, high)
    assert clipped.as_euler("xyz")[0] == pytest.approx(-(math.pi - 0.1))


def test_reference_clip_limits_lead_over_measured_pose():
    clip = {
        "min": [-0.005, -0.005, -0.005, -0.02, -0.02, -0.02],
        "max": [0.005, 0.005, 0.005, 0.02, 0.02, 0.02],
    }
    pipeline = _make_pipeline(use_rotation=True, end_effector_rot_step=ROT_STEP, reference_clip=clip)
    action = pipeline((_delta(dx=1.0, rot=(0.0, 0.0, 1.0)), _obs()))
    assert _xyz(action) == pytest.approx([0.305, 0.0, 0.2])
    offset = (_rotation(action) * Rotation.from_euler("xyz", DOWN).inv()).as_rotvec()
    assert offset == pytest.approx([0.0, 0.0, 0.02])


def test_teleop_delta_to_ee_frame_round_trip():
    observation = _obs(rpy=(math.pi, 0.0, 0.5))
    teleop = {**_delta(dx=0.5, dy=-0.3, dz=0.2, rot=(0.0, 0.1, 0.4)), "gripper": CLOSE}
    step = TeleopDeltaToEEFrameStep()
    transition = {
        TransitionKey.OBSERVATION: observation,
        TransitionKey.COMPLEMENTARY_DATA: {TELEOP_ACTION_KEY: teleop},
    }
    ee_action = step(transition)[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_KEY]
    assert ee_action["gripper"] == CLOSE

    # Executing the EE-frame delta reproduces the base-frame motion the operator commanded.
    pipeline = _make_pipeline(use_rotation=True, end_effector_rot_step=ROT_STEP, action_frame="ee")
    base_pipeline = _make_pipeline(use_rotation=True, end_effector_rot_step=ROT_STEP)
    executed = pipeline((dict(ee_action), observation))
    commanded = base_pipeline((dict(teleop), observation))
    assert _xyz(executed) == pytest.approx(_xyz(commanded))
    assert _same_rotation(_rotation(executed), _rotation(commanded))


def test_ee_relative_observation():
    step = EERelativeObservationStep()
    reset_rpy = [math.pi, 0.0, 0.5]
    rest = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.7]  # force, torque, gripper: untouched
    reset_state = torch.tensor([0.3, 0.1, 0.2, *reset_rpy, *[0.0] * 6, *rest])
    out = step.observation({OBS_STATE: reset_state})[OBS_STATE]
    assert out.tolist() == pytest.approx([0.0] * 12 + rest, abs=1e-6)

    # Moving 1 cm down (base -z) is +z in the frame of a gripper pointing down.
    twist = [0.0, 0.0, -0.1, 0.0, 0.0, 0.2]
    state = torch.tensor([0.3, 0.1, 0.19, *reset_rpy, *twist, *rest])
    out = step.observation({OBS_STATE: state})[OBS_STATE].tolist()
    assert out[:6] == pytest.approx([0.0, 0.0, 0.01, 0.0, 0.0, 0.0], abs=1e-6)
    assert out[6:12] == pytest.approx([0.0, 0.0, 0.1, 0.0, 0.0, -0.2], abs=1e-6)

    step.reset()
    out = step.observation({OBS_STATE: state})[OBS_STATE].tolist()
    assert out[:6] == pytest.approx([0.0] * 6, abs=1e-6)


def test_cartesian_transform_features():
    step = CartesianDeltaToTargetStep(end_effector_step_sizes={"x": STEP, "y": STEP, "z": STEP})
    features = {
        PipelineFeatureType.ACTION: {
            key: PolicyFeature(type=FeatureType.ACTION, shape=(1,))
            for key in ["enabled", "target_x", "target_y", "target_z", "gripper_vel"]
        }
    }
    out = step.transform_features(features)
    assert set(out[PipelineFeatureType.ACTION]) == {
        "ee.x",
        "ee.y",
        "ee.z",
        "ee.rx",
        "ee.ry",
        "ee.rz",
        "gripper.pos",
    }


def test_robot_action_to_tensor_order():
    step = RobotActionToTensorStep(action_keys=["ee.x", "ee.y", "ee.z", "gripper.pos"])
    tensor = step.action({"gripper.pos": 1.0, "ee.z": 0.3, "ee.x": 0.1, "ee.y": 0.2})
    assert torch.equal(tensor, torch.tensor([0.1, 0.2, 0.3, 1.0]))

    with pytest.raises(ValueError):
        step.action({"ee.x": 0.1})

    out = step.transform_features({PipelineFeatureType.ACTION: {}})
    assert out[PipelineFeatureType.ACTION][ACTION].shape == (4,)
