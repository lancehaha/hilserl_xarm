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


"""Processor steps for robots whose controller tracks Cartesian end-effector targets natively (no URDF IK)."""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ActionProcessorStep,
    ComplementaryDataProcessorStep,
    ObservationProcessorStep,
    PolicyAction,
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    TransitionKey,
)
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.rotation import Rotation

EE_POSITION_KEYS = ("ee.x", "ee.y", "ee.z")
# Roll / pitch / yaw in radians: fixed-axis X, Y, Z rotations (scipy's extrinsic "xyz" Euler angles).
EE_ORIENTATION_KEYS = ("ee.rx", "ee.ry", "ee.rz")
EE_POSE_KEYS = EE_POSITION_KEYS + EE_ORIENTATION_KEYS
GRIPPER_POS_KEY = "gripper.pos"

# Teleoperator delta keys: translation, then rotation (only emitted by rotation-capable teleoperators).
DELTA_POSITION_KEYS = ("delta_x", "delta_y", "delta_z")
DELTA_ROTATION_KEYS = ("delta_rx", "delta_ry", "delta_rz")

# Discrete gripper commands, see `GamepadTeleop.GripperAction` and `SpaceMouseTeleop`.
GRIPPER_CLOSE = 0
GRIPPER_OPEN = 2


def measured_pose(observation: dict[str, Any]) -> tuple[np.ndarray, Rotation]:
    """Measured end-effector position and orientation from a raw robot observation."""
    missing = [key for key in EE_POSE_KEYS if key not in observation]
    if missing:
        raise ValueError(f"The observation must contain the measured end-effector pose, missing {missing}.")
    position = np.array([float(observation[key]) for key in EE_POSITION_KEYS])
    orientation = Rotation.from_euler(
        "xyz", np.array([float(observation[key]) for key in EE_ORIENTATION_KEYS])
    )
    return position, orientation


def clip_rotation_to_box(rotation: Rotation, low: np.ndarray, high: np.ndarray) -> Rotation:
    """
    Clip the roll / pitch / yaw of a rotation to a box, as `FrankaEnv.clip_safety_box` in rail-berkeley/hil-serl.

    Roll is clipped by magnitude and keeps its sign: a gripper pointing down sits at roll = +-pi, where the
    Euler angle jumps between pi and -pi, so the bounds of roll apply to |roll|.
    """
    euler = rotation.as_euler("xyz")
    sign = np.sign(euler[0])
    euler[0] = sign * np.clip(np.abs(euler[0]), low[0], high[0])
    euler[1:] = np.clip(euler[1:], low[1:], high[1:])
    return Rotation.from_euler("xyz", euler)


@ProcessorStepRegistry.register("cartesian_delta_to_target")
@dataclass
class CartesianDeltaToTargetStep(RobotActionProcessorStep):
    """
    Turns a normalized end-effector delta into an absolute Cartesian target for a robot that does its own IK.

    Takes the output of `MapDeltaActionToRobotActionStep` and plays the role of `EEReferenceAndDelta` +
    `EEBoundsAndSafety` + `GripperVelocityToJoint` for robots whose observation already holds the measured
    end-effector pose (`ee.x`, `ee.y`, `ee.z`, `ee.rx`, `ee.ry`, `ee.rz`) and the normalized gripper opening
    (`gripper.pos`). It follows `FrankaEnv.step` and `RelativeFrame` of rail-berkeley/hil-serl:

    1. With `action_frame="ee"`, the delta is expressed in the end-effector frame and rotated into the base
       frame with the measured orientation, then clipped to [-1, 1].
    2. Target position = reference + delta[:3] * `end_effector_step_sizes`. With `use_rotation`, target
       orientation = Rotation.from_rotvec(delta[3:] * `end_effector_rot_step`) * reference orientation;
       otherwise the orientation measured after the reset is held.
    3. The target is clipped to `end_effector_bounds` (hil-serl's safety box, `ABS_POSE_LIMIT_LOW/HIGH`).
    4. With `reference_clip`, the target is kept within that offset from the measured pose, like the
       `translational_clip_*` / `rotational_clip_*` reference limiting of hil-serl's impedance controller, which
       bounds the force the arm applies when it is blocked.

    While there is no input (`enabled` is false) the last target is held.

    Attributes:
        end_effector_step_sizes: Displacement (m) for a full-scale delta, per axis ("x", "y", "z").
        end_effector_bounds: Optional "min" / "max" lists the target is clipped to: xyz (m), optionally followed
            by roll / pitch / yaw (rad, roll by magnitude).
        delta_reference: Pose the delta is applied to.
            "measured": the measured end-effector pose, as in rail-berkeley/hil-serl. The target never leads
                the arm by more than one step, so the arm stops right away when the input stops.
            "commanded": the previous target, as `EEReferenceAndDelta(use_ik_solution=True)` does in the SO
                pipeline. Motion runs at exactly `step size * fps` but the target can run ahead of an arm
                that lags or is blocked.
        discrete_gripper: Map the discrete gripper command {0 = close, 1 = stay, 2 = open} to a latched
            binary target (1 = closed, 0 = open). "Stay" keeps the last target, so a grasp keeps its force.
        use_rotation: Apply the rotation deltas (`target_wx`, `target_wy`, `target_wz`), 6-DoF control.
        end_effector_rot_step: Rotation (rad) for a full-scale rotation delta.
        action_frame: Frame of the delta, "base" or "ee" (end-effector frame, hil-serl's `RelativeFrame`).
        reference_clip: Optional "min" / "max" lists bounding target - measured: xyz (m), optionally followed
            by a rotation vector (rad).
    """

    end_effector_step_sizes: dict[str, float]
    end_effector_bounds: dict[str, list[float]] | None = None
    delta_reference: Literal["measured", "commanded"] = "measured"
    discrete_gripper: bool = True
    use_rotation: bool = False
    end_effector_rot_step: float = 0.0
    action_frame: Literal["base", "ee"] = "base"
    reference_clip: dict[str, list[float]] | None = None
    _last_position: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_orientation: Rotation | None = field(default=None, init=False, repr=False)
    _gripper_target: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.delta_reference not in ("measured", "commanded"):
            raise ValueError(
                f"`delta_reference` must be 'measured' or 'commanded', got '{self.delta_reference}'."
            )
        if self.action_frame not in ("base", "ee"):
            raise ValueError(f"`action_frame` must be 'base' or 'ee', got '{self.action_frame}'.")
        if not self.discrete_gripper:
            raise NotImplementedError("Only the discrete gripper command is supported.")
        if self.use_rotation and self.end_effector_rot_step <= 0:
            raise ValueError("`end_effector_rot_step` must be > 0 with `use_rotation`.")
        for name, limits in (
            ("end_effector_bounds", self.end_effector_bounds),
            ("reference_clip", self.reference_clip),
        ):
            if limits is not None and any(len(limits[key]) not in (3, 6) for key in ("min", "max")):
                raise ValueError(f"`{name}` 'min' / 'max' must have 3 (xyz) or 6 (xyz + rotation) values.")

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("The observation must contain the measured end-effector pose.")
        position, orientation = measured_pose(observation)

        enabled = bool(action.pop("enabled"))
        delta = np.array([float(action.pop(f"target_{axis}")) for axis in "xyz"])
        delta_rot = np.array([float(action.pop(f"target_w{axis}", 0.0)) for axis in "xyz"])
        if not self.use_rotation:
            delta_rot[:] = 0.0
        gripper_command = round(float(action.pop("gripper_vel")))

        if enabled or self._last_position is None:
            if not enabled:
                delta[:], delta_rot[:] = 0.0, 0.0
            if self.action_frame == "ee":
                # `RelativeFrame.transform_action`, then the [-1, 1] clip of `FrankaEnv.step`.
                delta = np.clip(orientation.apply(delta), -1.0, 1.0)
                delta_rot = np.clip(orientation.apply(delta_rot), -1.0, 1.0)

            reference_position, reference_orientation = position, orientation
            if (
                self.delta_reference == "commanded"
                and self._last_position is not None
                and self._last_orientation is not None
            ):
                reference_position, reference_orientation = self._last_position, self._last_orientation

            step_sizes = np.array([self.end_effector_step_sizes[axis] for axis in "xyz"])
            target_position = reference_position + delta * step_sizes
            if self.use_rotation:
                target_orientation = (
                    Rotation.from_rotvec(delta_rot * self.end_effector_rot_step) * reference_orientation
                )
            else:
                # 3-DoF control holds the orientation measured after the reset.
                target_orientation = (
                    self._last_orientation if self._last_orientation is not None else orientation
                )

            if self.end_effector_bounds is not None:
                low, high = (np.asarray(self.end_effector_bounds[key], dtype=float) for key in ("min", "max"))
                target_position = np.clip(target_position, low[:3], high[:3])
                if self.use_rotation and len(low) == 6:
                    target_orientation = clip_rotation_to_box(target_orientation, low[3:], high[3:])

            if self.reference_clip is not None:
                low, high = (np.asarray(self.reference_clip[key], dtype=float) for key in ("min", "max"))
                target_position = position + np.clip(target_position - position, low[:3], high[:3])
                if self.use_rotation and len(low) == 6:
                    offset = np.clip((target_orientation * orientation.inv()).as_rotvec(), low[3:], high[3:])
                    target_orientation = Rotation.from_rotvec(offset) * orientation

            self._last_position = target_position
            self._last_orientation = target_orientation

        if gripper_command == GRIPPER_CLOSE:
            self._gripper_target = 1.0
        elif gripper_command == GRIPPER_OPEN:
            self._gripper_target = 0.0
        elif self._gripper_target is None:
            self._gripper_target = float(observation.get(GRIPPER_POS_KEY, 0.0))

        assert self._last_orientation is not None
        target_pose = np.concatenate([self._last_position, self._last_orientation.as_euler("xyz")])
        return {
            **{key: float(value) for key, value in zip(EE_POSE_KEYS, target_pose, strict=True)},
            GRIPPER_POS_KEY: self._gripper_target,
        }

    def reset(self):
        self._last_position = None
        self._last_orientation = None
        self._gripper_target = None

    def get_config(self) -> dict[str, Any]:
        return {
            "end_effector_step_sizes": self.end_effector_step_sizes,
            "end_effector_bounds": self.end_effector_bounds,
            "delta_reference": self.delta_reference,
            "discrete_gripper": self.discrete_gripper,
            "use_rotation": self.use_rotation,
            "end_effector_rot_step": self.end_effector_rot_step,
            "action_frame": self.action_frame,
            "reference_clip": self.reference_clip,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["enabled", "target_x", "target_y", "target_z", "target_wx", "target_wy", "target_wz"]:
            features[PipelineFeatureType.ACTION].pop(feat, None)
        features[PipelineFeatureType.ACTION].pop("gripper_vel", None)
        for key in (*EE_POSE_KEYS, GRIPPER_POS_KEY):
            features[PipelineFeatureType.ACTION][key] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        return features


@ProcessorStepRegistry.register("teleop_delta_to_ee_frame")
@dataclass
class TeleopDeltaToEEFrameStep(ComplementaryDataProcessorStep):
    """
    Rotates a teleoperator's base-frame delta into the end-effector frame.

    The teleoperator (e.g. a SpaceMouse) commands motion in the robot base frame, while a policy with
    `CartesianDeltaToTargetStep(action_frame="ee")` acts in the end-effector frame. As `RelativeFrame` does with
    `intervene_action` in rail-berkeley/hil-serl, this converts the teleop delta before
    `InterventionActionProcessorStep` turns it into the stored action, so recorded and executed interventions
    are in the policy's action frame. The result is clipped to [-1, 1] (the policy's action range), and the
    clipped delta is what gets executed.

    Place it after `AddTeleopActionAsComplimentaryDataStep`. The observation must be the raw robot observation
    holding the measured pose.
    """

    def complementary_data(self, complementary_data: dict) -> dict:
        teleop_action = complementary_data.get(TELEOP_ACTION_KEY)
        if not isinstance(teleop_action, dict):
            return complementary_data
        observation = self.transition.get(TransitionKey.OBSERVATION) or {}
        _, orientation = measured_pose(observation)
        base_to_ee = orientation.inv()

        new_action = dict(teleop_action)
        for keys in (DELTA_POSITION_KEYS, DELTA_ROTATION_KEYS):
            if all(key in teleop_action for key in keys):
                vector = np.array([float(teleop_action[key]) for key in keys])
                rotated = np.clip(base_to_ee.apply(vector), -1.0, 1.0)
                new_action.update(zip(keys, rotated.tolist(), strict=True))

        new_complementary_data = dict(complementary_data)
        new_complementary_data[TELEOP_ACTION_KEY] = new_action
        return new_complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("ee_relative_observation")
@dataclass
class EERelativeObservationStep(ObservationProcessorStep):
    """
    Expresses the end-effector state relative to the reset pose and the end-effector frame.

    The counterpart of `RelativeFrame` + `Quat2EulerWrapper` of rail-berkeley/hil-serl, for an
    `observation.state` that starts with the base-frame pose [x, y, z, roll, pitch, yaw] followed by the
    base-frame twist [vx, vy, vz, wx, wy, wz] (the layout of `XArm.observation_features`). It rewrites:

    - the pose as the pose relative to the first pose seen after `reset()` (position in the reset frame, then
      roll / pitch / yaw of the relative rotation), so states do not depend on the random reset offset;
    - the twist in the current end-effector frame.

    The rest of the state (force, torque, gripper) is left as is.
    """

    _reset_position: np.ndarray | None = field(default=None, init=False, repr=False)
    _reset_orientation: Rotation | None = field(default=None, init=False, repr=False)

    def observation(self, observation: dict) -> dict:
        state = observation.get(OBS_STATE)
        if state is None:
            raise ValueError(f"{OBS_STATE} is not in observation")
        if state.shape[-1] < 12:
            raise ValueError(
                f"{OBS_STATE} must start with the 6-D pose and 6-D twist, got shape {state.shape}."
            )

        values = state.detach().cpu().numpy().astype(np.float64).reshape(-1)
        position, orientation = values[:3], Rotation.from_euler("xyz", values[3:6])
        if self._reset_position is None:
            self._reset_position, self._reset_orientation = position.copy(), orientation
        assert self._reset_orientation is not None

        reset_inv = self._reset_orientation.inv()
        relative_position = reset_inv.apply(position - self._reset_position)
        relative_rpy = (reset_inv * orientation).as_euler("xyz")
        ee_inv = orientation.inv()
        twist = np.concatenate([ee_inv.apply(values[6:9]), ee_inv.apply(values[9:12])])

        new_values = np.concatenate([relative_position, relative_rpy, twist, values[12:]])
        new_observation = dict(observation)
        new_observation[OBS_STATE] = torch.as_tensor(
            new_values, dtype=state.dtype, device=state.device
        ).reshape(state.shape)
        return new_observation

    def reset(self) -> None:
        self._reset_position = None
        self._reset_orientation = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("robot_action_to_tensor")
@dataclass
class RobotActionToTensorStep(ActionProcessorStep):
    """
    Stacks a robot action dictionary into a tensor, in the order of `action_keys`.

    The counterpart of `RobotActionToPolicyActionProcessorStep` for robots whose action keys are not
    `{motor}.pos` (e.g. `ee.x`). Typically `action_keys = list(robot.action_features)`.
    """

    action_keys: list[str]

    def action(self, action: RobotAction) -> PolicyAction:
        missing = [key for key in self.action_keys if key not in action]
        if missing:
            raise ValueError(f"Action is missing the keys {missing}.")
        return torch.tensor([float(action[key]) for key in self.action_keys], dtype=torch.float32)

    def get_config(self) -> dict[str, Any]:
        return {"action_keys": self.action_keys}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features[PipelineFeatureType.ACTION][ACTION] = PolicyFeature(
            type=FeatureType.ACTION, shape=(len(self.action_keys),)
        )
        return features
