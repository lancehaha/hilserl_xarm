#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass

import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import PolicyAction, RobotAction

from .pipeline import ActionProcessorStep, ProcessorStepRegistry, RobotActionProcessorStep


@ProcessorStepRegistry.register("map_tensor_to_delta_action_dict")
@dataclass
class MapTensorToDeltaActionDictStep(ActionProcessorStep):
    """
    Maps a flat action tensor from a policy to a structured delta action dictionary.

    This step is typically used after a policy outputs a continuous action vector.
    It decomposes the vector into named components for delta movements of the
    end-effector (x, y, z) and optionally the gripper.

    Attributes:
        use_gripper: If True, assumes the element after the deltas is the
                     gripper action.
        use_rotation: If True, elements 3-5 are the rotation deltas
                      (`delta_rx`, `delta_ry`, `delta_rz`), for 6-DoF control.
    """

    use_gripper: bool = True
    use_rotation: bool = False

    def action(self, action: PolicyAction) -> RobotAction:
        if not isinstance(action, torch.Tensor):
            raise ValueError("Only PolicyAction is supported for this processor")

        if action.dim() > 1:
            action = action.squeeze(0)

        delta_action = {
            "delta_x": action[0].item(),
            "delta_y": action[1].item(),
            "delta_z": action[2].item(),
        }
        gripper_index = 3
        if self.use_rotation:
            delta_action.update(
                {
                    "delta_rx": action[3].item(),
                    "delta_ry": action[4].item(),
                    "delta_rz": action[5].item(),
                }
            )
            gripper_index = 6
        if self.use_gripper:
            delta_action["gripper"] = action[gripper_index].item()
        return delta_action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        axes = ["x", "y", "z", "rx", "ry", "rz"] if self.use_rotation else ["x", "y", "z"]
        for axis in axes:
            features[PipelineFeatureType.ACTION][f"delta_{axis}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        if self.use_gripper:
            features[PipelineFeatureType.ACTION]["gripper"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )
        return features


@ProcessorStepRegistry.register("map_delta_action_to_robot_action")
@dataclass
class MapDeltaActionToRobotActionStep(RobotActionProcessorStep):
    """
    Maps delta actions from teleoperators to robot target actions for inverse kinematics.

    This step converts a dictionary of delta movements (e.g., from a gamepad)
    into a target action format that includes an "enabled" flag and target
    end-effector positions. It also handles scaling and noise filtering.

    Attributes:
        position_scale: A factor to scale the delta position inputs.
        noise_threshold: The magnitude below which delta inputs are considered noise
                         and do not trigger an "enabled" state.
    """

    # Scale factors for delta movements
    position_scale: float = 1.0
    noise_threshold: float = 1e-3  # 1 mm threshold to filter out noise

    def action(self, action: RobotAction) -> RobotAction:
        # NOTE (maractingi): Action can be a dict from the teleop_devices or a tensor from the policy
        # TODO (maractingi): changing this target_xyz naming convention from the teleop_devices
        delta_x = action.pop("delta_x")
        delta_y = action.pop("delta_y")
        delta_z = action.pop("delta_z")
        gripper = action.pop("gripper")
        # Rotation deltas are only present for 6-DoF teleoperators / policies.
        target_wx = float(action.pop("delta_rx", 0.0))
        target_wy = float(action.pop("delta_ry", 0.0))
        target_wz = float(action.pop("delta_rz", 0.0))

        # Determine if the teleoperator is actively providing input
        # Consider enabled if any significant movement delta is detected
        position_magnitude = (delta_x**2 + delta_y**2 + delta_z**2) ** 0.5  # Use Euclidean norm for position
        rotation_magnitude = (target_wx**2 + target_wy**2 + target_wz**2) ** 0.5
        enabled = (
            max(position_magnitude, rotation_magnitude) > self.noise_threshold
        )  # Small threshold to avoid noise

        # Scale the deltas appropriately
        scaled_delta_x = delta_x * self.position_scale
        scaled_delta_y = delta_y * self.position_scale
        scaled_delta_z = delta_z * self.position_scale

        # Update action with robot target format
        action = {
            "enabled": enabled,
            "target_x": scaled_delta_x,
            "target_y": scaled_delta_y,
            "target_z": scaled_delta_z,
            "target_wx": target_wx,
            "target_wy": target_wy,
            "target_wz": target_wz,
            "gripper_vel": float(gripper),
        }

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            features[PipelineFeatureType.ACTION].pop(f"delta_{axis}", None)
        features[PipelineFeatureType.ACTION].pop("gripper", None)

        for feat in [
            "enabled",
            "target_x",
            "target_y",
            "target_z",
            "target_wx",
            "target_wy",
            "target_wz",
            "gripper_vel",
        ]:
            features[PipelineFeatureType.ACTION][f"{feat}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features
