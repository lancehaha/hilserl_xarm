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

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig

SPACE_MOUSE_AXES = ("x", "y", "z")
SPACE_MOUSE_ROTATION_AXES = ("roll", "pitch", "yaw")


@TeleoperatorConfig.register_subclass("spacemouse")
@dataclass
class SpaceMouseTeleopConfig(TeleoperatorConfig):
    use_gripper: bool = True
    # Axis readings below this magnitude (in [0, 1]) are treated as zero.
    deadzone: float = 0.05
    # Multiplier applied to the axis readings before clipping deltas to [-1, 1].
    scale: float = 1.0
    # SpaceMouse axis driving each of (delta_x, delta_y, delta_z). Prefix with "-" to invert, e.g.
    # ["y", "-x", "z"]. pyspacemouse axes: x = right, y = forward (away from the cable), z = up.
    axis_mapping: list[str] = field(default_factory=lambda: ["x", "y", "z"])
    # Also emit rotation deltas (`delta_rx`, `delta_ry`, `delta_rz`, about the robot base X, Y, Z axes) from the
    # puck tilt / twist, for 6-DoF control as in rail-berkeley/hil-serl.
    use_rotation: bool = False
    # Puck rotation driving each of (delta_rx, delta_ry, delta_rz): "roll" / "pitch" / "yaw" are rotations about
    # the puck x (right) / y (forward) / z (up) axes, right-handed. "-" to invert, "0" to disable. If unset, it
    # follows `axis_mapping`: rotating the puck about the axis you push along rotates the TCP about the robot
    # axis that push moves along.
    rotation_mapping: list[str] | None = None
    # Rotation readings below this magnitude are treated as zero: tilting is easy to trigger while translating.
    rotation_deadzone: float = 0.15
    rotation_scale: float = 1.0
    # Index of the device to open when several SpaceMice are connected.
    device_index: int = 0
    # Intervene on every step with SpaceMouse input (puck outside the deadzone or a button held), as in
    # rail-berkeley/hil-serl. Space still latches the intervention on, e.g. to make the arm hold still.
    intervene_on_input: bool = True

    def __post_init__(self):
        if len(self.axis_mapping) != 3:
            raise ValueError(f"`axis_mapping` must have 3 entries, got {self.axis_mapping}.")
        for axis in self.axis_mapping:
            if axis.removeprefix("-") not in SPACE_MOUSE_AXES:
                raise ValueError(
                    f"Invalid axis '{axis}' in `axis_mapping`, expected one of {SPACE_MOUSE_AXES} "
                    "optionally prefixed with '-'."
                )
        if not 0.0 <= self.deadzone < 1.0:
            raise ValueError(f"`deadzone` must be in [0, 1), got {self.deadzone}.")
        if self.rotation_mapping is not None and len(self.rotation_mapping) != 3:
            raise ValueError(f"`rotation_mapping` must have 3 entries, got {self.rotation_mapping}.")
        for axis in self.rotation_mapping or []:
            if axis != "0" and axis.removeprefix("-") not in SPACE_MOUSE_ROTATION_AXES:
                raise ValueError(
                    f"Invalid axis '{axis}' in `rotation_mapping`, expected one of {SPACE_MOUSE_ROTATION_AXES} "
                    "optionally prefixed with '-', or '0'."
                )
        if not 0.0 <= self.rotation_deadzone < 1.0:
            raise ValueError(f"`rotation_deadzone` must be in [0, 1), got {self.rotation_deadzone}.")
