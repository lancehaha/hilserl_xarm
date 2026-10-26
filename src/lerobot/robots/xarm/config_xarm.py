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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("xarm")
@dataclass
class XArmConfig(RobotConfig):
    """Configuration for a UFACTORY xArm controlled in Cartesian end-effector space with an xArm Gripper."""

    # IP address of the xArm control box
    ip: str = "192.168.1.235"

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Speed and acceleration limits passed to every Cartesian target (xArm online trajectory planning mode).
    # Keep the speed above `end_effector_step_sizes * fps` so the arm can track the commanded targets.
    max_linear_speed_mm_s: float = 200.0
    max_linear_acc_mm_s2: float = 2000.0

    # Read the UFACTORY Six-axis F/T sensor and add the external force/torque to the observation
    # (`tcp_force` / `tcp_torque` of rail-berkeley/hil-serl). The sensor is zeroed after every reset, so run
    # the load identification in UFACTORY Studio first.
    use_ft_sensor: bool = False
    # Frame of the force / torque observation: "tool" (`ft_ext_force` as reported, as the Franka
    # `K_F_ext_hat_K` used by hil-serl) or "base" (rotated into the base frame with the TCP orientation).
    ft_sensor_frame: str = "tool"

    # Compliance during episodes, the counterpart of hil-serl's Cartesian impedance controller with
    # `COMPLIANCE_PARAM`: the controller's F/T admittance mode, switched off (stiff, as `PRECISION_PARAM`) while
    # resetting. Requires `use_ft_sensor`. Ranges from the UFACTORY SDK: stiffness 0-2000 N/m and 0-20 Nm/rad,
    # mass 0.02-1 kg and inertia 1e-4-0.01 kg m^2. Damping is set by the controller to 2 * sqrt(M * K).
    use_admittance: bool = False
    admittance_stiffness: list[float] = field(
        default_factory=lambda: [2000.0, 2000.0, 2000.0, 20.0, 20.0, 20.0]
    )
    admittance_mass: list[float] = field(default_factory=lambda: [0.06, 0.06, 0.06, 0.0006, 0.0006, 0.0006])
    # 1 = compliant along / about that axis of `admittance_frame` (0 = base, 1 = tool).
    admittance_axes: list[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])
    admittance_frame: int = 0

    # Joint speed used to move to the home joint configuration on `reset_to_home`.
    home_joint_speed_deg_s: float = 30.0

    # xArm Gripper positions (SDK units) mapped to the normalized `gripper.pos` 0 (open) and 1 (closed).
    gripper_open_pos: int = 800
    gripper_close_pos: int = 0
    gripper_speed: int = 5000

    # Half-width of a uniform random xy offset (meters) added to the home pose on every reset, as
    # `RANDOM_RESET` in rail-berkeley/hil-serl (their tasks use 0.01-0.02). 0 disables it.
    random_xy_range_m: float = 0.0
    # Half-width of a uniform random yaw offset (radians) added on every reset, as `RANDOM_RZ_RANGE` in
    # rail-berkeley/hil-serl (their tasks use 0.05-0.1). 0 disables it.
    random_rz_range_rad: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.gripper_open_pos == self.gripper_close_pos:
            raise ValueError("`gripper_open_pos` and `gripper_close_pos` must differ.")
        if self.random_xy_range_m < 0:
            raise ValueError(f"`random_xy_range_m` must be >= 0, got {self.random_xy_range_m}.")
        if self.random_rz_range_rad < 0:
            raise ValueError(f"`random_rz_range_rad` must be >= 0, got {self.random_rz_range_rad}.")
        if self.ft_sensor_frame not in ("tool", "base"):
            raise ValueError(f"`ft_sensor_frame` must be 'tool' or 'base', got '{self.ft_sensor_frame}'.")
        if self.use_admittance and not self.use_ft_sensor:
            raise ValueError("`use_admittance` requires `use_ft_sensor`.")
        for name in ("admittance_stiffness", "admittance_mass", "admittance_axes"):
            if len(getattr(self, name)) != 6:
                raise ValueError(f"`{name}` must have 6 values (x, y, z, rx, ry, rz).")
