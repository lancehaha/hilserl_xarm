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

"""
Teleoperate an xArm with a SpaceMouse in end-effector space, and find the values for the HIL-SERL configs.

The SpaceMouse deltas go through the same processor steps as in HIL-SERL (`MapDeltaActionToRobotActionStep` +
`CartesianDeltaToTargetStep`), so the arm moves exactly as it will when recording demos and intervening.
The pose the script starts from (after `--home_joints_deg`, if set) is the home of every episode. Jog the TCP around
the region the task needs; where you finish does not matter. On Ctrl+C the script prints the config values it found:
`end_effector_bounds` (the visited xyz box, plus +/- `--box_rad` roll / pitch / yaw around the home with
`--teleop.use_rotation=true`), the start joints as `fixed_reset_joint_positions` and the matching
`policy.dataset_stats` for the hil-serl state (`relative_ee_frame`): pose relative to the home, twist,
force / torque (with `--robot.use_ft_sensor=true`) and gripper. With `--write_configs=true` it then offers to write
them into `configs/env.json` and `configs/train.json`.

With `--robot.use_admittance=true` (requires `--robot.use_ft_sensor=true`) the arm is compliant as during HIL-SERL, so
contacts can be tried out; add `--reference_clip` to bound the push as in the configs. The F/T sensor is zeroed at the
start pose: don't touch the gripper while the script starts or after a fault.

The SpaceMouse moves the arm in the base frame here; during HIL-SERL its deltas are converted to the end-effector
frame before they reach the same steps, so the motion is the same.

Example:

```shell
uv run python examples/xarm_hilserl/teleoperate.py \
    --robot.ip=192.168.1.235 \
    --robot.cameras='{wrist: {type: intelrealsense, serial_number_or_name: "<serial>", width: 640, height: 480, fps: 30}}' \
    --teleop.axis_mapping='[x, y, z]' \
    --teleop.use_rotation=true \
    --robot.use_ft_sensor=true \
    --end_effector_step_sizes='{x: 0.015, y: 0.015, z: 0.015}' \
    --end_effector_rot_step=0.1 \
    --write_configs=true
```
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat

import numpy as np

from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.processor import (
    MapDeltaActionToRobotActionStep,
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.xarm import XArm, XArmConfig
from lerobot.robots.xarm.robot_cartesian_processor import EE_POSE_KEYS, CartesianDeltaToTargetStep
from lerobot.teleoperators.space_mouse import SpaceMouseTeleop, SpaceMouseTeleopConfig
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.rotation import Rotation
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data, shutdown_rerun

IMAGENET_STATS = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
CONFIGS_DIR = Path(__file__).parent / "configs"


@dataclass
class XArmTeleoperateConfig:
    robot: XArmConfig = field(default_factory=XArmConfig)
    teleop: SpaceMouseTeleopConfig = field(default_factory=SpaceMouseTeleopConfig)
    fps: int = 10
    # TCP displacement (meters) for a full SpaceMouse deflection, per control step. Use the same values in
    # `env.processor.inverse_kinematics.end_effector_step_sizes` and keep them fixed once demos are recorded.
    end_effector_step_sizes: dict[str, float] = field(
        default_factory=lambda: {"x": 0.01, "y": 0.01, "z": 0.01}
    )
    # Rotation (rad) for a full SpaceMouse tilt / twist, per control step (`--teleop.use_rotation=true`). Use the
    # same value in `env.processor.inverse_kinematics.end_effector_rot_step`.
    end_effector_rot_step: float = 0.1
    # Safety box while teleoperating (meters / radians, robot base frame, roll by magnitude). If unset, a box of
    # +/- `box_m` and +/- `box_rad` around the start pose is used. With `write_configs`, the written bounds are the
    # visited xyz range and +/- `box_rad` around the start pose (the home of every episode) for roll / pitch / yaw.
    end_effector_bounds: dict[str, list[float]] | None = None
    box_m: float = 0.30
    box_rad: float = 1.57
    # Range of the force (N) / torque (Nm) normalization in `policy.dataset_stats`, about the contact forces
    # expected during the task.
    force_range_n: float = 30.0
    torque_range_nm: float = 3.0
    # Optional "min" / "max" bounds on target - measured pose (xyz in m, then rotation vector in rad), as
    # `env.processor.inverse_kinematics.reference_clip`. With admittance, a blocked arm pushes at most about
    # `robot.admittance_stiffness` x this offset.
    reference_clip: dict[str, list[float]] | None = None
    # "measured" (hil-serl) or "commanded" (lerobot SO pipeline), see `CartesianDeltaToTargetStep`.
    delta_reference: str = "measured"
    # Joint angles (degrees) to move to before teleoperating, e.g. your `fixed_reset_joint_positions`.
    home_joints_deg: list[float] | None = None
    display_data: bool = False
    # On exit, offer to write the values found into the env and train configs (asks for confirmation).
    write_configs: bool = False
    env_config_path: Path = CONFIGS_DIR / "env.json"
    train_config_path: Path = CONFIGS_DIR / "train.json"

    def __post_init__(self):
        if not self.teleop.use_gripper:
            raise ValueError("`teleop.use_gripper` must be true: HIL-SERL uses the discrete gripper action.")


def bounded_rpy(orientation: Rotation) -> np.ndarray:
    """Roll / pitch / yaw with roll by magnitude, the coordinates of `clip_rotation_to_box`."""
    rpy = orientation.as_euler("xyz")
    rpy[0] = abs(rpy[0])
    return rpy


@dataclass
class VisitedRange:
    """
    The poses visited since the start: range of the absolute pose (bounds) and of the pose relative to the home
    (state stats). The first pose is the home, `home_joints` its joint angles (degrees).
    """

    home_joints: list[float]
    positions: list[np.ndarray] = field(default_factory=list)
    orientations: list[Rotation] = field(default_factory=list)
    pose_min: np.ndarray | None = None
    pose_max: np.ndarray | None = None

    def update(self, position: np.ndarray, orientation: Rotation) -> None:
        self.positions.append(position)
        self.orientations.append(orientation)
        pose = np.concatenate([position, bounded_rpy(orientation)])
        if self.pose_min is None or self.pose_max is None:
            self.pose_min, self.pose_max = pose.copy(), pose.copy()
        self.pose_min, self.pose_max = np.minimum(self.pose_min, pose), np.maximum(self.pose_max, pose)

    def relative_range(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Range of the visited poses relative to the first one, the home of every episode, with the transform of
        `EERelativeObservationStep`: the state the policy sees if it goes where you went.
        """
        home_position, home_inv = self.positions[0], self.orientations[0].inv()
        relative = np.array(
            [
                np.concatenate(
                    [home_inv.apply(position - home_position), (home_inv * orientation).as_euler("xyz")]
                )
                for position, orientation in zip(self.positions, self.orientations, strict=True)
            ]
        )
        return relative.min(axis=0), relative.max(axis=0)


def find_config_values(
    cfg: XArmTeleoperateConfig, robot: XArm, visited: VisitedRange
) -> tuple[dict, list[float], dict]:
    """Config values from this session: `inverse_kinematics`, home joints and `policy.dataset_stats`."""
    use_rotation = cfg.teleop.use_rotation
    n_pose = 6 if use_rotation else 3
    assert visited.pose_min is not None and visited.pose_max is not None
    relative_min, relative_max = visited.relative_range()
    # xyz: the workspace you visited. Roll / pitch / yaw: +/- `box_rad` around the home (the start pose), so the
    # policy can rotate both ways from where every episode starts. |roll| cannot exceed pi.
    bounds_min, bounds_max = visited.pose_min[:3].tolist(), visited.pose_max[:3].tolist()
    if use_rotation:
        home_rpy = bounded_rpy(visited.orientations[0])
        rpy_min, rpy_max = home_rpy - cfg.box_rad, home_rpy + cfg.box_rad
        rpy_min[0], rpy_max[0] = max(rpy_min[0], 0.0), min(rpy_max[0], np.pi)
        bounds_min += rpy_min.tolist()
        bounds_max += rpy_max.tolist()
        # The relative orientation in the state can reach +/- `box_rad` from the home.
        relative_min[3:6], relative_max[3:6] = -cfg.box_rad, cfg.box_rad
    inverse_kinematics = {
        "urdf_path": None,
        "end_effector_bounds": {
            "min": np.round(bounds_min, 4).tolist(),
            "max": np.round(bounds_max, 4).tolist(),
        },
        "end_effector_step_sizes": cfg.end_effector_step_sizes,
        "delta_reference": cfg.delta_reference,
        "use_rotation": use_rotation,
    }
    if use_rotation:
        inverse_kinematics["end_effector_rot_step"] = cfg.end_effector_rot_step
    if cfg.reference_clip is not None:
        inverse_kinematics["reference_clip"] = cfg.reference_clip
    home_joints = [round(a, 2) for a in visited.home_joints]

    # State layout of `XArm.observation_features` after `EERelativeObservationStep`: relative pose, twist in the
    # end-effector frame, [force, torque], gripper. The end-effector frame mixes the base axes, so every linear
    # (angular) velocity gets the largest speed at full deflection.
    linear_speed = round(max(cfg.end_effector_step_sizes.values()) * cfg.fps, 4)
    angular_speed = round((cfg.end_effector_rot_step if use_rotation else 0.1) * cfg.fps, 4)
    state_min = [*relative_min.round(4).tolist(), *[-linear_speed] * 3, *[-angular_speed] * 3]
    state_max = [*relative_max.round(4).tolist(), *[linear_speed] * 3, *[angular_speed] * 3]
    if not use_rotation:
        # The orientation is held, keep a non-degenerate range for the normalization.
        state_min[3:6], state_max[3:6] = [-0.1] * 3, [0.1] * 3
    if cfg.robot.use_ft_sensor:
        state_min += [-cfg.force_range_n] * 3 + [-cfg.torque_range_nm] * 3
        state_max += [cfg.force_range_n] * 3 + [cfg.torque_range_nm] * 3
    state_min.append(0.0)
    state_max.append(1.0)
    dataset_stats = {
        **{f"observation.images.{key}": IMAGENET_STATS for key in robot.cameras},
        "observation.state": {"min": state_min, "max": state_max},
        "action": {"min": [-1.0] * n_pose, "max": [1.0] * n_pose},
    }
    return inverse_kinematics, home_joints, dataset_stats


def write_config(path: Path, inverse_kinematics: dict, home_joints: list[float], dataset_stats: dict) -> None:
    """Update the env values (and `policy.dataset_stats` for a train config) of a config file in place."""
    config = json.loads(path.read_text())
    processor = config["env"]["processor"]
    processor["inverse_kinematics"].update(inverse_kinematics)
    processor["reset"]["fixed_reset_joint_positions"] = home_joints
    # The stats are for the hil-serl state layout.
    observation = processor.setdefault("observation", {})
    observation["relative_ee_frame"] = True
    observation["add_joint_velocity_to_observation"] = False
    if "policy" in config:
        config["policy"]["dataset_stats"].update(json.loads(json.dumps(dataset_stats)))
    path.write_text(json.dumps(config, indent=2) + "\n")


def report_config_values(cfg: XArmTeleoperateConfig, robot: XArm, visited: VisitedRange):
    inverse_kinematics, home_joints, dataset_stats = find_config_values(cfg, robot, visited)
    print("\n\n===== env.processor =====")
    print(
        json.dumps(
            {
                "inverse_kinematics": inverse_kinematics,
                "reset": {"fixed_reset_joint_positions": home_joints},
            },
            indent=2,
        )
    )
    print("\n===== policy.dataset_stats =====")
    print(json.dumps(dataset_stats, indent=2))
    print("\nAdd a margin to the bounds for the UFACTORY Studio safety boundary (it is in millimeters).")

    if not cfg.write_configs:
        return
    paths = [cfg.env_config_path, cfg.train_config_path]
    answer = input(f"\nWrite these values into {', '.join(str(p) for p in paths)}? [y/N] ")
    if answer.strip().lower() != "y":
        print("Configs left unchanged.")
        return
    for path in paths:
        write_config(path, inverse_kinematics, home_joints, dataset_stats)
        print(f"Updated {path}")


@parser.wrap()
def teleoperate(cfg: XArmTeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="xarm_teleoperation")

    robot = XArm(cfg.robot)
    teleop = SpaceMouseTeleop(cfg.teleop)
    robot.connect()
    teleop.connect()

    if cfg.home_joints_deg is not None or cfg.robot.use_admittance:
        # Without home joints the arm stays in place; the reset still zeroes the F/T sensor and switches
        # admittance on, as at the start of every HIL-SERL episode.
        robot.reset_to_home(cfg.home_joints_deg)

    start_pose = np.array([robot.get_observation()[key] for key in EE_POSE_KEYS])
    start_position, start_orientation = start_pose[:3], Rotation.from_euler("xyz", start_pose[3:])
    start_rpy = bounded_rpy(start_orientation)
    bounds = cfg.end_effector_bounds or {
        "min": [*(start_position - cfg.box_m).tolist(), *(start_rpy - cfg.box_rad).tolist()],
        "max": [*(start_position + cfg.box_m).tolist(), *(start_rpy + cfg.box_rad).tolist()],
    }
    logging.info(f"End-effector bounds (m, rad): {bounds}")

    spacemouse_to_target = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            MapDeltaActionToRobotActionStep(),
            CartesianDeltaToTargetStep(
                end_effector_step_sizes=cfg.end_effector_step_sizes,
                end_effector_bounds=bounds,
                delta_reference=cfg.delta_reference,
                use_rotation=cfg.teleop.use_rotation,
                end_effector_rot_step=cfg.end_effector_rot_step,
                reference_clip=cfg.reference_clip,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # The start pose is the home of every episode.
    visited = VisitedRange(home_joints=robot.get_joint_positions())
    visited.update(start_position, start_orientation)
    timer = CycleTimer(cfg.fps, records_data=False)
    try:
        while True:
            timer.tick()
            with timer.section("observe"):
                obs = robot.get_observation()

            if robot.has_fault:
                print(f"\n{robot.fault_description}")
                input("Fix the cause, then press Enter to clear the error and move home (Ctrl+C to quit).")
                robot.reset_to_home(cfg.home_joints_deg)
                spacemouse_to_target.reset()
                timer.restart()
                continue

            with timer.section("teleop"):
                action = spacemouse_to_target((teleop.get_action(), obs))
            with timer.section("send"):
                robot.send_action(action)

            tcp = np.array([obs[key] for key in EE_POSE_KEYS])
            visited.update(tcp[:3], Rotation.from_euler("xyz", tcp[3:]))
            if cfg.display_data:
                log_rerun_data(observation=obs, action=action)

            force = ""
            if cfg.robot.use_ft_sensor:
                force = (
                    f"  force (N) {np.array2string(np.array([obs[f'ft.f{a}'] for a in 'xyz']), precision=1)}"
                )
            print(
                f"TCP (m, rad) {np.array2string(tcp, precision=3)}  gripper {obs['gripper.pos']:.2f}{force}  "
                f"visited min {np.array2string(visited.pose_min, precision=3)} max "
                f"{np.array2string(visited.pose_max, precision=3)}"
            )
            move_cursor_up(1)
            timer.wait()
    except KeyboardInterrupt:
        pass
    finally:
        timer.log_run_summary()
        if robot.is_connected:
            report_config_values(cfg, robot, visited)
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()
        if cfg.display_data:
            shutdown_rerun()


def main():
    teleoperate()


if __name__ == "__main__":
    main()
