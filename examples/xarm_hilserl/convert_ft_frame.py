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
Convert the force / torque of an xArm dataset from the base frame to the tool frame.

Datasets recorded with `ft_sensor_frame: "tool"` before it stopped rotating `ft_ext_force` store the wrench in
the base frame. This writes a copy with the tool-frame wrench f_tool = R^-1 f_base, where R = R_reset R_rel is
the absolute TCP orientation: R_rel comes from the relative roll / pitch / yaw in the state
(`relative_ee_frame`), R_reset is the TCP orientation at the fixed reset joints (no random reset rotation).
The source dataset is not modified.

Example:

```shell
# R_reset from the arm's forward kinematics at the reset joints (the arm does not move)
uv run python examples/xarm_hilserl/convert_ft_frame.py \
    --repo-id local/xarm7_hilserl_demos_6dof_ft --ip 192.168.1.235

# or R_reset given directly, as roll pitch yaw in radians from `get_position(is_radian=True)` at reset
uv run python examples/xarm_hilserl/convert_ft_frame.py \
    --repo-id local/xarm7_hilserl_demos_6dof_ft --reset-rpy 3.14 0.0 1.57
```
"""

import argparse
import shutil

import numpy as np
import pandas as pd

from lerobot.datasets.dataset_tools import recompute_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME, OBS_STATE
from lerobot.utils.rotation import Rotation

# `observation.state` with `relative_ee_frame`: pose (6), twist (6), force (3), torque (3), gripper (1).
RPY_SLICE = slice(3, 6)
FORCE_SLICE = slice(12, 15)
TORQUE_SLICE = slice(15, 18)
STATE_DIM = 19

DEFAULT_RESET_JOINTS_DEG = [102.12, -8.46, 12.85, 38.66, 2.26, 44.96, 46.88]


def reset_rpy_from_arm(ip: str, joints_deg: list[float]) -> np.ndarray:
    """TCP roll / pitch / yaw (rad) at `joints_deg`, from the controller's forward kinematics."""
    from xarm.wrapper import XArmAPI

    api = XArmAPI(ip)
    try:
        code, pose = api.get_forward_kinematics(joints_deg, input_is_radian=False, return_is_radian=True)
        if code != 0:
            raise RuntimeError(f"get_forward_kinematics failed with code {code}")
    finally:
        api.disconnect()
    return np.asarray(pose[3:6], dtype=float)


def convert_parquet(path, reset_orientation: Rotation) -> int:
    """Rotate the wrench of every frame into the tool frame. Returns the number of frames."""
    df = pd.read_parquet(path)
    states = np.stack(df[OBS_STATE].to_numpy()).astype(np.float64)
    if states.shape[1] != STATE_DIM:
        raise ValueError(f"{path}: expected a {STATE_DIM}-D state, got {states.shape[1]}")

    for state in states:
        to_tool = (reset_orientation * Rotation.from_euler("xyz", state[RPY_SLICE])).inv()
        state[FORCE_SLICE] = to_tool.apply(state[FORCE_SLICE])
        state[TORQUE_SLICE] = to_tool.apply(state[TORQUE_SLICE])

    df[OBS_STATE] = list(states.astype(np.float32))
    df.to_parquet(path, index=False)
    return len(df)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo-id", default="local/xarm7_hilserl_demos_6dof_ft")
    parser.add_argument("--new-repo-id", default=None, help="Defaults to `<repo-id>_toolft`.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--reset-rpy", type=float, nargs=3, help="TCP roll pitch yaw (rad) at reset.")
    source.add_argument("--ip", help="xArm IP, to compute the reset TCP orientation by forward kinematics.")
    parser.add_argument("--reset-joints-deg", type=float, nargs=7, default=DEFAULT_RESET_JOINTS_DEG)
    args = parser.parse_args()

    if args.reset_rpy:
        reset_rpy = np.asarray(args.reset_rpy, dtype=float)
    else:
        reset_rpy = reset_rpy_from_arm(args.ip, args.reset_joints_deg)
    print(f"Reset TCP roll / pitch / yaw (rad): {np.round(reset_rpy, 4).tolist()}")
    reset_orientation = Rotation.from_euler("xyz", reset_rpy)

    new_repo_id = args.new_repo_id or f"{args.repo_id}_toolft"
    src_root = HF_LEROBOT_HOME / args.repo_id
    dst_root = HF_LEROBOT_HOME / new_repo_id
    if dst_root.exists():
        raise FileExistsError(f"{dst_root} already exists, remove it or pass another --new-repo-id")
    shutil.copytree(src_root, dst_root)

    paths = sorted((dst_root / "data").glob("*/*.parquet"))
    n_frames = sum(convert_parquet(path, reset_orientation) for path in paths)

    dataset = recompute_stats(LeRobotDataset(new_repo_id, root=dst_root))
    states = np.stack(dataset.hf_dataset[OBS_STATE])
    print(f"\nSaved {new_repo_id}: {dataset.num_episodes} episodes, {n_frames} frames")
    for name, sl in (("force (N)", FORCE_SLICE), ("torque (Nm)", TORQUE_SLICE)):
        low, high = states[:, sl].min(0).astype(float), states[:, sl].max(0).astype(float)
        print(f"  {name}: min {np.round(low, 2).tolist()} max {np.round(high, 2).tolist()}")


if __name__ == "__main__":
    main()
