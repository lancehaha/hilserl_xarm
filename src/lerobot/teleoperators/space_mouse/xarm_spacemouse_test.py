"""Check SpaceMouse -> xArm axis orientation by jogging the TCP in its base frame.

Without --move it only prints the target pose (dry run). With --move the TCP follows the puck and
holding the left / right button closes / opens the xArm Gripper (0 = closed, 850 = open).

Safety limits (all enforced in this script, the xArm controller itself has no boundary set):
  - position: within +/- --box_mm of the starting_pose pose on each axis, a step that would leave the box is
    dropped (EDGE is shown) instead of sliding along it
  - rotation (--rot only): at most --max_rot_deg away from the starting_pose orientation
  - speed --step_mm * --fps mm/s
  - on an xArm controller error (e.g. 22 self-collision) motion pauses until Enter clears it

    .venv/bin/python xarm_spacemouse_test.py [--mapping y -x z] [--rot [--rot_mapping roll pitch yaw]] [--move]
"""

import argparse
import time

import numpy as np
from xarm.core.config.x_code import ControllerError
from xarm.wrapper import XArmAPI

from lerobot.teleoperators.space_mouse import SpaceMouseTeleop, SpaceMouseTeleopConfig
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.rotation import Rotation
from lerobot.utils.utils import init_logging

ROT_AXES = ("roll", "pitch", "yaw")

parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="192.168.1.235")
parser.add_argument("--mapping", nargs=3, default=["x", "y", "z"], help="puck axis driving robot X, Y, Z")
parser.add_argument("--step_mm", type=float, default=5.0, help="TCP step per tick at full puck deflection")
parser.add_argument(
    "--box_mm", type=float, default=300.0, help="max distance from the starting_pose pose per axis"
)
parser.add_argument("--rot", action="store_true", help="also follow puck tilt / twist")
parser.add_argument(
    "--rot_mapping",
    nargs=3,
    default=list(ROT_AXES),
    help="puck rotation (roll/pitch/yaw, '-' to invert, '0' to disable) driving rotation about robot X, Y, Z",
)
parser.add_argument(
    "--rot_deadzone", type=float, default=0.15, help="rotation readings below this are ignored"
)
parser.add_argument("--rot_step_deg", type=float, default=3.0, help="rotation per tick at full deflection")
parser.add_argument(
    "--max_rot_deg", type=float, default=180.0, help="max rotation away from the starting_pose orientation"
)
parser.add_argument("--fps", type=float, default=10.0)
parser.add_argument(
    "--gripper_step", type=float, default=100.0, help="gripper units per tick while a button is held"
)
parser.add_argument("--move", action="store_true", help="actually move the robot")
args = parser.parse_args()
init_logging()

for axis in args.rot_mapping:
    if axis != "0" and axis.removeprefix("-") not in ROT_AXES:
        parser.error(
            f"invalid --rot_mapping entry '{axis}', expected one of {ROT_AXES}, optionally '-', or '0'"
        )


# xArm roll/pitch/yaw are fixed-axis X, Y, Z rotations: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
def rpy_to_mat(rpy_deg):
    r, p, y = np.radians(rpy_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def mat_to_rpy(m):
    pitch = -np.arcsin(np.clip(m[2, 0], -1.0, 1.0))
    roll = np.arctan2(m[2, 1], m[2, 2])
    yaw = np.arctan2(m[1, 0], m[0, 0])
    return np.degrees([roll, pitch, yaw])


def describe_error(code):
    err = ControllerError(code)
    return f"{err.title['en']}. {err.description['en']}"


_last_api_codes = {}


def check_api(name, code):
    """Report an xArm call that starts failing, once per new code, without stopping the loop."""
    if code != 0 and _last_api_codes.get(name) != code:
        print(f"\n\033[93mxArm {name} returned code {code}\033[0m")
    _last_api_codes[name] = code


def enable_motion():
    arm.clean_error()
    arm.motion_enable(True)
    arm.set_mode(7)  # cartesian online trajectory planning
    arm.set_state(0)
    time.sleep(0.5)


def read_pose():
    code, pose = arm.get_position(is_radian=False)  # [x, y, z, roll, pitch, yaw] in mm / deg
    if code != 0:
        raise RuntimeError(f"get_position failed with code {code}")
    return np.array(pose, dtype=float)


arm = XArmAPI(args.ip)
if args.move:
    enable_motion()
    arm.set_gripper_enable(True)
    arm.set_gripper_mode(0)
    arm.set_gripper_speed(3000)

GRIPPER_OPEN_POS = 850.0
code, grip_pos = arm.get_gripper_position()
grip_pos = float(grip_pos) if code == 0 and grip_pos is not None else GRIPPER_OPEN_POS

starting_pose = read_pose()
target_pos = starting_pose[:3].copy()
start_rot = rpy_to_mat(starting_pose[3:])
target_rot = start_rot.copy()


def rot_from_start(rot):
    return np.degrees(np.linalg.norm(Rotation.from_matrix(rot @ start_rot.T).as_rotvec()))


rot_axes = [None if a == "0" else ROT_AXES.index(a.removeprefix("-")) for a in args.rot_mapping]
rot_signs = np.array([-1.0 if a.startswith("-") else 1.0 for a in args.rot_mapping])

teleop = SpaceMouseTeleop(SpaceMouseTeleopConfig(axis_mapping=args.mapping, use_gripper=True))
teleop.connect()
print(
    f"starting_pose TCP xyz={starting_pose[:3].round(1)} mm rpy={starting_pose[3:].round(1)} deg  move={args.move}  (Ctrl+C to stop)"
)
if args.rot:
    print(f"rotation: mapping={args.rot_mapping} max {args.max_rot_deg} deg from starting_pose")
print("xArm base frame: +X out of the front of the base, +Y to its left, +Z up")

timer = CycleTimer(args.fps, records_data=False)
try:
    while True:
        if args.move and arm.error_code != 0:
            print(f"\n\033[91mxArm error {arm.error_code}: {describe_error(arm.error_code)}\033[0m")
            input(
                "Motion paused. Fix the cause, then press Enter to clear the error and resume (Ctrl+C to quit)"
            )
            enable_motion()
            if arm.error_code != 0:
                continue
            # resume from where the arm actually stopped, not from the target that triggered the error
            pose = read_pose()
            target_pos = pose[:3]
            target_rot = rpy_to_mat(pose[3:])
            print("Error cleared, resuming.")
            timer.restart()  # keep the paused time out of the cadence stats

        timer.tick()
        a = teleop.get_action()
        # limits step size
        delta = np.array([a["delta_x"], a["delta_y"], a["delta_z"]]) * args.step_mm
        # drop the whole step if it takes any axis further out of the box (no sliding along the edge)
        candidate_pos = target_pos + delta

        pos_low = starting_pose[:3] - args.box_mm
        pos_high = starting_pose[:3] + args.box_mm

        if np.any((candidate_pos < pos_low) | (candidate_pos > pos_high)):
            delta = np.zeros(3)
            print("Position exceeded bounding box.")
        else:
            target_pos = candidate_pos

        raw_rot = teleop._rotation.copy()
        rot_cmd = np.zeros(3)
        rot_limited = False
        if args.rot:
            raw = np.where(np.abs(raw_rot) < args.rot_deadzone, 0.0, raw_rot)
            rot_cmd = np.array([0.0 if ax is None else raw[ax] for ax in rot_axes]) * rot_signs
            rot_cmd = np.clip(rot_cmd, -1.0, 1.0)
            if np.any(rot_cmd):
                # rotate about the base axes, like rail-berkeley/hil-serl
                candidate = (
                    Rotation.from_rotvec(np.radians(rot_cmd * args.rot_step_deg)).as_matrix() @ target_rot
                )
                # past the limit (e.g. after resuming from an error) only steps back toward the starting_pose are allowed
                new_angle = rot_from_start(candidate)
                if new_angle <= args.max_rot_deg or new_angle < rot_from_start(target_rot):
                    target_rot = candidate
                else:
                    rot_cmd = np.zeros(3)
                    rot_limited = True

        if args.move and (np.any(delta) or np.any(rot_cmd)):
            code = arm.set_position(
                *target_pos,
                *mat_to_rpy(target_rot),
                speed=args.fps * args.step_mm,
                is_radian=False,
                wait=False,
            )
            check_api("set_position", code)

        # teleop gripper: 0 = close, 1 = stay, 2 = open (same convention as HIL-SERL)
        grip_dir = a["gripper"] - 1
        if grip_dir:
            grip_pos = float(np.clip(grip_pos + grip_dir * args.gripper_step, 0.0, GRIPPER_OPEN_POS))
            if args.move:
                check_api("set_gripper_position", arm.set_gripper_position(grip_pos, wait=False))

        off = target_pos - starting_pose[:3]
        rot_off = np.degrees(Rotation.from_matrix(target_rot @ start_rot.T).as_rotvec())
        limit = " ROT-LIMIT" if rot_limited else ""
        print(
            f"\r\033[K"
            f"T({a['delta_x']:+.2f},{a['delta_y']:+.2f},{a['delta_z']:+.2f}) "
            f"raw r/p/y({raw_rot[0]:+.2f},{raw_rot[1]:+.2f},{raw_rot[2]:+.2f}) "
            f"btn={teleop._buttons} grip={grip_pos:3.0f} | "
            f"off x{off[0]:+5.0f} y{off[1]:+5.0f} z{off[2]:+5.0f}mm "
            f"rot X{rot_off[0]:+5.1f} Y{rot_off[1]:+5.1f} Z{rot_off[2]:+5.1f}deg{limit}",
            end="",
            flush=True,
        )
        timer.wait()
except KeyboardInterrupt:
    pass
finally:
    print()
    timer.log_run_summary()
    teleop.disconnect()
    if args.move:
        arm.set_mode(0)
        arm.set_state(0)
    arm.disconnect()
