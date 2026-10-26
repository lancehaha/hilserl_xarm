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
Hardware checks for the xArm Six-axis F/T sensor before running HIL-SERL with admittance control.

Stand by the emergency stop for every subcommand except `status`.

    # Sensor, controller error and payload status (no motion)
    uv run python examples/xarm_hilserl/probe_admittance.py status

    # Identify the load (gripper) on the sensor and write it with the TCP payload. The wrist moves.
    uv run python examples/xarm_hilserl/probe_admittance.py identify

    # Print the compensated force for `--duration` s, stiff arm. Push the gripper along base +X, then +Z, to see
    # in which frame `ft_ext_force` is expressed.
    uv run python examples/xarm_hilserl/probe_admittance.py frame --duration 15

    # Enable admittance, stream +-`--amplitude` m targets along base X in mode 7 (or `--mode 1`), report the
    # tracking error, then hold the start pose for `--hold` s: push the gripper by hand, it should yield and return.
    uv run python examples/xarm_hilserl/probe_admittance.py track --mode 7 --amplitude 0.01 --hold 10
"""

import argparse
import time

import numpy as np
from xarm.core.config.x_code import ControllerError
from xarm.wrapper import XArmAPI

ONLINE_CARTESIAN_MODE = 7
SERVO_CARTESIAN_MODE = 1


def describe_error(code: int) -> str:
    if code == 0:
        return "none"
    error = ControllerError(code)
    return f"{code} ({error.title['en']})"


def status(arm: XArmAPI) -> None:
    code, config = arm.get_ft_sensor_config()
    print(
        f"controller error: {describe_error(arm.error_code)}, warn {arm.warn_code}, state {arm.state}, mode {arm.mode}"
    )
    print(f"collision sensitivity {arm.collision_sensitivity}, TCP payload {arm.tcp_load}")
    print(f"F/T sensor error: {arm.get_ft_sensor_error()}")
    if code == 0:
        print(f"F/T mode {config[0]}, started {config[1]}, load mass {config[5]} kg, centroid {config[7]}")
        print(f"admittance frame {config[9]}, axes {config[10]}, M {config[11]}, K {config[12]}")
    print(f"ext force {np.round(arm.ft_ext_force, 2).tolist()}")
    print(f"raw force {np.round(arm.ft_raw_force, 2).tolist()}")


def enable(arm: XArmAPI, mode: int = 0) -> None:
    arm.clean_error()
    arm.clean_warn()
    time.sleep(0.3)
    arm.motion_enable(enable=True)
    arm.set_mode(mode)
    arm.set_state(0)
    time.sleep(0.5)
    if arm.error_code != 0:
        raise RuntimeError(f"controller error {describe_error(arm.error_code)}")


def start_sensor(arm: XArmAPI) -> None:
    arm.set_ft_sensor_enable(1)
    time.sleep(0.5)
    arm.set_ft_sensor_zero()
    time.sleep(0.2)


def identify(arm: XArmAPI) -> None:
    enable(arm)
    start_sensor(arm)
    print("Identifying the load, the wrist moves...")
    code, result = arm.iden_ft_sensor_load_offset()
    print(f"identification code {code}: {result}")
    if code != 0:
        return
    print("set load offset:", arm.set_ft_sensor_load_offset(result, association_setting_tcp_load=True))
    time.sleep(0.5)
    arm.set_ft_sensor_zero()
    time.sleep(0.5)
    print(f"TCP payload now {arm.tcp_load}")
    for _ in range(5):
        print(f"ext force at rest {np.round(arm.ft_ext_force, 2).tolist()}")
        time.sleep(0.2)


def frame(arm: XArmAPI, duration: float) -> None:
    enable(arm)
    start_sensor(arm)
    print(f"Push the gripper along base +X, then base +Z, within {duration:.0f} s.")
    code, pose = arm.get_position(is_radian=False)
    print(f"TCP pose [mm, deg] {np.round(pose, 1).tolist()}")
    end = time.perf_counter() + duration
    while time.perf_counter() < end:
        print(f"ext force {np.round(arm.ft_ext_force, 1).tolist()}")
        time.sleep(0.25)


def track(arm: XArmAPI, mode: int, amplitude: float, stiffness: list[float], hold: float, fps: float) -> None:
    enable(arm)
    code, start = arm.get_position(is_radian=False)
    start = np.array(start, dtype=float)

    mass = [0.06, 0.06, 0.06, 0.0006, 0.0006, 0.0006]
    print("admittance params:", arm.set_ft_sensor_admittance_parameters(M=mass, K=stiffness, B=[0] * 6))
    print("admittance axes:", arm.set_ft_sensor_admittance_parameters(coord=0, c_axis=[1] * 6))
    start_sensor(arm)
    print("ft mode 1:", arm.set_ft_sensor_mode(1))
    arm.set_mode(mode)
    arm.set_state(0)
    time.sleep(0.5)
    print(
        f"mode {arm.mode}, state {arm.state}, F/T mode {arm.get_ft_sensor_mode()}, "
        f"error {describe_error(arm.error_code)}"
    )

    if mode == SERVO_CARTESIAN_MODE:
        # Servo mode executes each target at once: stream at a high rate so consecutive targets stay close.
        fps = max(fps, 100.0)
    period = 1.0 / fps
    errors = []
    t0 = time.perf_counter()
    duration = 4.0
    while time.perf_counter() - t0 < duration and arm.error_code == 0:
        t = time.perf_counter() - t0
        target = start.copy()
        target[0] += 1000 * amplitude * np.sin(2 * np.pi * t / duration)
        if mode == SERVO_CARTESIAN_MODE:
            code = arm.set_servo_cartesian(target.tolist(), is_radian=False)
        else:
            code = arm.set_position(*target, speed=100, mvacc=1000, is_radian=False, wait=False)
        _, pose = arm.get_position(is_radian=False)
        errors.append(abs(pose[0] - target[0]))
        if code != 0:
            print(f"command returned code {code}")
        time.sleep(max(period - (time.perf_counter() - t0 - t), 0.0))
    if errors:
        print(
            f"tracking |x error| mean {np.mean(errors):.1f} mm, max {np.max(errors):.1f} mm over {len(errors)} steps"
        )

    print(f"Holding the start pose for {hold:.0f} s: push the gripper by hand.")
    end = time.perf_counter() + hold
    while time.perf_counter() < end and arm.error_code == 0:
        if mode == SERVO_CARTESIAN_MODE:
            arm.set_servo_cartesian(start.tolist(), is_radian=False)
        else:
            arm.set_position(*start, speed=100, mvacc=1000, is_radian=False, wait=False)
        _, pose = arm.get_position(is_radian=False)
        offset = np.array(pose[:3]) - start[:3]
        print(
            f"offset [mm] {np.round(offset, 1).tolist()}  force {np.round(arm.ft_ext_force[:3], 1).tolist()}"
        )
        time.sleep(0.25)

    print(f"error after test: {describe_error(arm.error_code)}")
    arm.set_ft_sensor_mode(0)
    arm.set_mode(0)
    arm.set_state(0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["status", "identify", "frame", "track"])
    parser.add_argument("--ip", default="192.168.1.235")
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--mode", type=int, default=ONLINE_CARTESIAN_MODE, choices=[1, 7])
    parser.add_argument("--amplitude", type=float, default=0.01, help="meters")
    parser.add_argument("--stiffness", type=float, nargs=6, default=[2000, 2000, 2000, 20, 20, 20])
    parser.add_argument("--hold", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    arm = XArmAPI(args.ip)
    try:
        if args.command == "identify":
            identify(arm)
        elif args.command == "frame":
            frame(arm, args.duration)
        elif args.command == "track":
            track(arm, args.mode, args.amplitude, args.stiffness, args.hold, args.fps)
        status(arm)
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
