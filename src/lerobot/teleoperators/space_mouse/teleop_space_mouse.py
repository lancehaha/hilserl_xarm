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

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.lerobot_types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _pynput_available, _pyspacemouse_available, require_package
from lerobot.utils.keyboard_input import pynput_can_capture

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_space_mouse import SpaceMouseTeleopConfig

if TYPE_CHECKING or _pyspacemouse_available:
    import pyspacemouse

if TYPE_CHECKING:
    from pynput import keyboard

logger = logging.getLogger(__name__)

# Gripper actions are expected to be 0 (close), 1 (stay), 2 (open), see `GamepadTeleop`.
GRIPPER_CLOSE = 0
GRIPPER_STAY = 1
GRIPPER_OPEN = 2

# How long the reader thread sleeps between HID reads. The SpaceMouse streams at a few hundred Hz and
# `pyspacemouse` consumes a single packet per `read()`, so this must stay well below the packet period.
_READ_PERIOD_S = 0.001


class SpaceMouseTeleop(Teleoperator):
    """
    Teleop class using a 3Dconnexion SpaceMouse for end-effector motion and gripper control,
    and the keyboard for human-in-the-loop events.

    SpaceMouse:
        - Puck translation: delta_x, delta_y, delta_z (see `axis_mapping` in the config)
        - Left button (held): close gripper
        - Right button (held): open gripper

    Intervention is active while the SpaceMouse is in use (puck moved or button held, see
    `intervene_on_input`), or while latched on with Space.

    Keyboard:
        - Space: toggle the latched intervention
        - Enter: end episode with SUCCESS
        - Backspace: end episode with FAILURE
        - r: re-record episode
    """

    config_class = SpaceMouseTeleopConfig
    name = "spacemouse"

    def __init__(self, config: SpaceMouseTeleopConfig):
        require_package("pyspacemouse", extra="spacemouse")
        require_package("pynput", extra="spacemouse")
        super().__init__(config)
        self.config = config

        self._axes = [axis.removeprefix("-") for axis in config.axis_mapping]
        self._signs = np.array([-1.0 if axis.startswith("-") else 1.0 for axis in config.axis_mapping])
        self._rotation_matrix = self._make_rotation_matrix(config)

        self._device: pyspacemouse.SpaceMouseDevice | None = None
        self._reader_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._translation: np.ndarray = np.zeros(3, dtype=np.float32)
        self._rotation: np.ndarray = np.zeros(3, dtype=np.float32)  # rotation about the puck x, y, z axes
        self._buttons: list[int] = []

        self._listener: keyboard.Listener | None = None
        self._events_lock = threading.Lock()
        self._is_intervention = False
        self._episode_end_status: TeleopEvents | None = None
        # Whether the action last returned by `get_action` had input, so the intervention flag matches it.
        self._last_action_has_input = False

    @property
    def action_features(self) -> dict:
        names = ["delta_x", "delta_y", "delta_z"]
        if self.config.use_rotation:
            names += ["delta_rx", "delta_ry", "delta_rz"]
        if self.config.use_gripper:
            names.append("gripper")
        return {
            "dtype": "float32",
            "shape": (len(names),),
            "names": {name: i for i, name in enumerate(names)},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._reader_thread is not None and self._reader_thread.is_alive()

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        # HID_Z_UP: same translation axes as the default (legacy) convention, and roll / pitch / yaw about those
        # same x / y / z axes. The legacy convention swaps roll and pitch and flips yaw.
        self._device = pyspacemouse.open(
            nonblocking=True,
            device_index=self.config.device_index,
            axis_convention=pyspacemouse.AxisConvention.HID_Z_UP,
        )
        logger.info(f"{self} connected: {self._device.describe_connection()}")

        self._stop_event.clear()
        self._reader_thread = threading.Thread(target=self._read_loop, name=f"{self} reader", daemon=True)
        self._reader_thread.start()

        self._start_keyboard_listener()

    def _read_loop(self) -> None:
        assert self._device is not None
        while not self._stop_event.is_set():
            state = self._device.read()
            with self._state_lock:
                self._translation[:] = (state.x, state.y, state.z)
                self._rotation[:] = (state.roll, state.pitch, state.yaw)
                self._buttons = list(state.buttons)
            time.sleep(_READ_PERIOD_S)

    def _start_keyboard_listener(self) -> None:
        if not (_pynput_available and pynput_can_capture()):
            logger.warning(
                "Keyboard events are unavailable in this environment (pynput needs an X11 session, a "
                "Windows desktop, or macOS with Accessibility permission). SpaceMouse motion still works, "
                "but intervention and episode events cannot be triggered."
            )
            return

        from pynput import keyboard

        def on_press(key):
            with self._events_lock:
                if key == keyboard.Key.space:
                    self._is_intervention = not self._is_intervention
                    logger.info(f"Latched intervention {'ON' if self._is_intervention else 'OFF'}")
                elif key == keyboard.Key.enter:
                    self._episode_end_status = TeleopEvents.SUCCESS
                elif key == keyboard.Key.backspace:
                    self._episode_end_status = TeleopEvents.FAILURE
                elif getattr(key, "char", None) == "r":
                    self._episode_end_status = TeleopEvents.RERECORD_EPISODE

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.start()

        print("SpaceMouse teleop controls:")
        print("  Puck: move end effector | Left/Right button: close/open gripper")
        if self.config.intervene_on_input:
            print("  Moving the puck or holding a button intervenes | Space: latch intervention on/off")
        else:
            print("  Space: toggle intervention")
        print("  Enter: end episode with SUCCESS | Backspace: end episode with FAILURE | r: re-record")

    def _get_deltas(self) -> np.ndarray:
        with self._state_lock:
            translation = dict(zip(("x", "y", "z"), self._translation.tolist(), strict=True))
        deltas = self._signs * np.array([translation[axis] for axis in self._axes])
        deltas[np.abs(deltas) < self.config.deadzone] = 0.0
        return np.clip(deltas * self.config.scale, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _make_rotation_matrix(config: SpaceMouseTeleopConfig) -> np.ndarray:
        """Matrix taking the puck rotation (about its x, y, z axes) to (delta_rx, delta_ry, delta_rz)."""
        puck_axes = {"x": 0, "y": 1, "z": 2}
        rotation_axes = {"roll": 0, "pitch": 1, "yaw": 2}
        matrix = np.zeros((3, 3))
        if config.rotation_mapping is None:
            for row, axis in enumerate(config.axis_mapping):
                matrix[row, puck_axes[axis.removeprefix("-")]] = -1.0 if axis.startswith("-") else 1.0
            # A rotation is an axial vector: a mirrored mapping (e.g. one inverted axis) also flips its sense.
            return matrix * np.linalg.det(matrix)
        for row, axis in enumerate(config.rotation_mapping):
            if axis != "0":
                matrix[row, rotation_axes[axis.removeprefix("-")]] = -1.0 if axis.startswith("-") else 1.0
        return matrix

    def _get_rotation_deltas(self) -> np.ndarray:
        with self._state_lock:
            rotation: np.ndarray = self._rotation.astype(float)
        rotation[np.abs(rotation) < self.config.rotation_deadzone] = 0.0
        deltas = self._rotation_matrix @ rotation
        return np.clip(deltas * self.config.rotation_scale, -1.0, 1.0).astype(np.float32)

    def _get_buttons(self) -> list[int]:
        with self._state_lock:
            return list(self._buttons)

    @staticmethod
    def _gripper_action_from_buttons(buttons: list[int]) -> int:
        close = len(buttons) > 0 and bool(buttons[0])
        open_ = len(buttons) > 1 and bool(buttons[1])
        if close == open_:
            return GRIPPER_STAY
        return GRIPPER_CLOSE if close else GRIPPER_OPEN

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        deltas = self._get_deltas()
        buttons = self._get_buttons()
        action_dict: RobotAction = {
            "delta_x": deltas[0],
            "delta_y": deltas[1],
            "delta_z": deltas[2],
        }
        has_input = bool(np.any(deltas != 0.0))
        if self.config.use_rotation:
            rotation_deltas = self._get_rotation_deltas()
            action_dict.update(
                {
                    "delta_rx": rotation_deltas[0],
                    "delta_ry": rotation_deltas[1],
                    "delta_rz": rotation_deltas[2],
                }
            )
            has_input = has_input or bool(np.any(rotation_deltas != 0.0))
        if self.config.use_gripper:
            action_dict["gripper"] = self._gripper_action_from_buttons(buttons)
        self._last_action_has_input = has_input or (self.config.use_gripper and any(buttons))
        return action_dict

    def get_teleop_events(self) -> dict[TeleopEvents, bool]:
        """
        Get extra control events from the keyboard such as intervention status,
        episode termination, success indicators, etc.

        With `intervene_on_input`, the human intervenes on the steps where the action last returned by
        `get_action` has input, so call `get_action` first, as the HIL processor pipeline does.

        Returns:
            Dictionary containing:
                - is_intervention: bool - Whether human is currently intervening
                - terminate_episode: bool - Whether to terminate the current episode
                - success: bool - Whether the episode was successful
                - rerecord_episode: bool - Whether to rerecord the episode
        """
        with self._events_lock:
            is_intervention = self._is_intervention or (
                self.config.intervene_on_input and self._last_action_has_input
            )
            episode_end_status = self._episode_end_status
            self._episode_end_status = None
            # Hand control back to the policy at the start of the next episode.
            if episode_end_status is not None:
                self._is_intervention = False

        terminate_episode = episode_end_status in [
            TeleopEvents.RERECORD_EPISODE,
            TeleopEvents.FAILURE,
        ]
        success = episode_end_status == TeleopEvents.SUCCESS
        rerecord_episode = episode_end_status == TeleopEvents.RERECORD_EPISODE

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }

    @check_if_not_connected
    def disconnect(self) -> None:
        self._stop_event.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None
        if self._device is not None:
            self._device.close()
            self._device = None
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    @property
    def is_calibrated(self) -> bool:
        # SpaceMouse doesn't require calibration
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        # SpaceMouse doesn't support feedback
        pass
