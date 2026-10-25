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

"""Episode-aware replay for successful, efficient xArm manipulation attempts.

The index contains only slot numbers and small episode metadata. Images remain in
the ordinary replay buffer, so emphasizing successful episodes does not copy them.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from statistics import median

from lerobot.lerobot_types import BatchType
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.transition import Transition

from ..buffer import ReplayBuffer, concatenate_batch_transitions
from .data_mixer import DataMixer


@dataclass
class EpisodeRecord:
    episode_id: int
    successful: bool
    assisted: bool
    path_length_m: float
    steps: int
    slots: list[int] = field(default_factory=list)
    event_slots: list[int] = field(default_factory=list)
    complete: bool = True


class EpisodeReplayIndex:
    """Track complete episodes and close/open command events in a circular buffer."""

    def __init__(self, buffer: ReplayBuffer, success_reward: float = 1.0) -> None:
        self.buffer = buffer
        self.success_reward = success_reward
        self.episodes: dict[int, EpisodeRecord] = {}
        self.slot_owner: list[int | None] = [None] * buffer.capacity
        self.next_episode_id = 0

    def rebuild_from_buffer(self) -> None:
        """Restore episode labels for transitions loaded when training resumes.

        Saved datasets do not contain the original episode metadata or final
        next observation. Path length is therefore estimated from consecutive
        observations; incomplete episodes are never used for full sweeps.
        """
        if not self.buffer.initialized or not self.buffer.size:
            return
        first = self.buffer.position if self.buffer.size == self.buffer.capacity else 0
        ordered_slots = [(first + offset) % self.buffer.capacity for offset in range(self.buffer.size)]
        current: list[int] = []
        for slot in ordered_slots:
            current.append(slot)
            if bool(self.buffer.dones[slot]) or bool(self.buffer.truncateds[slot]):
                self._index_loaded_episode(current, complete=True)
                current = []
        if current:
            self._index_loaded_episode(current, complete=False)

    def _index_loaded_episode(self, slots: list[int], *, complete: bool) -> None:
        episode_id = self.next_episode_id
        self.next_episode_id += 1
        successful = complete and any(
            bool(self.buffer.dones[slot]) and float(self.buffer.rewards[slot]) >= self.success_reward
            for slot in slots
        )
        states = self.buffer.states.get("observation.state")
        first_state = states[slots[0]] if states is not None else None
        gripper_closed = bool(float(first_state.reshape(-1)[3]) >= 0.5) if first_state is not None else False
        path_length_m = 0.0
        event_slots: list[int] = []
        for index, slot in enumerate(slots):
            action = self.buffer.actions[slot].reshape(-1)
            if action.numel() >= 4:
                command = round(float(action[-1]))
                if command == 0 and not gripper_closed:
                    event_slots.append(slot)
                    gripper_closed = True
                elif command == 2 and gripper_closed:
                    event_slots.append(slot)
                    gripper_closed = False
            if states is not None and index + 1 < len(slots):
                delta = states[slots[index + 1]].reshape(-1)[:3] - states[slot].reshape(-1)[:3]
                path_length_m += float(delta.norm())
            self.slot_owner[slot] = episode_id
        self.episodes[episode_id] = EpisodeRecord(
            episode_id=episode_id,
            successful=successful,
            assisted=False,
            path_length_m=path_length_m,
            steps=len(slots),
            slots=slots.copy(),
            event_slots=event_slots,
            complete=complete,
        )

    def _eligible(self, slot: int, episode_id: int) -> bool:
        if self.slot_owner[slot] != episode_id:
            return False
        if (
            self.buffer.optimize_memory
            and slot == (self.buffer.position - 1) % self.buffer.capacity
            and not bool(self.buffer.dones[slot])
        ):
            return False
        return True

    def _evict_slot(self, slot: int, current_episode_id: int) -> None:
        old_id = self.slot_owner[slot]
        if old_id is None:
            return
        record = self.episodes[old_id]
        record.slots.remove(slot)
        if slot in record.event_slots:
            record.event_slots.remove(slot)
        record.complete = False
        if not record.slots and old_id != current_episode_id:
            del self.episodes[old_id]
        self.slot_owner[slot] = None

    def add_episode(self, transitions: list[Transition]) -> None:
        """Add one actor episode and index its executed gripper command changes."""
        if not transitions:
            return

        episode_id = self.next_episode_id
        self.next_episode_id += 1
        successful = any(
            bool(t["done"]) and float(t["reward"]) >= self.success_reward for t in transitions
        )
        assisted = any(
            bool((t.get("complementary_info") or {}).get(TeleopEvents.IS_INTERVENTION.value, False))
            for t in transitions
        )
        record = EpisodeRecord(
            episode_id=episode_id,
            successful=successful,
            assisted=assisted,
            path_length_m=0.0,
            steps=len(transitions),
        )
        self.episodes[episode_id] = record

        first_state = transitions[0]["state"].get("observation.state")
        gripper_closed = bool(float(first_state.reshape(-1)[3]) >= 0.5) if first_state is not None else False

        for transition in transitions:
            slot = self.buffer.position
            self._evict_slot(slot, episode_id)
            action = transition["action"].reshape(-1)
            event = False
            if action.numel() >= 4:
                command = round(float(action[-1]))
                if command == 0 and not gripper_closed:
                    event = True
                    gripper_closed = True
                elif command == 2 and gripper_closed:
                    event = True
                    gripper_closed = False

            state = transition["state"].get("observation.state")
            next_state = transition["next_state"].get("observation.state")
            if state is not None and next_state is not None:
                delta = next_state.reshape(-1)[:3] - state.reshape(-1)[:3]
                record.path_length_m += float(delta.norm())

            self.buffer.add(**transition)
            self.slot_owner[slot] = episode_id
            record.slots.append(slot)
            if event:
                record.event_slots.append(slot)

    def successful_episodes(self, *, complete_only: bool = False) -> list[EpisodeRecord]:
        return [
            record
            for record in self.episodes.values()
            if record.successful
            and (record.complete or not complete_only)
            and any(self._eligible(slot, record.episode_id) for slot in record.slots)
        ]

    def choose_short_success(
        self, *, complete_only: bool = False, short_weight: float = 2.0
    ) -> EpisodeRecord | None:
        records = self.successful_episodes(complete_only=complete_only)
        if not records:
            return None
        midpoint = median(record.path_length_m for record in records)
        weights = [short_weight if record.path_length_m <= midpoint else 1.0 for record in records]
        return random.choices(records, weights=weights, k=1)[0]

    def choose_step(self, record: EpisodeRecord, event_fraction: float) -> int | None:
        slots = [slot for slot in record.slots if self._eligible(slot, record.episode_id)]
        if not slots:
            return None
        events = [slot for slot in record.event_slots if self._eligible(slot, record.episode_id)]
        if events and random.random() < event_fraction:
            return random.choice(events)
        return random.choice(slots)


class EpisodeAwareMixer(DataMixer):
    """Mix uniform replay with efficient successes, command events, and episode sweeps."""

    def __init__(
        self,
        online_buffer: ReplayBuffer,
        offline_buffer: ReplayBuffer | None,
        index: EpisodeReplayIndex,
        online_ratio: float = 0.5,
        success_fraction: float = 0.25,
        event_fraction: float = 0.25,
        sweep_interval: int = 10,
        trajectory_fraction: float = 0.10,
        trajectory_short_weight: float = 2.0,
    ) -> None:
        self.online_buffer = online_buffer
        self.offline_buffer = offline_buffer
        self.index = index
        self.online_ratio = online_ratio
        self.success_fraction = success_fraction
        self.event_fraction = event_fraction
        self.sweep_interval = sweep_interval
        self.trajectory_fraction = trajectory_fraction
        self.trajectory_short_weight = trajectory_short_weight
        self.batch_count = 0
        self.pending_sweep: deque[tuple[int, int]] = deque()

    def _trajectory_slots(self, count: int) -> list[int]:
        if count <= 0:
            return []
        if (
            self.sweep_interval > 0
            and not self.pending_sweep
            and self.batch_count % self.sweep_interval == 0
        ):
            record = self.index.choose_short_success(
                complete_only=True, short_weight=self.trajectory_short_weight
            )
            if record is not None:
                self.pending_sweep.extend((record.episode_id, slot) for slot in record.slots)

        slots: list[int] = []
        while self.pending_sweep and len(slots) < count:
            episode_id, slot = self.pending_sweep.popleft()
            if self.index._eligible(slot, episode_id):
                slots.append(slot)

        return slots

    def _priority_slots(self, count: int) -> list[int]:
        if count <= 0:
            return []
        slots: list[int] = []
        records = self.index.successful_episodes()
        if records:
            midpoint = median(record.path_length_m for record in records)
            weights = [2.0 if record.path_length_m <= midpoint else 1.0 for record in records]
        while len(slots) < count:
            if not records:
                break
            record = random.choices(records, weights=weights, k=1)[0]
            slot = self.index.choose_step(record, self.event_fraction)
            if slot is not None:
                slots.append(slot)
        return slots

    def sample(self, batch_size: int) -> BatchType:
        self.batch_count += 1
        n_online = batch_size if self.offline_buffer is None else max(1, int(batch_size * self.online_ratio))
        n_offline = batch_size - n_online
        n_priority = int(n_online * self.success_fraction)
        n_trajectory = int(n_online * self.trajectory_fraction)
        priority_slots = self._priority_slots(n_priority)
        trajectory_slots = self._trajectory_slots(n_trajectory)

        replay_slots = priority_slots + trajectory_slots
        n_uniform = n_online - len(replay_slots)
        online_batch = self.online_buffer.sample(n_uniform) if n_uniform else None
        if replay_slots:
            priority_batch = self.online_buffer.sample(len(replay_slots), indices=replay_slots)
            online_batch = (
                concatenate_batch_transitions(online_batch, priority_batch)
                if online_batch is not None
                else priority_batch
            )
        if n_offline and self.offline_buffer is not None:
            offline_batch = self.offline_buffer.sample(n_offline)
            return concatenate_batch_transitions(online_batch, offline_batch)
        return online_batch

    def get_iterator(self, batch_size: int, async_prefetch: bool = True, queue_size: int = 2):
        """Sample synchronously so an episode sweep has a stable update order."""
        while True:
            yield self.sample(batch_size)
