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

"""Top-level pipeline config for distributed RL training (actor / learner)."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig

from .algorithms.configs import RLAlgorithmConfig
from .algorithms.factory import make_algorithm_config
from .algorithms.sac import SACAlgorithmConfig  # noqa: F401


@dataclass(kw_only=True)
class TrainRLServerPipelineConfig(TrainPipelineConfig):
    # NOTE: In RL, we don't need an offline dataset
    # TODO: Make `TrainPipelineConfig.dataset` optional
    dataset: DatasetConfig | None = None  # type: ignore[assignment] # because the parent class has made it's type non-optional

    # Algorithm config.
    algorithm: RLAlgorithmConfig | None = None

    # Data mixer strategy: uniform online/offline or episode-aware online replay.
    mixer: str = "online_offline"
    # Fraction sampled from online replay when using OnlineOfflineMixer.
    online_ratio: float = 0.5
    # Episode-aware replay keeps this fraction of online samples for successful episodes.
    success_replay_fraction: float = 0.25
    # Probability that a successful episode sample selects a close/open command event.
    event_replay_fraction: float = 0.25
    # Start covering every step of one short success every N sampled batches; 0 disables sweeps.
    episode_sweep_interval: int = 10
    # Independent share of online batches reserved for transitions from one complete success.
    trajectory_replay_fraction: float = 0.10
    # Relative episode-selection weight for the shorter half of complete successes.
    trajectory_short_weight: float = 2.0

    def validate(self) -> None:
        super().validate()

        if self.mixer not in {"online_offline", "episode_aware"}:
            raise ValueError(f"Unsupported replay mixer: {self.mixer}")
        for name in (
            "online_ratio", "success_replay_fraction", "event_replay_fraction", "trajectory_replay_fraction"
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")
        if self.episode_sweep_interval < 0:
            raise ValueError("episode_sweep_interval must be non-negative")
        if self.success_replay_fraction + self.trajectory_replay_fraction > 1.0:
            raise ValueError("success_replay_fraction + trajectory_replay_fraction must be at most 1")
        if self.trajectory_short_weight < 1.0:
            raise ValueError("trajectory_short_weight must be at least 1")

        if self.algorithm is None:
            self.algorithm = make_algorithm_config("sac")

        if getattr(self.algorithm, "policy_config", None) is None:
            self.algorithm.policy_config = self.policy
