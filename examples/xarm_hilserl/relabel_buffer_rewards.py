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
Review every episode in the replay buffers of a HIL-SERL run and correct the success labels.

The learner saves its replay buffers with every checkpoint, as `dataset/` (online) and `dataset_offline/`
(demos + interventions) in the run's output directory, and reloads them with `--resume=true`. This tool
shows each episode of the online buffer at its last frame (scrub to look at the rest), and you label how
the episode ended:

- success: the last frame gets reward 1 and ends the episode (`done`). A classifier reward elsewhere in the
  episode is set to 0.
- failure: every reward in the episode is set to 0. `done` is kept: the episode did end there, and the replay
  buffer takes the next stored step as the next state, which after `done` is the next episode's reset.

The same changes go to the offline buffer, where interventions are copied. Demo frames are not touched.

The run is never modified. Your decisions are saved in `<run>/relabel_decisions.json` after every session,
so you can stop with q and continue later. Once every episode is labeled (or with `--write`), the relabeled
buffers are written to a copy of the run (`<run>_relabeled` by default) whose checkpoint config points at the
copy. Resume from the copy. Stop the learner after a checkpoint first: later training is not in the copy.

Keys: Enter / y = ended in success, Backspace / n = failure, s = skip, u = previous episode,
a / d (or the arrow keys) = one frame back / forward, A / D = 10 frames, q = save and quit,
Esc = quit without saving this session.

Example:

```shell
uv run --with "opencv-python>=4.12,<4.14" python examples/xarm_hilserl/relabel_buffer_rewards.py \
    --run-dir outputs/train/2026-10-19/17-34-24_xarm7_hilserl
uv run python -m lerobot.rl.learner --config_path examples/xarm_hilserl/configs/train.json \
    --output_dir=outputs/train/2026-10-19/17-34-24_xarm7_hilserl_relabeled --resume=true
```
"""

import argparse
import contextlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

# Imported before lerobot: creating an OpenCV (Qt) window hangs if lerobot's dependencies are loaded first.
import cv2
import numpy as np
import pandas as pd
import torch

from lerobot.datasets.dataset_tools import recompute_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import (
    ACTION,
    CHECKPOINTS_DIR,
    DEFAULT_FEATURES,
    DONE,
    LAST_CHECKPOINT_LINK,
    OBS_STATE,
    PRETRAINED_MODEL_DIR,
    REWARD,
)

TRAIN_CONFIG_NAME = "train_config.json"
ONLINE_DIR = "dataset"
OFFLINE_DIR = "dataset_offline"
DECISIONS_FILE = "relabel_decisions.json"
INTERVENTION_KEY = "complementary_info.is_intervention"
CAMERA_KEYS = ("observation.images.side", "observation.images.front", "observation.images.wrist")
TILE_SIZE = 384
EXPORT_EPISODE_LENGTH = 50

KEY_ENTER = (13, 10)
KEY_BACKSPACE = (8, 65288)
KEY_ESC = 27
KEY_LEFT = (ord("a"), 65361, 2424832)
KEY_RIGHT = (ord("d"), 65363, 2555904)
KEY_LEFT_10 = (ord("A"),)
KEY_RIGHT_10 = (ord("D"),)

# Decisions are keyed by the episode's last frame; the value is the success frame's key, or None for failure.
Decisions = dict[str, str | None]


@dataclass
class Episode:
    index: int
    start: int  # first frame, as a dataset index
    keys: list[str]  # frame keys, in order
    reward_offsets: list[int]  # frames with a reward, relative to `start`
    intervention_frames: int

    @property
    def key(self) -> str:
        return self.keys[-1]

    def __len__(self) -> int:
        return len(self.keys)


def load_frames(dataset_dir: Path) -> pd.DataFrame:
    """All frames of a saved buffer, in buffer order."""
    frames = [pd.read_parquet(path) for path in sorted((dataset_dir / "data").glob("*/*.parquet"))]
    return pd.concat(frames, ignore_index=True).sort_values("index", ignore_index=True)


def frame_keys(df: pd.DataFrame) -> list[str]:
    """Identify a transition by its state and action. An intervention copied to the offline buffer keeps both."""
    state = np.stack(df[OBS_STATE].to_numpy()).astype(np.float32)
    action = np.stack(df[ACTION].to_numpy()).astype(np.float32)
    return [(s.tobytes() + a.tobytes()).hex() for s, a in zip(state, action, strict=True)]


def rewards(df: pd.DataFrame) -> np.ndarray:
    return np.stack(df[REWARD].to_numpy()).reshape(-1)


def load_episodes(dataset_dir: Path) -> list[Episode]:
    df = load_frames(dataset_dir)
    df["_key"] = frame_keys(df)
    df["_reward"] = rewards(df)
    df["_intervention"] = (
        np.stack(df[INTERVENTION_KEY].to_numpy()).reshape(-1) > 0 if INTERVENTION_KEY in df else False
    )
    episodes = []
    for episode_index, group in df.groupby("episode_index", sort=True):
        episodes.append(
            Episode(
                index=int(episode_index),
                start=int(group["index"].iloc[0]),
                keys=list(group["_key"]),
                reward_offsets=[int(i) for i in np.flatnonzero(group["_reward"].to_numpy() > 0)],
                intervention_frames=int(group["_intervention"].sum()),
            )
        )
    return episodes


def label_changes(episodes: list[Episode], decisions: Decisions) -> tuple[set[str], set[str]]:
    """Frame keys whose reward must be cleared, and frame keys that become successes."""
    clear, success = set(), set()
    for episode in episodes:
        if episode.key not in decisions:
            continue
        success_key = decisions[episode.key]
        clear |= {episode.keys[i] for i in episode.reward_offsets} - {success_key}
        if success_key is not None:
            success.add(success_key)
    return clear, success


def relabel_buffer(dataset_dir: Path, clear: set[str], success: set[str]) -> tuple[int, int]:
    """Clear the rewards of `clear` (keeping `done`) and make `success` rewarded terminal frames.

    Returns the number of rewards cleared and of successes added.
    """
    cleared = added = 0
    for path in sorted((dataset_dir / "data").glob("*/*.parquet")):
        df = pd.read_parquet(path)
        keys = frame_keys(df)
        has_reward = rewards(df) > 0
        to_clear = np.array([k in clear for k in keys]) & has_reward
        to_add = np.array([k in success for k in keys]) & ~has_reward
        to_done = np.array([k in success for k in keys])
        if not (to_clear.any() or to_add.any() or to_done.any()):
            continue
        # Keep each value's type: depending on how the dataset was written, these are scalars or 1-element arrays.
        df[REWARD] = [
            r * 0 if c else (r * 0 + 1 if a else r)
            for c, a, r in zip(to_clear, to_add, df[REWARD], strict=True)
        ]
        df[DONE] = [d | True if s else d for s, d in zip(to_done, df[DONE], strict=True)]
        df.to_parquet(path, index=False)
        cleared += int(to_clear.sum())
        added += int(to_add.sum())
    return cleared, added


def to_bgr(image) -> np.ndarray:
    rgb = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (TILE_SIZE, TILE_SIZE))


def render(dataset: LeRobotDataset, episode: Episode, offset: int, header: str) -> np.ndarray:
    """The cameras side by side at frame `offset` of the episode."""
    frame = dataset[episode.start + offset]
    canvas = np.hstack([to_bgr(frame[key]) for key in CAMERA_KEYS if key in frame])
    label = f"frame {offset}/{len(episode) - 1}"
    if offset in episode.reward_offsets:
        label += " (classifier reward)"
    if offset == len(episode) - 1:
        label += " (last)"
    cv2.putText(canvas, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    bar = np.zeros((60, canvas.shape[1], 3), dtype=np.uint8)
    for i, line in enumerate(header.split("\n")):
        cv2.putText(
            bar, line, (6, 22 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
        )
    return np.vstack([bar, canvas])


def describe(episode: Episode, decisions: Decisions) -> str:
    if episode.key not in decisions:
        return "unlabeled"
    return "failure" if decisions[episode.key] is None else "success"


def start_offset(episode: Episode, decisions: Decisions) -> int:
    """Labels are about how the episode ended, so start at the last frame."""
    return len(episode) - 1


def review(
    dataset: LeRobotDataset, episodes: list[Episode], to_review: list[int], decisions: Decisions
) -> bool:
    """Label the episodes at the positions `to_review` in `decisions`. Returns False to quit without saving."""
    window = "relabel episodes"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(window, 100, 100)
    # Not every OpenCV GUI backend supports it; without it the window can open behind the terminal.
    with contextlib.suppress(cv2.error):
        cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
    print(
        f"Opened the '{window}' window (it may be behind other windows: use Alt+Tab). "
        "Click it so it gets the key presses."
    )
    position = 0
    offset = None
    try:
        while position < len(to_review):
            episode = episodes[to_review[position]]
            if offset is None:
                offset = start_offset(episode, decisions)
            classifier = (
                f"classifier reward @ {', '.join(map(str, episode.reward_offsets))}"
                if episode.reward_offsets
                else "no classifier reward"
            )
            labeled = sum(e.key in decisions for e in episodes)
            header = (
                f"episode {position + 1}/{len(to_review)} (ep {episode.index}, {labeled}/{len(episodes)} labeled)  "
                f"{classifier}  interv {episode.intervention_frames}  [{describe(episode, decisions)}]\n"
                "Enter/y ended in success  Bksp/n failure  s skip  u back  a/d A/D scrub  q save+quit"
            )
            cv2.imshow(window, render(dataset, episode, offset, header))
            key = cv2.waitKeyEx(0)
            if key in KEY_ENTER or key == ord("y"):
                decisions[episode.key] = episode.keys[-1]
            elif key in KEY_BACKSPACE or key == ord("n"):
                decisions[episode.key] = None
            elif key == ord("s"):
                pass
            elif key == ord("u"):
                position, offset = max(position - 1, 0), None
                continue
            elif key in KEY_LEFT or key in KEY_RIGHT or key in KEY_LEFT_10 or key in KEY_RIGHT_10:
                step = 10 if key in KEY_LEFT_10 or key in KEY_RIGHT_10 else 1
                step = -step if key in KEY_LEFT or key in KEY_LEFT_10 else step
                offset = int(np.clip(offset + step, 0, len(episode) - 1))
                continue
            elif key == ord("q"):
                return True
            elif key == KEY_ESC:
                return False
            else:
                continue
            position, offset = position + 1, None
        return True
    finally:
        cv2.destroyAllWindows()


def write_relabeled_run(
    run_dir: Path, output_dir: Path, episodes: list[Episode], decisions: Decisions
) -> dict[str, tuple[int, int]]:
    """Copy the run to `output_dir` and relabel the copy's buffers. Returns (cleared, added) per buffer."""
    # Keep the `checkpoints/last` symlink as a link.
    shutil.copytree(run_dir, output_dir, symlinks=True)
    # On resume the learner replaces its config by the checkpoint's, `output_dir` included.
    config_path = (
        output_dir / CHECKPOINTS_DIR / LAST_CHECKPOINT_LINK / PRETRAINED_MODEL_DIR / TRAIN_CONFIG_NAME
    )
    if config_path.exists():
        config = json.loads(config_path.read_text())
        config["output_dir"] = str(output_dir)
        config_path.write_text(json.dumps(config, indent=4))
    # The copy's buffers are relabeled, so its decisions would not match them any more.
    (output_dir / DECISIONS_FILE).unlink(missing_ok=True)

    clear, success = label_changes(episodes, decisions)
    changes = {}
    for name in (ONLINE_DIR, OFFLINE_DIR):
        buffer_dir = output_dir / name
        if not buffer_dir.exists():
            continue
        changes[name] = relabel_buffer(buffer_dir, clear, success)
        if any(changes[name]):
            recompute_stats(LeRobotDataset(repo_id="local/relabel", root=buffer_dir))
        print(f"{name}: {changes[name][0]} rewards cleared, {changes[name][1]} successes added")
    return changes


def export_classifier_dataset(
    run_dir: Path,
    episodes: list[Episode],
    decisions: Decisions,
    repo_id: str,
    template_repo_id: str,
    holdout: float = 0.15,
    negatives_per_success: int = 3,
    seed: int = 0,
) -> list[int]:
    """Write the labeled episodes as reward-classifier data, in the format of `template_repo_id`.

    Labels are about how each episode ended. Successes are the last frames of the success episodes; their
    earlier frames are left out, as the task may already have been done before the end. Failures are every
    classifier reward you rejected (the classifier's mistakes), topped up with random frames from failure
    episodes, up to `negatives_per_success` failures per success. A `holdout` fraction of the labeled
    episodes is left out for evaluation. Returns their episode indexes.
    """
    rng = np.random.default_rng(seed)
    labeled = [e for e in episodes if e.key in decisions]
    holdout_count = round(len(labeled) * holdout)
    held_out = {labeled[i].index for i in rng.permutation(len(labeled))[:holdout_count]}

    positives, hard_negatives, other_negatives = [], [], []
    for episode in labeled:
        if episode.index in held_out:
            continue
        success_key = decisions[episode.key]
        if success_key is None:
            hard = set(episode.reward_offsets)
            hard_negatives += [(episode, i) for i in sorted(hard)]
            other_negatives += [(episode, i) for i in range(len(episode)) if i not in hard]
        else:
            positives.append((episode, episode.keys.index(success_key)))
    budget = max(0, negatives_per_success * len(positives) - len(hard_negatives))
    picks = rng.choice(len(other_negatives), size=min(budget, len(other_negatives)), replace=False)
    negatives = hard_negatives + [other_negatives[i] for i in sorted(picks)]

    selected = sorted(
        (episode.start + offset, reward)
        for (episode, offset), reward in [(p, 1.0) for p in positives] + [(n, 0.0) for n in negatives]
    )
    # The classifier ignores episodes. Pack the frames into episodes of `EXPORT_EPISODE_LENGTH`, with no
    # 1-frame episode at the end: the metadata writer fails on 1-frame episodes.
    chunks = [selected[i : i + EXPORT_EPISODE_LENGTH] for i in range(0, len(selected), EXPORT_EPISODE_LENGTH)]
    if len(chunks) > 1 and len(chunks[-1]) == 1:
        chunks[-2] += chunks.pop()

    template = LeRobotDataset(template_repo_id)
    features = {k: v for k, v in template.meta.features.items() if k not in DEFAULT_FEATURES}
    task = str(template.meta.tasks.index[0])
    source = LeRobotDataset(repo_id="local/relabel", root=run_dir / ONLINE_DIR)
    dataset = LeRobotDataset.create(
        repo_id,
        fps=template.meta.fps,
        features=features,
        robot_type=template.meta.robot_type,
        use_videos=True,
    )
    for frames in chunks:
        for i, (index, reward) in enumerate(frames):
            frame = source[index]
            dataset.add_frame(
                {
                    # The buffer stores some 1-element features as scalars.
                    **{
                        k: frame[k].reshape(tuple(features[k]["shape"]))
                        for k in features
                        if k not in (REWARD, DONE)
                    },
                    REWARD: torch.tensor([reward], dtype=torch.float32),
                    DONE: torch.tensor([i == len(frames) - 1]),
                    "task": task,
                }
            )
        dataset.save_episode()
    dataset.finalize()
    (dataset.root / "meta" / "relabel_holdout.json").write_text(
        json.dumps({"run_dir": str(run_dir), "held_out_episodes": sorted(held_out)}, indent=1)
    )
    print(
        f"Exported {len(positives)} successes and {len(negatives)} failures ({len(hard_negatives)} rejected "
        f"classifier rewards) from {len(labeled) - len(held_out)} episodes to {dataset.root}. "
        f"Held out {len(held_out)} episodes for evaluation."
    )
    return sorted(held_out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="Output directory of the learner.")
    parser.add_argument(
        "--output-run-dir",
        type=Path,
        default=None,
        help="Where to write the relabeled run. Defaults to `<run>_relabeled`.",
    )
    parser.add_argument("--all", action="store_true", help="Also show the episodes already labeled.")
    parser.add_argument(
        "--successes",
        action="store_true",
        help="Only show the episodes labeled as successes, to check them again.",
    )
    parser.add_argument(
        "--write", action="store_true", help="Write the relabeled run even if some episodes are unlabeled."
    )
    parser.add_argument(
        "--export-classifier",
        default=None,
        metavar="REPO_ID",
        help="Instead of reviewing, export the labeled episodes as reward-classifier data to this dataset.",
    )
    parser.add_argument(
        "--classifier-template",
        default="local/xarm7_hilserl_classifier",
        help="Classifier dataset whose format the export copies (and that it is meant to be merged with).",
    )
    parser.add_argument(
        "--holdout", type=float, default=0.15, help="Fraction of episodes left out of the export."
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir.resolve()
    output_dir: Path = (args.output_run_dir or run_dir.with_name(run_dir.name + "_relabeled")).resolve()
    online_dir = run_dir / ONLINE_DIR
    decisions_path = run_dir / DECISIONS_FILE
    decisions: Decisions = json.loads(decisions_path.read_text()) if decisions_path.exists() else {}

    episodes = load_episodes(online_dir)
    if args.export_classifier:
        export_classifier_dataset(
            run_dir, episodes, decisions, args.export_classifier, args.classifier_template, args.holdout
        )
        return
    if args.successes:
        to_review = [i for i, e in enumerate(episodes) if decisions.get(e.key) is not None]
    else:
        to_review = [i for i, e in enumerate(episodes) if args.all or e.key not in decisions]
    print(
        f"{len(episodes)} episodes in {online_dir}, {sum(bool(e.reward_offsets) for e in episodes)} with a "
        f"classifier reward. {len(to_review)} to review."
    )
    if to_review:
        dataset = LeRobotDataset(repo_id="local/relabel", root=online_dir)
        if not review(dataset, episodes, to_review, decisions):
            print("Quit without saving this session.")
            return
        decisions_path.write_text(json.dumps(decisions, indent=1))

    labeled = [e for e in episodes if e.key in decisions]
    successes = sum(decisions[e.key] is not None for e in labeled)
    print(
        f"Labeled {len(labeled)}/{len(episodes)} episodes: {successes} successes, "
        f"{len(labeled) - successes} failures. Decisions saved to {decisions_path}."
    )
    if args.successes and not args.write:
        print("Pass --write (and a new --output-run-dir) to write a relabeled run with these labels.")
        return
    if len(labeled) < len(episodes) and not args.write:
        print("Run again to continue, or pass --write to write the relabeled run now.")
        return
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists, pass another --output-run-dir.")
    write_relabeled_run(run_dir, output_dir, episodes, decisions)
    print(
        f"Wrote the relabeled run to {output_dir} ({run_dir}'s data is unchanged). Resume from it with "
        f"--output_dir={output_dir} --resume=true."
    )


if __name__ == "__main__":
    main()
