"""Draw the crop boxes from a config on frames sampled across every episode of a dataset.

Writes one contact sheet: a row per episode, columns at evenly spaced points in time (start ... end),
each cell showing every camera side by side. A camera without a box (or `crop_params_dict: null`) gets
the full frame. Check that everything the task needs stays inside the box in every row.

    uv run python examples/xarm_hilserl/check_crop.py --repo-id local/xarm7_hilserl_demos \
        --config examples/xarm_hilserl/configs/env.json --out-dir outputs/crop_check
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--config", required=True, help="env.json or train.json holding crop_params_dict")
    parser.add_argument("--out-dir", default="outputs/crop_check")
    parser.add_argument("--frames-per-episode", type=int, default=6)
    parser.add_argument("--thumb-width", type=int, default=320)
    args = parser.parse_args()

    crops = (
        json.loads(Path(args.config).read_text())["env"]["processor"]["image_preprocessing"][
            "crop_params_dict"
        ]
        or {}
    )
    dataset = LeRobotDataset(args.repo_id, root=args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    camera_keys = dataset.meta.camera_keys
    for key in crops.keys() - set(camera_keys):
        print(f"warning: {key} has a crop box but is not in the dataset")

    rows = []
    for ep in range(dataset.meta.total_episodes):
        start = dataset.meta.episodes["dataset_from_index"][ep]
        end = dataset.meta.episodes["dataset_to_index"][ep]
        indices = np.linspace(start, end - 1, args.frames_per_episode).round().astype(int)
        cells = []
        for idx in indices:
            item = dataset[int(idx)]
            thumbs = []
            for key in camera_keys:
                img = (item[key].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                img = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                top, left, height, width = crops.get(key, (0, 0, img.shape[0], img.shape[1]))
                cv2.rectangle(img, (left, top), (left + width, top + height), (0, 255, 0), 3)
                scale = args.thumb_width / img.shape[1]
                thumb = cv2.resize(img, None, fx=scale, fy=scale)
                lines = [key.split(".")[-1], f"ep {ep}  t={int(idx - start)}/{end - start - 1}"]
                for i, line in enumerate(lines):
                    cv2.putText(thumb, line, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                thumbs.append(thumb)
            # Separate time points with a white bar so the camera groups are easy to tell apart.
            cells.append(np.hstack(thumbs))
            cells.append(np.full((thumbs[0].shape[0], 8, 3), 255, dtype=np.uint8))
        rows.append(np.hstack(cells[:-1]))
        print(f"episode {ep} done")

    path = out_dir / "crop_check.jpg"
    cv2.imwrite(str(path), np.vstack(rows))
    print(f"wrote {path} (cameras left to right: {', '.join(k.split('.')[-1] for k in camera_keys)})")


if __name__ == "__main__":
    main()
