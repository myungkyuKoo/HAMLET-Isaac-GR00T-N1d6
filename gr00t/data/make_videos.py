#!/usr/bin/env python
"""Generate the `videos/` directory of a LeRobot v2.1 dataset from parquet-embedded images.

Some LeRobot datasets on the Hub (e.g., RoboMME `Yinpei/robomme_data_lerobot`) store camera
frames as `image`-dtype features inside the per-episode parquet files and ship no `videos/`
directory, while `meta/info.json` still declares a `video_path` pattern. The GR00T episode
loader decodes frames exclusively from that pattern
(`videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4`), so such
datasets must be transcoded once before training.

Usage:
    python gr00t/data/make_videos.py --dataset-path data/robomme [--num-workers 16]

For every feature with dtype == "image" in meta/info.json, each episode's embedded frames
are decoded and re-encoded as one mp4 (h264, yuv420p, fps from meta/info.json, GOP=2 for
fast random access during training). Parquet and video locations are both resolved from
the `data_path` / `video_path` patterns in meta/info.json. An existing output is kept
only if it opens as a video whose frame count matches the episode parquet; zero-byte,
truncated, or wrong-length files are re-encoded, so interrupted runs resume safely.
"""

from __future__ import annotations

import io
import json
from multiprocessing import Pool
from pathlib import Path

import av
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
from tqdm import tqdm


def _encode_episode_video(
    frames: list[bytes], out_path: Path, fps: int, width: int, height: int
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp.mp4")
    with av.open(str(tmp_path), mode="w") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        # Short GOP: training samples a handful of random frames per episode, so seek
        # speed dominates decode cost; mirrors LeRobot's default encoder settings.
        stream.options = {"g": "2", "crf": "23"}
        for png_bytes in frames:
            img = np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    tmp_path.rename(out_path)


def _video_is_valid(path: Path, expected_frames: int) -> bool:
    """True when `path` opens as a video whose stream reports exactly `expected_frames`.

    Used to vet pre-existing outputs before skipping them: a bare existence check would
    keep zero-byte or truncated files (e.g., from an interrupted copy) forever."""
    try:
        with av.open(str(path)) as container:
            return container.streams.video[0].frames == expected_frames
    except Exception:
        return False


def _convert_episode(args: tuple[str, int, list[str], str, str, int, int]) -> str | None:
    dataset_path, episode_index, image_keys, data_path_pattern, video_path_pattern, chunk_size, fps = args
    root = Path(dataset_path)
    chunk_idx = episode_index // chunk_size
    parquet_path = root / data_path_pattern.format(
        episode_chunk=chunk_idx, episode_index=episode_index
    )
    expected_frames = pq.read_metadata(parquet_path).num_rows

    pending = {}
    for key in image_keys:
        out_path = root / video_path_pattern.format(
            episode_chunk=chunk_idx, video_key=key, episode_index=episode_index
        )
        if not _video_is_valid(out_path, expected_frames):
            pending[key] = out_path
    if not pending:
        return None

    table = pq.read_table(parquet_path, columns=list(pending.keys()))
    for key, out_path in pending.items():
        cells = table[key].to_pylist()
        frames = [c["bytes"] for c in cells]
        first = np.asarray(Image.open(io.BytesIO(frames[0])))
        _encode_episode_video(frames, out_path, fps, first.shape[1], first.shape[0])
    return None


def main(dataset_path: Path | str, num_workers: int = 16, episodes: int | None = None):
    root = Path(dataset_path)
    with open(root / "meta" / "info.json") as f:
        info = json.load(f)

    image_keys = [k for k, ft in info["features"].items() if ft.get("dtype") == "image"]
    if not image_keys:
        print("No image-dtype features found; nothing to do (dataset already video-based?).")
        return
    data_path_pattern = info.get("data_path")
    assert data_path_pattern, "meta/info.json has no data_path pattern to read from"
    video_path_pattern = info.get("video_path")
    assert video_path_pattern, "meta/info.json has no video_path pattern to write to"

    total = int(info["total_episodes"]) if episodes is None else episodes
    fps = int(info.get("fps", 30))
    chunk_size = int(info["chunks_size"])
    print(f"Encoding {total} episodes x {image_keys} -> {video_path_pattern} (fps={fps})")

    jobs = [
        (str(root), ep, image_keys, data_path_pattern, video_path_pattern, chunk_size, fps)
        for ep in range(total)
    ]
    with Pool(num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(_convert_episode, jobs), total=len(jobs)):
            pass
    print("Done.")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
