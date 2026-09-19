#!/usr/bin/env python3
"""Extract deterministic torchvision R3D-18 embeddings from UCF101 videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.video import R3D_18_Weights, r3d_18


EXPERIMENT_ID = "tensorjoin_20260903_ucf101_r3d18_embedding_d2"
SEED = 20260903
FRAME_COUNT = 16
EMBEDDING_DIMENSION = 512
EXPECTED_VIDEOS = 13_320
WEIGHTS = R3D_18_Weights.KINETICS400_V1
WEIGHTS_URL = "https://download.pytorch.org/models/r3d_18-b3b3357e.pth"
GROUP_PATTERN = re.compile(r"^(v_.+_g\d\d)_c\d\d\.avi$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0,
                        help="Nonzero only for an explicitly invalid smoke run")
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_lines(lines: list[str]) -> str:
    payload = "".join(line + "\n" for line in lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def configure_determinism() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)


def clip_metadata(path: Path, root: Path) -> tuple[str, str, str]:
    relative = path.relative_to(root).as_posix()
    category = path.parent.name
    match = GROUP_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected UCF101 filename: {relative}")
    group = match.group(1)
    return relative, category, group


def decode_center_clip(path: Path) -> tuple[np.ndarray | None, int, list[int], str | None]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None, 0, [], "open_failed"
    total = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if total <= 0:
        capture.release()
        return None, total, [], "invalid_frame_count"
    start = max((total - FRAME_COUNT) // 2, 0)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames: list[np.ndarray] = []
    indices: list[int] = []
    for offset in range(min(FRAME_COUNT, total - start)):
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            return None, total, indices, f"read_failed_at_{start + offset}"
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        indices.append(start + offset)
    capture.release()
    if not frames:
        return None, total, [], "no_frames"
    while len(frames) < FRAME_COUNT:
        frames.append(frames[-1].copy())
        indices.append(indices[-1])
    return np.ascontiguousarray(np.stack(frames), dtype=np.uint8), total, indices, None


def main() -> int:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("D2 extraction requires CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError("Refusing to overwrite D2 extraction evidence")
    configure_determinism()

    root = args.dataset_root.resolve()
    checkpoint = args.checkpoint.resolve()
    paths = sorted(root.rglob("*.avi"), key=lambda path: path.relative_to(root).as_posix())
    if len(paths) != EXPECTED_VIDEOS:
        raise ValueError(f"Expected {EXPECTED_VIDEOS} AVI files, found {len(paths)}")
    relative_paths = [path.relative_to(root).as_posix() for path in paths]
    manifest_sha256 = sha256_lines(relative_paths)
    if args.limit:
        paths = paths[: args.limit]
    checkpoint_sha256 = sha256_file(checkpoint)

    model = r3d_18(weights=None)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.fc = torch.nn.Identity()
    model.eval().cuda()
    preprocess = WEIGHTS.transforms()

    print(
        "RUN_START "
        + json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "host": platform.node(),
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "torchvision_weights": str(WEIGHTS),
                "opencv": cv2.__version__,
                "dataset_root": str(root),
                "dataset_manifest_sha256": manifest_sha256,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "batch_size": args.batch_size,
                "decode_workers": args.decode_workers,
                "limit": args.limit,
                "seed": SEED,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    features: list[np.ndarray] = []
    clip_names: list[str] = []
    categories: list[str] = []
    groups: list[str] = []
    decoded_counts: list[int] = []
    selected_indices: list[list[int]] = []
    failures: list[dict[str, str | int]] = []
    start_time = time.time()

    def process_batch(batch_paths: list[Path]) -> None:
        with ThreadPoolExecutor(max_workers=args.decode_workers) as pool:
            decoded = list(pool.map(decode_center_clip, batch_paths))
        valid_frames: list[np.ndarray] = []
        valid_meta: list[tuple[str, str, str, int, list[int]]] = []
        for path, (frames, total, indices, error) in zip(batch_paths, decoded):
            relative, category, group = clip_metadata(path, root)
            if error is not None or frames is None:
                failures.append({"path": relative, "frame_count": total, "error": error or "unknown"})
                continue
            valid_frames.append(frames)
            valid_meta.append((relative, category, group, total, indices))
        if not valid_frames:
            return
        processed = []
        for frames in valid_frames:
            video = torch.from_numpy(frames).permute(0, 3, 1, 2).cuda()
            processed.append(preprocess(video))
        video = torch.stack(processed)
        with torch.inference_mode():
            embedding = model(video).to(torch.float32)
        if embedding.shape != (len(valid_frames), EMBEDDING_DIMENSION):
            raise AssertionError(embedding.shape)
        norms = torch.linalg.vector_norm(embedding, dim=1)
        if not bool(torch.all(torch.isfinite(embedding)).item()):
            raise ValueError("Non-finite video embedding")
        if not bool(torch.all(norms > 0).item()):
            raise ValueError("Zero-norm video embedding")
        embedding = embedding / norms[:, None]
        features.append(embedding.cpu().numpy())
        for relative, category, group, total, indices in valid_meta:
            clip_names.append(relative)
            categories.append(category)
            groups.append(group)
            decoded_counts.append(total)
            selected_indices.append(indices)

    for offset in range(0, len(paths), args.batch_size):
        process_batch(paths[offset : offset + args.batch_size])
        done = min(offset + args.batch_size, len(paths))
        if done % 256 == 0 or done == len(paths):
            print(
                f"EXTRACT_PROGRESS attempted={done}/{len(paths)} valid={len(clip_names)} "
                f"failures={len(failures)} elapsed_s={time.time()-start_time:.3f}",
                flush=True,
            )

    matrix = np.ascontiguousarray(np.concatenate(features, axis=0), dtype=np.float32)
    if len(matrix) < 4_608:
        raise ValueError(f"Only {len(matrix)} valid videos")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        np.savez(
            handle,
            features=matrix,
            clip=np.asarray(clip_names),
            category=np.asarray(categories),
            group=np.asarray(groups),
            decoded_frame_count=np.asarray(decoded_counts, dtype=np.int32),
            selected_frame_indices=np.asarray(selected_indices, dtype=np.int32),
        )
    norms = np.linalg.norm(matrix, axis=1)
    receipt = {
        "experiment_id": EXPERIMENT_ID,
        "formal": args.limit == 0,
        "host": platform.node(),
        "gpu": torch.cuda.get_device_name(0),
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision_weights": str(WEIGHTS),
        "weights_url": WEIGHTS_URL,
        "opencv": cv2.__version__,
        "shape": list(matrix.shape),
        "attempted_videos": len(paths),
        "decode_failures": failures,
        "unique_groups": len(set(groups)),
        "finite": bool(np.isfinite(matrix).all()),
        "zero_norm_vectors": int(np.count_nonzero(norms == 0)),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "dataset_manifest_sha256": manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "script_sha256": sha256_file(Path(__file__)),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "elapsed_s": time.time() - start_time,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("RUN_SUMMARY " + json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
