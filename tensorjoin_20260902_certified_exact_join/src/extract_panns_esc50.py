#!/usr/bin/env python3
"""Extract deterministic PANNs Cnn14 embeddings from ESC-50 segments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


EXPERIMENT_ID = "tensorjoin_20260902_panns_embedding_extract_d0"
ESC50_COMMIT = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
PANN_SAMPLE_RATE = 32_000
SEGMENT_SAMPLES = 32_000
SEGMENTS_PER_CLIP = 5
EMBEDDING_DIMENSION = 2048
CLASSES_NUM = 527
SEED = 20260902


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--panns-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.md5()  # nosec: official artifact identity, not security
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def load_vendor_cnn14(repo: Path):
    package_dir = repo / "panns_inference"
    if not (package_dir / "models.py").is_file():
        raise FileNotFoundError(package_dir / "models.py")
    package_name = "tensorjoin_panns_vendor"
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_dir)]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.models", package_dir / "models.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError(package_dir / "models.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Cnn14


def load_categories(dataset_root: Path) -> dict[str, str]:
    path = dataset_root / "meta" / "esc50.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["filename"]: row["category"] for row in csv.DictReader(handle)}


def configure_determinism() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def load_segments(path: Path) -> list[torch.Tensor]:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(samples).mean(dim=1, keepdim=False).unsqueeze(0)
    if sample_rate != PANN_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, PANN_SAMPLE_RATE,
            resampling_method="sinc_interp_hann",
        )
    waveform = waveform.squeeze(0)
    required = SEGMENTS_PER_CLIP * SEGMENT_SAMPLES
    if waveform.numel() < required:
        waveform = torch.nn.functional.pad(waveform, (0, required - waveform.numel()))
    elif waveform.numel() > required:
        waveform = waveform[:required]
    return [
        waveform[i * SEGMENT_SAMPLES : (i + 1) * SEGMENT_SAMPLES].contiguous()
        for i in range(SEGMENTS_PER_CLIP)
    ]


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    panns_repo = args.panns_repo.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    receipt = args.receipt.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen extraction run")
    configure_determinism()
    audio_paths = sorted((dataset_root / "audio").glob("*.wav"))
    if len(audio_paths) != 2000:
        raise ValueError(f"Expected 2000 ESC-50 WAV files, found {len(audio_paths)}")
    if checkpoint.stat().st_size != 327_428_481:
        raise ValueError(f"Unexpected checkpoint size: {checkpoint.stat().st_size}")
    checkpoint_md5 = md5_file(checkpoint)
    if checkpoint_md5 != "541141fa2ee191a88f24a3219fff024e":
        raise ValueError(f"Unexpected checkpoint MD5: {checkpoint_md5}")
    categories = load_categories(dataset_root)
    Cnn14 = load_vendor_cnn14(panns_repo)
    model = Cnn14(
        sample_rate=PANN_SAMPLE_RATE,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14_000,
        classes_num=CLASSES_NUM,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval().cuda()
    print(
        "RUN_START "
        + json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "host": platform.node(),
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "torchaudio": torchaudio.__version__,
                "dataset_root": str(dataset_root),
                "panns_repo": str(panns_repo),
                "checkpoint": str(checkpoint),
                "checkpoint_md5": checkpoint_md5,
                "batch_size": args.batch_size,
                "seed": SEED,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    features: list[np.ndarray] = []
    clip_names: list[str] = []
    segment_ids: list[int] = []
    category_names: list[str] = []
    batch_waveforms: list[torch.Tensor] = []
    batch_meta: list[tuple[str, int, str]] = []
    start = time.time()

    def flush() -> None:
        if not batch_waveforms:
            return
        audio = torch.stack(batch_waveforms).cuda(non_blocking=False)
        with torch.inference_mode():
            embedding = model(audio, None)["embedding"]
        embedding = embedding.to(torch.float32)
        norms = torch.linalg.vector_norm(embedding, dim=1)
        if not bool(torch.all(torch.isfinite(embedding)).item()):
            raise ValueError("Non-finite embedding")
        if not bool(torch.all(norms > 0).item()):
            raise ValueError("Zero-norm embedding")
        embedding = embedding / norms[:, None]
        features.append(embedding.cpu().numpy())
        for clip, segment, category in batch_meta:
            clip_names.append(clip)
            segment_ids.append(segment)
            category_names.append(category)
        batch_waveforms.clear()
        batch_meta.clear()

    for clip_index, audio_path in enumerate(audio_paths, start=1):
        for segment_id, waveform in enumerate(load_segments(audio_path)):
            batch_waveforms.append(waveform)
            batch_meta.append((audio_path.name, segment_id, categories[audio_path.name]))
            if len(batch_waveforms) == args.batch_size:
                flush()
        if clip_index % 100 == 0:
            flush()
            print(
                f"EXTRACT_PROGRESS clips={clip_index}/2000 vectors={len(clip_names)} "
                f"elapsed_s={time.time()-start:.3f}",
                flush=True,
            )
    flush()
    matrix = np.ascontiguousarray(np.concatenate(features, axis=0), dtype=np.float32)
    expected_shape = (len(audio_paths) * SEGMENTS_PER_CLIP, EMBEDDING_DIMENSION)
    if matrix.shape != expected_shape:
        raise AssertionError((matrix.shape, expected_shape))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        np.savez(
            handle,
            features=matrix,
            clip=np.asarray(clip_names),
            segment=np.asarray(segment_ids, dtype=np.int16),
            category=np.asarray(category_names),
        )
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "esc50_commit": ESC50_COMMIT,
        "host": platform.node(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "shape": list(matrix.shape),
        "finite": bool(np.isfinite(matrix).all()),
        "zero_norm_vectors": int(np.count_nonzero(np.linalg.norm(matrix, axis=1) == 0)),
        "norm_min": float(np.linalg.norm(matrix, axis=1).min()),
        "norm_max": float(np.linalg.norm(matrix, axis=1).max()),
        "checkpoint_md5": checkpoint_md5,
        "checkpoint_sha256": sha256_file(checkpoint),
        "panns_commit": (panns_repo / ".git" / "HEAD").read_text().strip()
        if (panns_repo / ".git" / "HEAD").is_file()
        else "vendored-copy-see-receipt",
        "script_sha256": sha256_file(Path(__file__)),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "elapsed_s": time.time() - start,
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("RUN_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
