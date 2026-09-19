#!/usr/bin/env python3
"""Build an isolated SM120/Python adapter for the pinned GDS-Join source."""

from __future__ import annotations

import hashlib
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "external/gpu_self_join"
OUTPUT = PROJECT / "adapters/gds_g2a"
RECEIPT = PROJECT / "receipts/g2a_gds_adapter_build.json"
NVCC = Path("/usr/local/cuda-13.1/bin/nvcc")
REVISION = "8093a4fdea93a24cf50a63ac9b074c31b7f66dfb"
SOURCE_TREE = "42bd3106002e2169be69cf02d14262a0e56f57af"


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def replace_define(text: str, name: str, value: str) -> str:
    pattern = rf"(?m)^#define\s+{re.escape(name)}\s+.*$"
    updated, count = re.subn(pattern, f"#define {name} {value}", text)
    if count != 1:
        raise ValueError(f"Expected one {name} define, found {count}")
    return updated


def run(command: list[str], cwd: Path) -> None:
    print("COMMAND " + json.dumps(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    if OUTPUT.exists() or RECEIPT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT} or {RECEIPT}")
    if not SOURCE.is_dir() or not NVCC.is_file():
        raise FileNotFoundError(f"Missing source/toolkit: {SOURCE}, {NVCC}")

    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp.{os.getpid()}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        SOURCE,
        temporary,
        ignore=shutil.ignore_patterns(
            ".git", "build*", "*.o", "main", "libgpuselfjoin.so", "gpu_stats.txt"
        ),
    )
    commands: list[list[str]] = []
    try:
        params_path = temporary / "params.h"
        original_params = params_path.read_text(encoding="utf-8")
        params = replace_define(original_params, "GPUNUMDIM", "512")
        params = replace_define(params, "NUMINDEXEDDIM", "6")
        params = replace_define(params, "DTYPE", "double")
        params_path.write_text(params, encoding="utf-8")

        common = [
            str(NVCC),
            "-std=c++17",
            "-O3",
            "-Xcompiler=-fopenmp",
            "-lineinfo",
            "-D_MWAITXINTRIN_H_INCLUDED",
            "-D_FORCE_INLINES",
            "-DPYTHON",
            "-Xcompiler=-fPIC",
            "-arch=sm_120",
        ]
        for source_name in ("GPU.cu", "kernel.cu", "main.cu"):
            object_name = Path(source_name).with_suffix(".o").name
            command = [*common, "-c", source_name, "-o", object_name]
            commands.append(command)
            run(command, temporary)
        link = [
            str(NVCC),
            "-std=c++17",
            "-O3",
            "-Xcompiler=-fopenmp",
            "-shared",
            "GPU.o",
            "kernel.o",
            "main.o",
            "-o",
            "libgpuselfjoin.so",
            "-lcuda",
        ]
        commands.append(link)
        run(link, temporary)
        library = temporary / "libgpuselfjoin.so"
        if not library.is_file() or library.stat().st_size == 0:
            raise RuntimeError("GDS-Join shared library was not produced")

        diff = temporary / "g2a_params.patch"
        diff.write_text(
            "".join(
                difflib.unified_diff(
                    original_params.splitlines(keepends=True),
                    params.splitlines(keepends=True),
                    fromfile="upstream/params.h",
                    tofile="adapter/params.h",
                )
            ),
            encoding="utf-8",
        )
        os.replace(temporary, OUTPUT)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    library = OUTPUT / "libgpuselfjoin.so"
    receipt = {
        "name": "GDS-Join G2A canonical-output adapter",
        "upstream_revision": REVISION,
        "upstream_tree": SOURCE_TREE,
        "upstream_path": str(SOURCE),
        "adapter_path": str(OUTPUT),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "params_sha256": sha256_file(OUTPUT / "params.h"),
        "patch_sha256": sha256_file(OUTPUT / "g2a_params.patch"),
        "library_bytes": library.stat().st_size,
        "library_sha256": sha256_file(library),
        "nvcc": subprocess.check_output([str(NVCC), "--version"], text=True),
        "commands": commands,
        "configuration": {
            "GPUNUMDIM": 512,
            "NUMINDEXEDDIM": 6,
            "DTYPE": "double",
            "STAMP": 0,
            "REORDER": 1,
            "architecture": "sm_120",
            "language_standard": "c++17",
            "python_api": "GDSJoinPy/copyResultIntoPythonArray",
        },
        "algorithm_change": False,
    }
    atomic_json(RECEIPT, receipt)
    print("BUILD_COMPLETE " + json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
