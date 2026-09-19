#!/usr/bin/env python3
"""Separate FP64 reference agreement from an exact-real distance predicate."""

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import struct
import sys


ROOT = Path(__file__).resolve().parents[1]


def binary32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def outward(value, upper):
    rounded = float(value)
    represented = Fraction.from_float(rounded)
    if (upper and represented < value) or (not upper and represented > value):
        rounded = math.nextafter(rounded, math.inf if upper else -math.inf)
    return rounded


def check_case(name, a, b):
    x = [binary32(a), binary32(b)] + [0.0] * 510
    y = [0.0] * 512
    assert x[:2] == [a, b]
    exact = sum((Fraction.from_float(v) - Fraction.from_float(w)) ** 2
                for v, w in zip(x, y))
    rounded = sum((v - w) * (v - w) for v, w in zip(x, y))
    center = Fraction.from_float(rounded)
    radius = Fraction(1, 1 << 40)
    lo = outward(center * (1 - radius), upper=False)
    hi = outward(center * (1 + radius), upper=True)
    state = "accept" if hi <= 1.0 else "reject" if lo > 1.0 else "unresolved"
    exact_inside = exact <= 1
    assert Fraction.from_float(lo) <= exact <= Fraction.from_float(hi)
    assert state == "unresolved" or (state == "accept") == exact_inside
    return {
        "name": name, "nonzero_prefix": x[:2], "shape": [2, 512],
        "exact_squared_distance": str(exact), "exact_margin_above_T": str(exact - 1),
        "binary64_squared_distance": rounded, "binary64_hex": rounded.hex(),
        "exact_accept": exact_inside, "binary64_accept": rounded <= 1.0,
        "binary64_disagrees_with_exact": (rounded <= 1.0) != exact_inside,
        "interval_lower": lo, "interval_upper": hi, "interval_state": state,
        "interval_enclosure_pass": True,
        "input_sha256": hashlib.sha256(struct.pack("<1024f", *(x + y))).hexdigest(),
    }


def main():
    cases = [check_case(*case) for case in (
        ("inside", 0.5, 0.0), ("equality", 1.0, 0.0),
        ("just_outside", 1.0, 2.0 ** -27),
        ("clearly_outside", 1.0, 2.0 ** -13),
    )]
    assert [c["name"] for c in cases if c["binary64_disagrees_with_exact"]] == ["just_outside"]
    assert [c["interval_state"] for c in cases] == ["accept", "unresolved", "unresolved", "reject"]
    sources = ["PROTOCOL_G10_CERTIFICATE_AUDIT.md", "src/check_g10_exactness_contract_cpu.py",
               "src/run_g2a_tensorjoin.py", "src/run_g3b_r1_gpu_analytic_certificate.py",
               "src/run_g3c_b_r1_gpu_cascade.py", "src/g9_tc_kernels.py",
               "src/run_g9_tc_refinement.py"]
    result = {
        "experiment_id": "tensorjoin_20260904_g10_fp64_vs_exact_predicate_cpu",
        "host": platform.node(), "platform": platform.platform(), "python": sys.version,
        "executable": sys.executable, "gpu_executed": False, "performance_measured": False,
        "scope": "CPU arithmetic illustration; not an execution of any TensorJoin GPU kernel",
        "all_illustration_checks_pass": True, "squared_threshold": 1.0, "cases": cases,
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in sources},
        "allowed_conclusion": "FP64-oracle equality does not imply an exact-real predicate; an unresolved interval is not a false decision.",
        "not_claimed": "A new GPU failure, public-data error rate, performance result, or production correctness proof.",
    }
    path = ROOT / "results/g10_exactness_contract_cpu_a0.json"
    with path.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
