# Callable hidden-E3M4 instruction probes on SM120

**E3M4 is callable on both tested dense instruction families.** This artifact
reproduces a bounded format matrix on RTX PRO 6000 Blackwell Server Edition,
SM120, CUDA 13.1.115, driver 590.48.01. It is an undocumented research probe,
not an NVIDIA-supported API, a tuned GEMM, or a new selector discovery.

## Support matrix

| Instruction family | Documented control pairs | Pairs containing E3M4 | Result |
|---|---:|---:|---|
| Plain `QMMA.16832.F32` | 25 | 11 | 36/36 numerical probes passed |
| Block-scaled `QMMA.SF.16832.F32` with UE8M0 | 25 | 11 | 36/36 numerical probes passed |

E3M4 can be A or B with each of E4M3FN, E5M2, E3M2, E2M3, E2M1, or E3M4.
The 11 count is six choices on each axis, minus the shared E3M4/E3M4 pair.
Each pair includes every valid raw code on both operand axes, 128 deterministic
finite mixed-code records, and explicit E3M4-vs-E4M3 discrimination where
applicable. E3M4 contributes all 256 codes on each tested axis. Scaled probes
also test power-of-two scale products with positive and negative exponents.
Every record checks all 128 FP32 tile outputs, including NaN classification and
infinity sign. Signed-zero sign is not required.

The probe inputs are **uniform within each tile**. Different test records have
different values, but this does not validate a nonuniform matrix's lane layout,
arbitrary scale-selector layout, or general GEMM. No speedup is measured here.

## Recovered invocation boundary

Both families are compiled separately from their documented E4M3 forms.
The 25 compiler-generated controls per family confirm these LSB0 instruction
fields, with bits listed from low to high significance:

| Operand | Physical bits | Absolute format code |
|---|---|---|
| A | 78, 82, 83 | E4M3=0, E5M2=1, **E3M4=2**, E3M2=3, E2M3=4, E2M1=5 |
| B | 79, 84, 85 | Same mapping |

Starting from E4M3/E4M3, selecting E3M4 on A changes bit 82; on B it changes
bit 84. These are **instruction-local fields**, not global enable bits. Each
candidate is a private cubin copy; the patcher proves that no other binary bit
changes. Plain-to-scaled conversion by guessed bit toggles is not used.

The absolute type code above must not be confused with the historical donor's
XOR-delta pattern numbered relative to an E2M1 baseline. A different baseline,
architecture, shape or instruction family requires fresh derivation.

The format oracle uses E3M4 bias 3, minimum subnormal 1/64, minimum normal 1/4,
maximum finite 30, and NaN magnitude code 0x7f. For example, with K=32 and
unit scales, raw 0x10 multiplied by itself gives **2.0** under E3M4 versus
**0.03125** under E4M3FN. Raw 0x3c gives **98.0** under E3M4 versus **72.0**
under E4M3FN. This explicitly rejects an unchanged E4M3 execution path.

## Use the owned interface

Build on the target Linux CUDA host, from this directory:

```sh
python3 survey.py selftest
python3 survey.py build --build build
```

The build verifies the exact compiler version, all 50 documented controls,
the one-instruction shape, format fields and all 22 copied/modified binaries.
It saves binary and source hashes in `build/manifest.json`. Required tools:
CUDA 13.1.115, a C++17 host compiler, Python 3, `readelf`, and CUDA driver access.
No Python package installation is needed. `CUDA_HOME` may override the toolkit
location, but not the pinned compiler version.

The Python interface accepts raw codes, **not already-quantized matrices**:

```python
from pathlib import Path
from survey import run_probe

report = run_probe(
    Path("build").resolve(), "scaled", "e3m4", "e3m4",
    records=[(0x10, 0x10, 0, 0), (0x3c, 0x3c, -2, 3)],
    output_dir=Path("demo-output").resolve(),
)
assert report["pass"]
# First output of each tile: 2.0, 196.0. All 128 outputs are checked.
```

The last two integers are power-of-two scale exponents. The current safe probe
domain is -5 through 4; the plain family requires zero exponents. E2M1 codes
are placed in the central four bits of each byte as required by f8f6f4;
FP6 codes occupy the lower six bits. Scale bytes repeat across their source
registers and all lane selectors are zero. Each call creates a fresh process
and CUDA context, verifies cubin/launcher hashes, rejects non-SM120 devices,
has a 120-second timeout, and refuses to overwrite its evidence files.

Run only on an explicitly idle GPU. For the full survey, reuse this repository's
existing ownership guard (substitute an idle physical GPU index and fresh labels):

```sh
HERE="$PWD"
python3 ../precision_format_routing/src/guard.py --gpu 0 --label e3m4-interface-replay \
  -- python3 "$HERE/survey.py" validate --build "$HERE/build" --out "$HERE/replay"
```

`memcheck`, `stress`, `racecheck`, `synccheck`, and `initcheck` are separate modes
and must also run under ownership supervision. Stress checks every output of
20 consecutive launches per E3M4 pair. It does not test Graph replay or pointer
churn. A raw `run_probe` call does not acquire the campaign lock by itself.
There is no silent format fallback; unsupported configurations fail rather than
masquerading as E3M4. Never bypass the CUDA error stop by resetting a shared GPU.

## Evidence, limitations, and retained development failures

The final source revision passed all declared gates on 2026-09-19:

| Gate | Scope | Result |
|---|---|---|
| Numerical survey | 72 fresh processes; 4,027,904 FP32 outputs | Zero mismatches |
| Memcheck | All 22 E3M4-containing family/pair probes | Zero errors |
| Repeated launches | 20 per E3M4 probe, 440 total | All 30,074,880 outputs passed |
| Racecheck / synccheck / initcheck | E3M4/E3M4 in both families, 6 runs | Zero errors; zero race hazards |
| Low-optimization compiler controls | 12 probes, both nvcc and ptxas `-O0` | Same absolute format fields |
| Dual disassembly | cuobjdump and nvdisasm on both E4M3/E4M3 and E3M4/E3M4 families | Logs retained; not used as numerical proof |
| Direct PTX E3M4 spelling | Plain and scaled | Compiler rejected `.e3m4` in both |

Offline replay checked **36,107,264 outputs** across all retained validation,
sanitizer and stress records. Kernels reported 16 registers (plain) or 20
(scaled), zero local-memory bytes and zero shared-memory bytes. These resource
figures are not throughput or occupancy measurements. GPU 0 was supervised;
all owned processes exited. No foreign GPU work was stopped or reset.

Both disassemblers exited successfully but did **not** name the hidden E3M4
instruction: the mnemonic is missing at plain address `0x00e0` and scaled
address `0x0100` (nvdisasm omits those instruction lines). Decoder exit success
therefore does not mean E3M4 decode support. The bit-difference proof and
independent hardware numerical results, not a printed mnemonic, establish the
tested format identity.

- [Frozen contract](CONTRACT.md)
- [Source/binary/field manifest](results/BUILD_MANIFEST.json)
- [Full numerical survey](results/validate/SUMMARY.json)
- [Memcheck](results/memcheck/SUMMARY.json), [stress](results/stress/SUMMARY.json),
  [racecheck](results/racecheck/SUMMARY.json),
  [synccheck](results/synccheck/SUMMARY.json),
  [initcheck](results/initcheck/SUMMARY.json)
- [Low-optimization and decoder checks](results/static/STATIC_CHECKS.json),
  [numeric GPU supervision](results/GPU_SUPERVISION.json)
- Per-probe JSON records and losslessly compressed raw FP32 outputs live beside
  each summary. `verify_evidence.py` rechecks every retained output offline.
- Binaries stay outside Git; their hashes and deterministic build procedure are
  retained. Source identities must match the tested build manifest.
- The first build used immediate zero accumulator operands and failed in the
  scaled PTX form. Changing only selectors to typed u16 did not fix that error.
  Supplying zero through a typed FP32 operand fixed compilation. Neither failed
  build was executed on a GPU or counted in the final matrix.
- A read-only review caught missing timeout diagnostics in the first passing
  driver script. The final script preserves partial output and failure JSON,
  with a CPU subprocess regression test. The complete GPU matrix and all
  sanitizer/stress modes were rerun after this fix; the manifest matches the
  final tested sources rather than the superseded passing run.
- Reserved absolute format codes 6/7 were deliberately **not executed**; this
  is not an exhaustive 128-bit opcode sweep or a fresh proof of their illegality.
- Sparse instructions, F16 accumulation, other tile shapes, native conversion
  instructions, arbitrary matrices, scaled TensorJoin certificates, CUDA Graphs,
  pointer churn, sustained performance and cross-architecture support are not
  established by this artifact. No production path is promoted.

Offline checks and the optional compiler-only supplement:

```sh
python3 test_survey.py
python3 verify_evidence.py
python3 static_checks.py --build build --out static-replay
```

## Prior-art and TensorJoin boundary

[blackwell-isa at 8fe1478](https://github.com/kacper-daftcode/blackwell-isa/tree/8fe1478006d14bb52f6f644d4f7dc1993a85f25a)
already publishes the hidden selector and dense block-scaled combinations;
[cubit at 1eb8dc6](https://github.com/kacper-daftcode/cubit/tree/1eb8dc68d977c323fb6a24a60b8b76cc0813bc59)
provides the complementary assembler. This survey independently recompiles and
tests both families; it does not call cubit or redistribute those projects.
The documented syntax and padding are from
[PTX 9.1](https://docs.nvidia.com/cuda/archive/13.1.1/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma).

The earlier [TensorJoin screen](../precision_format_routing/README.md) used the
plain, per-row-scale path. This survey establishes the scaled instruction's
callability, **not** a block-scaled TensorJoin implementation or an end-to-end win.
Next admission step: nonuniform tile/layout validation, a real block-scaled
GEMM and a defensible certificate, then full-cost comparison with the strongest
fixed-format baseline including conversion and FP32/FP64 refinement costs.
