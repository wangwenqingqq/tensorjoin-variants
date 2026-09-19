# Reproduction and continuation

## Environment

Original host:gpu-host-8, gpu-host-8, RTX PRO6000 Blackwell Server Edition SM120,
driver590.48.01. Original target GPU2 UUID16f27f5a-dfcd-48e0-bb39-bebbe4009245;
secondary functional target GPU0 UUID9bd364a9-9d88-0b78-ef55-caac6f73d6b1.
Remote parent:@TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join.
Python:@TENSORJOIN_ROOT@/isaacsim6/env/bin/python, torch2.11.0+cu130.
Installed cuBLAS version130100; owned handle/default stream0/math16/compute68,
FP16 input and FP32 output, explicit8,519,680-byte workspace.
CUDA13.1 cuobjdump:/usr/local/cuda-13.1/bin/cuobjdump; NCU2025.4.1.

All executed commands and source hashes are in guard JSONs; inspect them rather
than reconstructing a command from a paper speed figure. `targets_r2.json`
separates GPU0 and GPU2; do not combine their timing denominators.

## Artifact-first audit

1. Verify every entry in delivery_manifest and protected_previous against bytes.
2. Inspect selected_trace_audit_r1: one runtime-selected function per shape,
   both full library container hashes and normalized selected SASS hash.
3. Inspect compiled_static_audit and gpu0_abi_a0: actual generated program and
   queried parameter/resource identity, not a source-only inference.
4. Inspect all numerical/safety JSONs together with their corresponding guard
   and raw log. A numerical `pass` cannot override a failed isolation guard.
5. Saved NPZ files allow CPU rational replay via the logic in offline_review.py.
   Scripts create new result files exclusively; replay in a fresh directory or
   choose separately versioned output names, never overwrite a sealed record.

## Runtime scripts and scope

- trace_half.py: owned GemmEx NCU trace; original GPU2 only, shape0..3.
- validate_control_r2.py: fixture/public/full modes, records physical GPU.
- safety_probe.py: stress32 or bounded probe2; consumer Graph only.
- audit_runtime_abi.py: load/query/unload the nine captured cubins.
- original_confirm.py: frozen sequence for original-card confirmation; its
  attempted guard failed before launch, so none of its gpu2_* children ran.

Use a new continuation delivery instead of editing this closed checkpoint.
Keep numeric gpu_kernels.py, owned_half.py, original four G16 cubins, and actual
compiled hashes fixed. The original confirmation sequence checks existing GPU0
qualifications, then ABI, fixture, public panels, full output, stress, and four
sanitizers. Preserve all old records and give the new guard a unique label.
Set NVIDIA_TF32_OVERRIDE=0 and an explicit private TRITON_CACHE_DIR. Recheck
GPU UUID, active users/processes, two campaign locks and30second quiescence.
The guard timeout is1800seconds and device-used cap4096MiB. No clock/power edits.

Only after original-card confirmation and a separately reviewed explicit
conditional-numerical scope may a new A/C timing contract be frozen. Include
preprocessing, matrix computation, classification/queues, both refinements,
all host control/transfers, and output canonicalization. Remeasure both methods
in direction-balanced processes. Do not use raw diagnostic seconds from this
qualification or substitute the previous pedantic-FP32 denominator.
