# G9 GPU7 revision R1

Declared before any G9 timing, 2026-09-04. Numerical and statistical contracts
remain `PROTOCOL_G9_TC_REFINEMENT_A0.md` and `NUMERICS_G9_INT14.md`.

G9 check A0 failed a boundary correctness test on GPU1; no timings were taken.
The corrected check A1 then failed preflight without launching because foreign
PID1084864 occupied GPU1 (about52 GiB). Do not touch it or wait for low
utilization as permission to share the device.

Move all fresh correctness, sanitizer, and timing processes to physical GPU7:
UUID GPU-e5a0e7f5-9917-c2a1-ee8e-7282cbba2603. The device has the same reported
model/driver and was freshly observed at14 MiB,0% utilization with no compute
process. Use its existing `.tensorjoin_gpu7_campaign.lock` exclusively; do not
hold unrelated GPU1 locks. Preserve the5-second quiescence and0.25-second
foreign-process monitoring. No clocks/power settings change.

Both performance processes and their comparator are measured on GPU7. Do not
combine GPU1 measurements with GPU7 or reuse G8 GPU1 timing as the denominator.
This is an occupancy-driven host revision, not favorable-result selection.

The boundary fix uses explicit FP64 constants and compile-time dtype checks
on every directed-FP64 helper. Preserve the failed original code and PTX.
Fresh check label: `g9_check_a2`; then `g9_memcheck_a0`, `g9_synccheck_a0`,
`g9_timing_p0_a0`, and reversed `g9_timing_p1_a0`, all on GPU7.
