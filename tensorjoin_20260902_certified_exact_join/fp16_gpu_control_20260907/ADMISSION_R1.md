# Guard R1: bounded NVML exit-race repair

The original shape-3 trace remains guard-failed. Its numerical JSON identifies
PID552285, matching every running occupancy record. The log records profiler
disconnection and a completed report. The final NVML sample reports that same
PID as `[No data]`, while device memory has already returned to14MiB. This
supports an exit-tombstone diagnosis, not a foreign active workload. No failed
record is changed to pass. A0 PID548489 was genuinely foreign and stays excluded.

New `guard_r1.py` remembers only live ancestry-verified process start ticks.
A missing `/proc` entry is accepted for at most2seconds only when the PID was
previously verified and NVML says `[No data]`. Live foreign/reused PIDs and
unknown tombstones remain rejected. All allowances are recorded. Locks,
quiescence, memory, timeout, and kill-only-own-child rules are unchanged.
Eight mocked identity checks pass; the next live run is a separately labeled
shape-3 repeat. GPU code admission then starts at fixture_a0. No timing change.
