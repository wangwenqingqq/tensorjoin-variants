# Rejected G4A formal campaign: GPU0 became occupied by SGLang

- Campaign start: 2026-09-03T04:56:57Z.
- Process 0 KC attempt 0 was prelaunch-blocked by a foreign dense benchmark.
- Process 0 KC attempt 1 and process 1 CK attempt 0 completed cleanly.
- Process 2 KC attempt 0 was invalidated when a foreign sglang scheduler
  acquired GPU0; the isolation monitor terminated only the TensorJoin process.
- The local driver was then interrupted before attempt 1 to preserve retries and
  avoid competing with the external server.
- Decision: exclude the partial campaign. Move G4A-R1 to idle physical GPU1,
  re-freeze the device identity, and rerun screen plus formal from scratch.
- No foreign process was killed or modified.
