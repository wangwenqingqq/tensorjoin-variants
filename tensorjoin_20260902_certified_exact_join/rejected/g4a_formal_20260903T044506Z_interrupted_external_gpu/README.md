# Rejected G4A formal campaign: external GPU interference

- Campaign start: 2026-09-03T04:45:06Z
- Clean completed slots: process 0 KC, process 1 CK, process 2 KC, process 3 CK.
- Rejected slot: process 4 CK attempts 0, 1, and 2.
- Reason: physical GPU0 was acquired by foreign dense_n8 / sparse_n8
  workloads. The isolation runner blocked before launching TensorJoin.
- Decision: do not include this partial campaign in the formal estimator. Preserve
  every artifact and restart the complete eight-process formal campaign only
  after a fresh 30-second empty-GPU admission.
- No foreign process was killed or modified.
