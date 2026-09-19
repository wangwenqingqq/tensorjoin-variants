# Rejected G4A formal campaign: second external GPU interference

- Campaign start: 2026-09-03T04:50:12Z
- Clean completed slots: process 0 KC.
- Rejected slot: process 1 CK attempts 0, 1, and 2.
- Reason: physical GPU0 was acquired by foreign dense_n8_bm64 /
  sparse_n8_bm64 workloads. TensorJoin was not launched in rejected attempts.
- Decision: exclude this partial campaign. Preserve all artifacts and require a
  longer external empty-GPU observation before restarting the full campaign.
- No foreign process was killed or modified.
