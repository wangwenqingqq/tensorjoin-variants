The second audit recognized only a non-saturating textual MMA form. Triton
emits mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 on sm_120a.
Every partial INT32 limb sum is bounded by D*127^2<2^24, so saturation cannot
activate on the admitted contract. Accept the documented saturating spelling,
record it explicitly, and preserve the same numerical range proof. No kernel
source or launch parameter changes; no performance measurement was started.
