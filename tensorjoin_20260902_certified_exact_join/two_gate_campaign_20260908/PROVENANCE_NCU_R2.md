# NCU source access-size unit calibration

r1 matched all six runtime-selected instruction bodies, then failed the
predeclared 64MiB split-score-store assertion by exactly 8x. The source Access
Size field is in bits, not bytes. The retained calibration shows eight
STG.E.128 instructions, each with 524,288 predicated-on thread executions and
Access Size 128. Their logical store total is
8 * 524288 * (128 / 8) = 67,108,864 bytes, exactly the 4096x4096 I32/FP32 buffer.

r2 only corrects this derived field's bits-to-bytes conversion. It does not
change kernels, raw metrics, source normalization, samples or timing. Both
failed parser attempts are preserved. Global-address-space logical traffic
is not asserted to equal DRAM traffic; cache residency remains relevant.
