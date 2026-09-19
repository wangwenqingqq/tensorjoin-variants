# G14 diagnostic observations

Instrumented observations only. No confidence interval, novelty pass,
production speedup claim, or historical result replacement.

| Block | Placement | Position | Method | Public s | Admitted |
|---|---|---|---|---:|---|
| 0 | default | 0 | g2b | 1.093086 | True |
| 0 | default | 1 | mistic | 5.021749 | True |
| 0 | default | 2 | g5 | 0.933772 | True |
| 1 | node0 | 0 | g2b | 0.927863 | True |
| 1 | node0 | 1 | mistic | 5.482189 | True |
| 1 | node0 | 2 | g5 | 0.932263 | True |
| 2 | node0 | 0 | g5 | 0.911490 | True |
| 2 | node0 | 1 | mistic | 5.283864 | True |
| 2 | node0 | 2 | g2b | 0.921459 | True |
| 3 | default | 0 | g5 | 0.950725 | True |
| 3 | default | 1 | mistic | 4.742475 | True |
| 3 | default | 2 | g2b | 0.934260 | True |

## Host-wall phase attribution (milliseconds)

| Block | Method | Preprocess | Schedule | H2D/alloc | S1+count | S2+count | S3+count | Accepted D2H | Canonicalize | Other |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | g2b | 683.640 | 6.440 | 241.103 | 23.762 | 8.671 | 7.645 | 10.792 | 106.993 | 4.040 |
| 0 | g5 | 546.378 | 4.942 | 235.098 | 18.316 | 8.744 | 6.789 | 11.428 | 97.407 | 4.671 |
| 1 | g2b | 556.765 | 4.904 | 222.935 | 18.155 | 8.179 | 7.148 | 10.298 | 96.183 | 3.297 |
| 1 | g5 | 554.615 | 4.612 | 230.282 | 17.750 | 8.162 | 6.252 | 10.474 | 96.453 | 3.664 |
| 2 | g5 | 546.560 | 4.790 | 216.768 | 17.574 | 8.259 | 6.389 | 10.478 | 96.940 | 3.732 |
| 2 | g2b | 554.364 | 4.889 | 218.353 | 18.360 | 8.266 | 7.458 | 10.154 | 96.307 | 3.309 |
| 3 | g5 | 563.696 | 4.496 | 234.892 | 18.291 | 8.803 | 6.975 | 11.237 | 97.891 | 4.444 |
| 3 | g2b | 554.427 | 5.178 | 225.698 | 19.273 | 8.802 | 7.771 | 11.226 | 97.754 | 4.132 |

Full per-batch wall/CPU/fault/switch observations, stream-span events,
process orders, distributions and paired estimators are in `results/summary.json`
and the immutable child JSON records. Stream spans are not isolated kernel times.
