# G14 claim-evidence supplement

Last verified 2026-09-05 on `gpu-host-8` GPU 2. This supplements, and does not
rewrite, original claim C35's failed G5 P4 gate.

| ID | Claim | State | Contract/evidence | Counterevidence and allowed wording |
|---|---|---|---|---|
| G14-C1 | Four instrumented G5 processes emit the exact frozen 60K output in 0.911–0.951 s, versus freshly measured MiSTIC in 4.742–5.482 s. | measured | `PROTOCOL.md`; `results/campaign.json`, `results/summary.json`; full pageable-FP32-host to sorted-uint64-host scope; all 12 guards/oracles pass; G5 P4 cubin/PTX and MiSTIC binary identities retained. | Only a small instrumented diagnostic, not formal or sustained production speedup; MiSTIC is not claimed strongest. Say precisely “four diagnostic observations,” not “the old failure is fixed.” |
| G14-C2 | Preparation, combined layout/H2D/allocation and canonicalization account for 94.12–94.54% of the measured G5 denominator. | measured | Per-batch wall/CPU records in child JSONs; phase sums exactly close; `OBSERVATIONS.md`. | H2D bucket includes transpose. CUDA spans include dispatch gaps. Do not call the combined bucket pure transfer or the event spans isolated kernel durations. |
| G14-C3 | NUMA explains or fixes historical G5 variance. | inconclusive | Two processes per placement; historical slow mode did not recur; no CPU exclusivity; changing shared load. | No causal or long-tail claim. The old 5.285 s observation and failed P4 gate remain. |
| G14-C4 | Data-path optimization and a strongest adaptive-FP32-first comparison will preserve a useful advantage. | unknown | No such variant/control executed in G14. | A bounded next experiment, not an observed result or current baseline win. |
| G14-C5 | The static-join bundle is sufficient for a novel database/systems paper. | unknown | No novelty test was reopened here; G10 prior-art overlap remains. | Existing positive engineering evidence survives, but packaging or preprocessing optimization alone does not establish non-incremental novelty. |

Implementation status: unchanged core plus verified additive instrumentation.
Mechanism status: positive within the measured diagnostic scope.
Thesis impact: retain the static-join asset and target its actual costs; no paper
promotion and no new replacement direction inferred from this campaign.
