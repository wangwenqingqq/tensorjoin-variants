# Gate1 A1 preflight repair, no numeric or timing observation changed

The directory initializer expected an older `delivery_manifest.json`; the
actual sealed FP32 screen uses `delivery.json`. Initialization failed before
the protocol, schedule and prerequisite files existed. A mistakenly continued
shell then launched g1_p00_a0, which exited at missing prerequisites before
creating a CUDA operator or taking any timing sample. Its failed guard/result
are retained. Correct the prerequisite filename and freeze schedule_a1 before
observations. Numeric code and timing runner are unchanged. This is preflight
repair, not replacement of a slow or incorrect performance observation.
