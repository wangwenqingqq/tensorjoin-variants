# A1: new idle-window admission, not a replacement measurement

A0 was rejected during quiescence: foreign PID548489 on GPU2. The guard's
child PID is null; no NCU or numerical test executed. Keep its failed result
and raw occupancy log unchanged. No process was signaled by this task.

The2026-09-07 15:51:56UTC recheck showed GPU2 idle again; the foreign PID was
gone. Therefore do not switch GPUs or weaken isolation. One new A1 admission
window uses the identical trace code, target UUID, library, shapes and30-second
quiescence. Run the four planned shapes with new `_a1` labels. Stop the sequence
on any failure. A0 remains a failed admission and contributes no observation;
A1 is not a favorable replacement of any numeric/performance sample.

No timing or performance estimator exists in this stage. The locally prepared
GPU kernel draft has not been transferred into the trace source inventory.
