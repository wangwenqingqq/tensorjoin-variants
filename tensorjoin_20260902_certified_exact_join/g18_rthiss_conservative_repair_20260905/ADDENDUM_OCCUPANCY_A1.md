# A1: resume after foreign preflight occupancy, no scientific rerun

Declared after the first campaign stopped, before continuation. GPU 2 received
foreign Physion PID1788163 (550 MiB) during the preflight quiescence for signed31.
The unchanged G5 guard raised before launching the child. There is no signed31
result or guard-completion receipt for this attempt. No foreign process was
signalled, and no partial numerical result is reclassified.

The first two slots (8192-pair A/B and full real31) passed with clean isolation;
retain them without rerunning. Preserve `campaign.json` and `raw/campaign.log`
exactly, including their failed occupancy slot. Resume on the same physical
GPU 2 only after fresh live preflight and the unchanged 30-second quiescence.
The previously blocked signed31 slot gets a new `_a1` record ID. All remaining
input/order/threshold/source/build/correctness/safety/stress/profile gates stay
unchanged. Stop again on any failed guard; do not kill, migrate or reinterpret
foreign work. GPU0/1 jobs and newly observed GPU7 sglang PID1780362 are untouched.

`campaign_resume_a1.json` is a separate receipt: the first two retained passes
plus the remaining predeclared slots. `frozen_resume_a1.json` binds this addendum,
old campaign/log, continuation source and slot list before launch. Closure must
report 20 successful planned observations plus this separate prelaunch block,
not pretend that every original attempt passed.
