"""Additive host-phase and CUDA-stream-span instrumentation, not promotion."""

from __future__ import annotations

import os
import resource
import time
from pathlib import Path

import torch


def usage():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "cpu_seconds": r.ru_utime + r.ru_stime,
        "minor_faults": r.ru_minflt,
        "major_faults": r.ru_majflt,
        "voluntary_switches": r.ru_nvcsw,
        "involuntary_switches": r.ru_nivcsw,
    }


def host_state():
    return {
        "affinity": sorted(os.sched_getaffinity(0)),
        "loadavg": list(os.getloadavg()),
        "proc_status": Path('/proc/self/status').read_text(),
        "numa_maps": Path('/proc/self/numa_maps').read_text(),
        "thread_env": {k: os.environ.get(k) for k in (
            'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'CUDA_VISIBLE_DEVICES', 'G14_PLACEMENT')},
    }


class Diagnostics:
    def __init__(self):
        self.active = False
        self.events = {}
        self.segments = []
        self.launched = []
        self.indices = {}
        self.batch = -1

    def prepare(self, batches=108):
        for stage in ('stage1', 'stage2', 'stage3'):
            self.events[stage] = [
                (torch.cuda.Event(enable_timing=True),
                 torch.cuda.Event(enable_timing=True)) for _ in range(batches)
            ]
            self.indices[stage] = 0
            # Force lazy event creation outside the public denominator.
            for start, stop in self.events[stage]:
                start.record()
                stop.record()
        torch.cuda.synchronize()
        self.host_before = host_state()
        self.usage_before = usage()

    def start(self, public_started):
        self.last = public_started
        self.last_usage = self.usage_before
        self.active = True

    def mark(self, phase):
        now = time.perf_counter()
        current = usage()
        self.segments.append({
            "phase": phase, "batch": self.batch,
            "wall_seconds": now - self.last,
            **{k: current[k] - self.last_usage[k] for k in current},
        })
        self.last, self.last_usage = now, current

    def finish(self, public_stopped):
        self.active = False
        current = usage()
        self.segments.append({
            "phase": "canonicalize", "batch": -1,
            "wall_seconds": public_stopped - self.last,
            **{k: current[k] - self.last_usage[k] for k in current},
        })
        spans = []
        for stage, index, batch in self.launched:
            start, stop = self.events[stage][index]
            spans.append({"stage": stage, "batch": batch,
                          "cuda_stream_span_ms": start.elapsed_time(stop)})
        totals = {}
        for segment in self.segments:
            target = totals.setdefault(segment['phase'], {})
            for key, value in segment.items():
                if key not in ('phase', 'batch'):
                    target[key] = target.get(key, 0) + value
        return {
            "diagnostic_only": True,
            "event_scope": "stream span around host dispatch; not isolated kernel duration",
            "segments": self.segments, "phase_totals": totals,
            "cuda_stream_spans": spans,
            "phase_wall_sum_seconds": sum(s['wall_seconds'] for s in self.segments),
            "host_before": self.host_before, "host_after": host_state(),
            "usage_before": self.usage_before, "usage_after": current,
        }


class KernelProxy:
    def __init__(self, kernel, stage, diagnostic):
        self.kernel, self.stage, self.diag = kernel, stage, diagnostic

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def invoke(*args, **kwargs):
            if not self.diag.active:
                return launch(*args, **kwargs)
            index = self.diag.indices[self.stage]
            start, stop = self.diag.events[self.stage][index]
            self.diag.indices[self.stage] += 1
            start.record()
            result = launch(*args, **kwargs)
            stop.record()
            self.diag.launched.append((self.stage, index, self.diag.batch))
            return result

        return invoke
