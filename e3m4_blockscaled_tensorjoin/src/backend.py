"""Minimal block32 extension of the retained TensorJoin research engine."""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import torch
import triton
import experiment as base
from block_kernels import prepare_block32, gram_block32

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts'
CUOBJDUMP = os.environ.get('CUOBJDUMP', '/usr/local/cuda-13.1/bin/cuobjdump')
FIELDS = ((78, 82, 83), (79, 84, 85))


def inspect_scaled(binary):
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / 'k.cubin'; p.write_bytes(binary)
        sections = subprocess.check_output(['readelf', '-SW', str(p)], text=True)
        sass = subprocess.check_output([CUOBJDUMP, '-sass', str(p)], text=True)
    sec = re.findall(r'\.text\.\S+\s+PROGBITS\s+[\da-fA-F]+\s+([\da-fA-F]+)', sections)
    assert len(sec) == 1, sec
    start = int(sec[0], 16)
    rows = []
    for line in sass.splitlines():
        if re.search(r'\bQMMA\.', line):
            assert 'QMMA.SF.16832.F32.' in line and '.E8' in line, line
            addr = int(re.search(r'/\*([\da-fA-F]+)\*/', line)[1], 16)
            word = int.from_bytes(binary[start+addr:start+addr+16], 'little')
            fields = [sum(((word >> bit) & 1) << i for i, bit in enumerate(axis)) for axis in FIELDS]
            rows.append({'address': addr, 'file_offset': start+addr, 'word': hex(word), 'fields': fields})
    assert rows and not re.search(r'\b[HI]MMA\.', sass), 'Native scaled QMMA required; no emulation'
    return rows, sass


def patch_scaled(kernel):
    assert kernel.metadata.target.arch == 120
    binary = kernel.kernel
    rows, sass = inspect_scaled(binary)
    buf = bytearray(binary)
    delta = (1 << 82) | (1 << 84)
    changed = set()
    for row in rows:
        assert row['fields'] == [0, 0], row
        off = row['file_offset']; word = int(row['word'], 16)
        buf[off:off+16] = (word ^ delta).to_bytes(16, 'little')
        assert int.from_bytes(buf[off:off+16], 'little') ^ word == delta
        changed.update((off + 82//8, off + 84//8))
    assert {i for i, (a, b) in enumerate(zip(binary, buf)) if a != b} == changed
    assert len(binary) == len(buf)
    clone = copy.copy(kernel); clone.asm = dict(kernel.asm); clone.kernel = bytes(buf)
    clone.asm['cubin'] = bytes(buf); clone.module = None; clone.function = None; clone._run = None
    ART.mkdir(exist_ok=True)
    key = hashlib.sha256(binary).hexdigest()
    (ART / (key + '.baseline.cubin')).write_bytes(binary)
    (ART / (key + '.e3m4.cubin')).write_bytes(buf)
    (ART / (key + '.baseline.sass')).write_text(sass)
    (ART / (key + '.patch.json')).write_text(json.dumps({
        'baseline_sha256': key, 'candidate_sha256': hashlib.sha256(buf).hexdigest(),
        'instruction_count': len(rows), 'instructions': rows,
        'allowed_xor_bits': [82, 84], 'changed_bytes': len(changed), 'family': 'QMMA.SF.16832.F32.E8'}, indent=2))
    return clone


class Engine(base.Engine):
    def __init__(self):
        super().__init__()
        self.scaled = {}; self.registry = {}

    def prepare(self, v, fmt):
        if not fmt.startswith('mx_'): return super().prepare(v, fmt)
        assert fmt in ('mx_e4m3', 'mx_e3m4') and v.shape[1] == 512 and v.is_contiguous()
        n = len(v); q = torch.empty_like(v, dtype=torch.uint8)
        s = torch.empty((n, 16), device='cuda', dtype=torch.uint8)
        md = [torch.empty(n, device='cuda') for _ in range(4)]
        invalid = torch.zeros(1, device='cuda', dtype=torch.int32)
        k = prepare_block32[(n,)](v, q, s, *md, invalid, FORMAT=fmt[3:], num_warps=4, enable_fp_fusion=False)
        self.kernels['prepare_' + fmt] = k
        assert invalid.item() == 0
        return q, s, md

    def dots(self, q, s, fmt, fmt_b=None):
        if not fmt.startswith('mx_'): return super().dots(q, s, fmt, fmt_b)
        n, d = q.shape
        assert d % 64 == 0 and q.is_contiguous() and s.shape == (n, d//32) and s.is_contiguous()
        fmt_b = fmt if fmt_b is None else fmt_b
        a, b = fmt[3:], fmt_b[3:]
        assert a in ('e4m3', 'e3m4', 'e5m2') and b in ('e4m3', 'e5m2', 'e3m4')
        assert ('e3m4' not in (a, b)) or a == b == 'e3m4'
        fa = 'e4m3' if a == 'e3m4' else a
        fb = 'e4m3' if b == 'e3m4' else b
        out = torch.empty((n, n), device='cuda')
        grid = (triton.cdiv(n, 32), triton.cdiv(n, 32), 1)
        args = (q, s, out, n, d, fa, fb, 32, 32, 64)
        key = (n, d, a, b)
        if key not in self.scaled:
            compiled = gram_block32.warmup(*args, grid=grid, num_warps=4, num_stages=2, enable_fp_fusion=False)
            inspect_scaled(compiled.kernel)
            self.scaled[key] = patch_scaled(compiled) if a == 'e3m4' else compiled
        kernel = self.scaled[key]; kernel[grid](*args)
        self.kernels[f'dot_{fmt}_{fmt_b}_{n}_{d}'] = kernel
        return out

    def remember(self):
        # Registry is outside the measured run. Keep every observed Triton
        # specialization, not merely the last shape/threshold per short name.
        for name, k in self.kernels.items():
            sha = hashlib.sha256(k.kernel).hexdigest(); key = name + ':' + sha
            if key not in self.registry:
                k._init_handles()
                self.registry[key] = {'name': name, 'cubin_sha256': sha, 'registers': k.n_regs,
                    'spills': k.n_spills, 'shared_bytes': k.metadata.shared,
                    'constants': {str(a): str(b) for a, b in k.src.constants.items()},
                    'target': str(k.metadata.target)}

    def run(self, *args, **kwargs):
        ids, result = super().run(*args, **kwargs)
        self.remember()
        return ids, result

    def capture(self, path):
        self.remember(); path.write_text(json.dumps(self.registry, indent=2))
