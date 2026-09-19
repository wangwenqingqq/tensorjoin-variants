"""Retained arithmetic plus CPU-certified tile lists and integer ID remapping."""
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import triton
import triton.language as tl
from geometry import build, digest, retained_mask

HERE = Path(__file__).resolve().parents[1]
ROOT = Path('@TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join')
SHARED = ROOT/'tensorjoin_20260908_output_audit'
sys.path.insert(0, str(SHARED/'src'))
import common as output

N, D, CAP = output.N, output.D, output.CAP
CELLS = output.CELLS
METHODS = ['F8', 'F16']
MODES = ['original_full', 'layout_full', 'geometric']

@triton.jit
def remap_ids(IDS, PERM, OUT, COUNT, NN: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0)*B + tl.arange(0, B)
    a = tl.load(IDS+i, i < COUNT, 0)
    r = a // NN
    c = a % NN
    rr = tl.load(PERM+r, i < COUNT, 0)
    cc = tl.load(PERM+c, i < COUNT, 0)
    result = tl.minimum(rr,cc)*NN + tl.maximum(rr,cc)
    tl.store(OUT+i, result, i < COUNT)

class Sink(output.OutputSink):
    def __init__(self, radix, permutation=None, diagnostic=False):
        super().__init__('gpu_radix', radix, diagnostic)
        self.permutation = permutation
        self.remap_kernel = None
    def push(self, out, n):
        if self.permutation is None:
            return super().push(out, n)
        start = time.perf_counter() if self.diagnostic else 0.
        part = torch.empty(n, device='cuda', dtype=torch.int64)
        if n:
            self.remap_kernel = remap_ids[(triton.cdiv(n,256),)](
                out, self.permutation, part, n, NN=N, B=256, num_warps=4)
        self.parts.append(part)
        self.gpu_copy_bytes += 8*n
        if self.diagnostic:
            self.collect_seconds += time.perf_counter()-start

def plan(x):
    selection = json.loads((HERE/'artifacts/selection.json').read_text())
    method = selection['selected']
    prefix = HERE/'artifacts'/f'cifar60000_{method}'
    perm = np.load(str(prefix)+'_permutation.npy')
    b = dict(np.load(str(prefix)+'_bounds.npz'))
    ideal = dict(np.load(str(prefix)+'_ideal.npz'))
    y = np.ascontiguousarray(x[perm])
    return dict(layout=method, perm=perm, y=y, bounds=b, ideal=ideal)

def execute(op, radix, x, cell, method, mode, prepared, cold=False, diagnostic=False):
    assert mode in MODES+['ideal_diagnostic']
    op.set_cell(cell)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    build_record = None
    if cold:
        assert mode == 'geometric'
        perm, y, b, build_record = build(x, prepared['layout'])
    else:
        perm, y, b = prepared['perm'], prepared['y'], prepared['bounds']
    built = time.perf_counter()
    tr, tc = np.triu_indices((N+63)//64)
    tr, tc = tr.astype(np.int32), tc.astype(np.int32)
    mask = None
    if mode == 'geometric':
        mask = retained_mask(b, cell['T'])
    elif mode == 'ideal_diagnostic':
        mask = prepared['ideal'][cell['name']]
    if mask is not None:
        tr, tc = tr[mask], tc[mask]
    listed = time.perf_counter()
    original = mode == 'original_full'
    v = torch.from_numpy(x if original else y).to('cuda')
    prep = op.prepare(v, 8 if method == 'F8' else 16)
    pg = None if original else torch.from_numpy(perm).to('cuda')
    sink = Sink(radix, pg, diagnostic)
    trg, tcg = torch.from_numpy(tr).to('cuda'), torch.from_numpy(tc).to('cuda')
    out = torch.empty(CAP,device='cuda',dtype=torch.int64)
    amb, fp64 = torch.empty_like(out), torch.empty_like(out)
    score = torch.empty(1,device='cuda',dtype=torch.int32 if method=='F8' else torch.float32)
    dummy = score.view(torch.float32)
    records, events = [], []
    prepared_at = time.perf_counter()
    for off in range(0,len(tr),4096):
        c = torch.zeros(6,device='cuda',dtype=torch.int32)
        if diagnostic:
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
            ev[0].record()
        op.stage1(method,prep,trg[off:off+4096],tcg[off:off+4096],out,amb,c,score,dummy,dummy)
        if diagnostic: ev[1].record()
        s1 = c.cpu().numpy().copy()
        n1 = int(s1[1])
        assert s1[2] == 0 and 0 <= n1 <= CAP
        if diagnostic: ev[2].record()
        op.stage2(v,amb,out,fp64,c,n1)
        if diagnostic: ev[3].record()
        s2 = c.cpu().numpy().copy()
        n2 = int(s2[3])
        assert s2[2] == 0 and 0 <= n2 <= CAP
        if diagnostic: ev[4].record()
        op.terminal(v,fp64,out,c,n2)
        if diagnostic:
            ev[5].record()
            events.append(ev)
        last = c.cpu().numpy().copy()
        n = int(last[0])
        assert last[2] == 0 and 0 <= n <= CAP
        sink.push(out,n)
        records.append([int(s1[0]),n1,int(s2[0]-s1[0]),int(s2[4]),n2,n])
    torch.cuda.synchronize()
    finalized_at = time.perf_counter()
    a, sr = sink.finish()
    end = time.perf_counter()
    rec = dict(seconds=end-start, build_wall_seconds=built-start,
        query_seconds=end-built, build=build_record, method=method, mode=mode,
        cell=cell['name'], cold=cold, diagnostic=diagnostic, layout=prepared['layout'],
        executed_tiles=len(tr), batches=len(records),
        stage_counts=np.sum(np.asarray(records,dtype=np.int64),axis=0).tolist(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), **sr)
    if diagnostic:
        rec.update(list_seconds=listed-built, preparation_seconds=prepared_at-listed,
            pre_finalize_seconds=finalized_at-built,
            stage_gpu_ms=np.sum([[e[i].elapsed_time(e[i+1]) for i in [0,2,4]]
                                for e in events],axis=0).tolist())
    if cold:
        assert digest(perm) == digest(prepared['perm'])
        assert np.array_equal(b['combined'],prepared['bounds']['combined'])
    kernel = sink.remap_kernel
    return a, rec, kernel

def call(op, radix, x, cell, method, mode, prepared, exact=False, cold=False, diagnostic=False):
    a, rec, kernel = execute(op,radix,x,cell,method,mode,prepared,cold,diagnostic)
    rec['output_sha256'] = output.check_output(a,cell,exact)
    del a
    return rec, kernel

def save_json(p, r):
    with Path(p).open('x') as f:
        json.dump(r,f,indent=2,default=str)

def capture_remap(kernel):
    if kernel is None:
        return None
    record = {}
    stem = 'remap_'+hashlib.sha256(kernel.asm['cubin']).hexdigest()[:16]
    for ext in ['cubin','ptx','llir','ttgir','ttir']:
        data = kernel.asm[ext]
        data = data if isinstance(data,bytes) else data.encode()
        p = HERE/'artifacts'/(stem+'.'+ext)
        h = hashlib.sha256(data).hexdigest()
        if p.exists(): assert output.sha(p) == h
        else: p.write_bytes(data)
        record[ext] = h
    assert kernel.n_spills == 0
    return record

def verify_freeze():
    for path, expected in json.loads((HERE/'artifacts/timing_freeze.json').read_text()).items():
        assert output.sha(Path(path)) == expected, path
