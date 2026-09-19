"""Untimed FP16 repair census against retained kernels and exhaustive terminal."""
import argparse
import ctypes as C
from fractions import Fraction as F
import hashlib
import json
import os
import platform
import time
import traceback
import numpy as np
import torch
from operators import Operators, full_source, N, D, CAP, EPS, T, TAU, BASE
from frozen_driver import HERE, ROOT, sha, verify


def digest(a):
    return hashlib.sha256(np.asarray(a, dtype='<i8').tobytes()).hexdigest()


class Probe(Operators):
    def __init__(self):
        super().__init__()
        self.blas._check(self.blas.library.cublasSetMathMode(self.blas.handle, 16), 'FP16 math mode')
        mode = C.c_int()
        self.blas._check(self.blas.library.cublasGetMathMode(self.blas.handle, C.byref(mode)), 'query mode')
        assert mode.value == 16
        self.fp16_record = dict(self.library_record, math_mode=16, required_math_mode=16,
                               compute_type=68, input_type=2, output_type=0,
                               arithmetic_model='conditional FP32 accumulation envelope; not a selected-function proof',
                               handle_ownership='borrowed Torch handle; no full-shutdown sanitizer claim')

    def half_dot(self, x, y):
        assert x.dtype == y.dtype == torch.float16 and x.is_contiguous() and y.is_contiguous()
        m, k = x.shape
        n = y.shape[0]
        out = torch.empty((m, n), device='cuda', dtype=torch.float32)
        alpha, beta = C.c_float(1), C.c_float(0)
        assert torch.cuda.current_stream().cuda_stream == self.stream == 0
        self.blas._check(self.blas.library.cublasGemmEx(
            self.blas.handle, 1, 0, n, m, k, C.byref(alpha),
            C.c_void_p(y.data_ptr()), 2, k, C.c_void_p(x.data_ptr()), 2, k,
            C.byref(beta), C.c_void_p(out.data_ptr()), 0, n, 68, -1), 'FP16 GemmEx')
        return out.cpu().numpy()

    def reference(self, v, ids):
        assert len(ids) <= CAP
        pairs = torch.from_numpy(ids).to('cuda')
        out = torch.empty(max(1, len(ids)), device='cuda', dtype=torch.int64)
        c = torch.zeros(6, device='cuda', dtype=torch.int32)
        self.terminal(v, pairs, out, c, len(ids))
        n = int(c[0].item())
        assert c[2].item() == 0
        return np.sort(out[:n].cpu().numpy().copy())

    def refine(self, v, ambiguous):
        ids = torch.from_numpy(ambiguous).to('cuda')
        out = torch.empty(max(1, len(ambiguous)), device='cuda', dtype=torch.int64)
        uncertain = torch.empty_like(out)
        c = torch.zeros(6, device='cuda', dtype=torch.int32)
        self.launch('stage2', len(ambiguous), [v, ids, out, uncertain, c],
                    [T, T, 2**-14, 2**-22, 2**-22, 4096*TAU])
        mid = c.cpu().numpy().copy()
        self.terminal(v, uncertain, out, c, int(mid[3]))
        last = c.cpu().numpy().copy()
        assert last[2] == 0 and last[0] <= len(ambiguous)
        return out[:int(last[0])].cpu().numpy().copy(), {
            'stage2_accept': int(mid[0]), 'stage2_reject': int(mid[4]),
            'fp64_inputs': int(mid[3]), 'terminal_accept': int(last[0]-mid[0])}

    def first_a(self, v, prepared, row, col, rows, cols):
        q, qt, scales, norms, errors = prepared
        rr, cc = np.meshgrid(np.arange(row//64, (row+rows+63)//64, dtype=np.int32),
                             np.arange(col//64, (col+cols+63)//64, dtype=np.int32), indexing='ij')
        tr, tc = rr[rr <= cc].copy(), cc[rr <= cc].copy()
        assert len(tr) <= 4096
        trg, tcg = torch.from_numpy(tr).to('cuda'), torch.from_numpy(tc).to('cuda')
        out = torch.empty(CAP, device='cuda', dtype=torch.int64)
        amb = torch.empty_like(out)
        c = torch.zeros(6, device='cuda', dtype=torch.int32)
        self.launch('stage1', len(tr), [q, qt, scales, norms, errors, trg, tcg, out, amb, c],
                    [EPS, EPS, 2**-16, 16*TAU, 2**-20, 16*TAU])
        st = c.cpu().numpy().copy()
        assert st[2] == 0 and max(st[0], st[1]) <= CAP
        return out[:int(st[0])].cpu().numpy().copy(), amb[:int(st[1])].cpu().numpy().copy()


def fixture():
    rng = np.random.default_rng(202609071)
    x = np.zeros((N, D), np.float32)
    x[:512] = rng.uniform(-.06, .06, (512, D)).astype(np.float32)
    x[:32] = 0
    x[1, 0] = EPS
    x[2] = x[1]
    x[2, 1:4] = 2**-28
    x[3, 0] = np.nextafter(np.float32(EPS), np.float32(0))
    x[4, 0] = np.nextafter(np.float32(EPS), np.float32(np.inf))
    x[5] = np.where(np.arange(D) % 2, 1, -1)
    x[6] = 1
    x[7] = -1
    x[8] = np.nextafter(np.float32(0), np.float32(1))
    x[9] = 2**-24
    x[10] = 2**-14
    x[11] = np.nextafter(np.float32(2**-14), np.float32(0))
    x[12] = 1 + 2**-11
    x[12] *= .5
    x[13] = np.nextafter(x[12], np.float32(0))
    x[14] = np.nextafter(x[12], np.float32(1))
    x[15] = -x[14]
    x[16] = x[0]
    return x


def rational_checks(x, z, residual, norms, rows):
    records = []
    for i in rows:
        xx = [F(float(v)) for v in x[i]]
        zz = [F(float(v)) for v in z[i]]
        rr = [F(float(v)) for v in residual[i]]
        assert all(a-b == c for a, b, c in zip(xx, zz, rr)), ('nonexact residual', i)
        for name, values in [('x', xx), ('z', zz), ('r', rr)]:
            exact = sum(v*v for v in values)
            center, radius, upper = [F(float(a[i])) for a in norms[name]]
            assert center-radius <= exact <= center+radius and upper*upper >= exact, (name, i)
        records.append({'row': int(i), 'pass': True})
    return records


def first_c(dots, norms, row, col):
    rows, cols = dots.shape
    nx, nr, _ = norms['x']
    lz, er = norms['z'][2], norms['r'][2]
    gamma = ((2*D+2)*2**-23) / (1-(2*D+2)*2**-23)
    accepted, ambiguous, allids = [], [], []
    widths = []
    for start in range(0, rows, 128):
        i = np.arange(row+start, row+min(start+128, rows))[:, None]
        j = np.arange(col, col+cols)[None, :]
        p = dots[start:start+len(i)].astype(np.float64)
        b = gamma*lz[i]*lz[j] + er[i]*lz[j] + lz[i]*er[j] + er[i]*er[j] + 4*D*TAU
        c = nx[i]+nx[j]-2*p
        magnitude = abs(nx[i])+abs(nx[j])+2*abs(p)+nr[i]+nr[j]+2*b
        radius = nr[i]+nr[j]+2*b+32*2**-52*magnitude+1e-12+2**-38*(abs(nx[i])+abs(nx[j])+nr[i]+nr[j])
        lower, upper = np.maximum(c-radius, 0), c+radius
        assert np.isfinite(lower).all() and np.isfinite(upper).all()
        valid = i <= j
        accept, reject = (upper <= T) & valid, (lower > T) & valid
        amb = valid & ~(accept | reject)
        ids = i*N+j
        accepted.append(ids[accept])
        ambiguous.append(ids[amb])
        allids.append(ids[valid])
        widths.append([float(radius[valid].min()), float(radius[valid].max())])
    return tuple(np.concatenate(v).astype(np.int64, copy=False) for v in [accepted, ambiguous, allids]), widths


def process_data(o, x, panels, tag, report, held):
    v = torch.from_numpy(x).to('cuda')
    assert v.data_ptr() not in [t.data_ptr() for t in held]
    held.append(v)
    half = v.to(torch.float16)
    half = torch.where(half.abs() < 2**-14, torch.zeros_like(half), half)
    z = half.to(torch.float32)
    residual = v-z
    md = {name: [t.cpu().numpy().copy() for t in o.norm(value)]
          for name, value in [('x', v), ('z', z), ('r', residual)]}
    selected = list(range(32)) + [255, 256, 511, 57344, 59999]
    zcpu, rcpu = z.cpu().numpy(), residual.cpu().numpy()
    checks = rational_checks(x, zcpu, rcpu, md, selected)
    np.savez_compressed(HERE/'results'/f'{tag}_metadata.npz', rows=selected,
                        x=x[selected], z=zcpu[selected], r=rcpu[selected],
                        **{k+'_'+str(i): a[selected] for k, arrays in md.items() for i, a in enumerate(arrays)})
    del zcpu, rcpu, z, residual
    q = torch.empty((N, D), device='cuda', dtype=torch.int8)
    s = torch.empty(N, device='cuda'); h = torch.empty_like(s); e = torch.empty_like(s)
    bad = torch.zeros(1, device='cuda', dtype=torch.int32)
    o.launch('metadata', N, [v, q, s, h, e, bad])
    assert bad.item() == 0
    prepared = q, q.T.contiguous(), s, h, e
    data = {'tag': tag, 'input_sha256': hashlib.sha256(x.tobytes()).hexdigest(),
            'input_pointer': v.data_ptr(), 'rational_metadata_checks': checks, 'panels': []}
    report['datasets'].append(data)
    for index, (row, col, rows, cols) in enumerate(panels):
        dots = o.half_dot(half[row:row+rows], half[col:col+cols])
        (ca, cu, ids), widths = first_c(dots, md, row, col)
        truth = o.reference(v, ids)
        aa, au = o.first_a(v, prepared, row, col, rows, cols)
        result = {'offset': [row, col], 'shape': [rows, cols], 'pairs': len(ids),
                  'reference_count': len(truth), 'reference_hash': digest(truth), 'pass': False,
                  'fp16_radius_min': min(t[0] for t in widths), 'fp16_radius_max': max(t[1] for t in widths)}
        data['panels'].append(result)
        arrays = {'reference': truth, 'C_direct': ca, 'C_uncertain': cu, 'A_direct': aa, 'A_uncertain': au}
        for method, direct, uncertain in [('A', aa, au), ('C', ca, cu)]:
            refinements, counts = o.refine(v, uncertain)
            final = np.sort(np.concatenate([direct, refinements]))
            false_accept = np.setdiff1d(direct, truth, assume_unique=False)
            false_reject = np.setdiff1d(truth, np.concatenate([direct, uncertain]), assume_unique=False)
            result[method] = {'direct_accept': len(direct), 'uncertain': len(uncertain),
                              'direct_reject': len(ids)-len(direct)-len(uncertain), **counts,
                              'count': len(final), 'hash': digest(final),
                              'false_direct_accept': len(false_accept), 'false_direct_reject': len(false_reject),
                              'exact_match': bool(np.array_equal(final, truth))}
            arrays[method+'_final'] = final
        np.savez_compressed(HERE/'results'/f'{tag}_panel{index}_ids.npz', **arrays)
        result['pass'] = all(result[m]['exact_match'] and result[m]['false_direct_accept'] == result[m]['false_direct_reject'] == 0 for m in ['A', 'C'])
        print(json.dumps(result), flush=True)
        assert result['pass'], ('panel correctness', tag, index)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=['fixture', 'public'], required=True); p.add_argument('--label', required=True)
    args = p.parse_args(); output = HERE/'results'/f'{args.label}.json'; assert not output.exists()
    r = {'label': args.label, 'mode': args.mode, 'pid': os.getpid(), 'host': platform.node(),
         'started': time.time(), 'pass': False, 'datasets': [], 'new_performance_claim': False,
         'novelty_pass': False, 'certificate_status': 'conditional diagnostic, not library-level proof'}
    try:
        assert platform.node() == 'gpu-host-8' and os.environ['CUDA_VISIBLE_DEVICES'] == '2'
        o = Probe(); r.update(identities=o.identities, abi=o.abi, fp16_api=o.fp16_record,
                              library_hashes=o.library_hashes, torch=torch.__version__, cuda=torch.version.cuda)
        held = []
        if args.mode == 'fixture':
            x = fixture(); np.save(HERE/'results'/f'{args.label}_fixture512.npy', x[:512])
            for i in range(2): process_data(o, x, [(0, 0, 512, 512)], f'{args.label}_churn{i}', r, held)
        else:
            gate = json.loads((HERE/'results/fixture_a0.json').read_text()); assert gate['pass']
            process_data(o, full_source(), [(0, 0, 4096, 4096), (0, 57344, 4096, 2656), (57344, 57344, 2656, 2656)], args.label, r, held)
        after = verify()
        for spec in BASE.values():
            for ext, h in spec['hashes'].items():
                key = spec['stem']+ext; assert sha(ROOT/key) == h; after[key] = h
        assert after == o.identities
        r['identities_after'] = after; r['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        r['pass'] = True
    except Exception:
        r['exception'] = traceback.format_exc(); print(r['exception'], flush=True)
    finally:
        r['ended'] = time.time()
        with output.open('x') as f: json.dump(r, f, indent=2)
    print(json.dumps({'pass': r['pass'], 'result': str(output)}), flush=True)
    return 0 if r['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
