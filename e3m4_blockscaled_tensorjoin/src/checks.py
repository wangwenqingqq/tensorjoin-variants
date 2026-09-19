"""Independent reconstruction, interval, native-format and execution checks."""
import numpy as np
import torch
import triton

import experiment as base
from retained import half_classify
from backend import inspect_scaled

GAMMA = 0.00012232370499987155


def reconstruction(q, s, fmt):
    qq = q.cpu().numpy(); ss = s.cpu().numpy()
    if fmt.startswith('mx_'):
        h = base.codebook(fmt[3:])[qq]
        scale = np.repeat(np.exp2(ss.astype(np.float64)-127), 32, axis=1)
    else:
        h = base.codebook(fmt)[qq] if fmt.startswith('e') else qq.astype(np.float64)
        scale = ss.astype(np.float64)[:, None]
    return h * scale


def assert_dot(got, z):
    truth = z @ z.T
    error = np.abs(got.astype(np.float64)-truth)
    norm = np.linalg.norm(z, axis=1)
    bound = GAMMA * norm[:, None] * norm[None, :] + 2048 * 2.0**-126
    assert np.all(error <= bound), ('dot envelope', float(np.max(error-bound)))
    return {'pairs': got.size, 'max_abs_error': float(error.max()),
            'max_error_over_bound': float((error/bound).max())}


def native_and_layout(engine):
    results = {'codebook': {}, 'layout': []}
    for fmt in ('mx_e4m3', 'mx_e3m4'):
        q = torch.arange(256, dtype=torch.int16, device='cuda').to(torch.uint8)[:, None].expand(256, 64).contiguous()
        s = torch.full((256, 2), 127, dtype=torch.uint8, device='cuda')
        got = engine.dots(q, s, fmt).cpu().numpy(); values = base.codebook(fmt[3:])
        expected = (values[:, None]*values[None, :]*64).astype(np.float32)
        assert np.array_equal(np.isnan(got), np.isnan(expected)), fmt
        finite = np.isfinite(expected)
        assert np.array_equal(got[finite], expected[finite]), fmt
        results['codebook'][fmt] = {'raw_codes_per_axis': 256, 'checked_pairs': got.size,
                                    'raw_0x10_k64': float(got[16, 16]), 'exact': True}
    assert results['codebook']['mx_e3m4']['raw_0x10_k64'] == 4.0
    assert results['codebook']['mx_e4m3']['raw_0x10_k64'] == .0625
    q = torch.zeros((32, 64), dtype=torch.uint8, device='cuda')
    s = torch.full((32, 2), 127, dtype=torch.uint8, device='cuda')
    controls = []
    for a, b, fields in [('mx_e4m3', 'mx_e4m3', [0, 0]),
                         ('mx_e5m2', 'mx_e4m3', [1, 0]),
                         ('mx_e4m3', 'mx_e5m2', [0, 1])]:
        engine.dots(q, s, a, b)
        rows, _ = inspect_scaled(engine.kernels[f'dot_{a}_{b}_32_64'].kernel)
        assert all(row['fields'] == fields for row in rows)
        controls.append({'a': a, 'b': b, 'fields': fields, 'instructions': len(rows)})
    results['compiler_controls'] = controls

    rng = np.random.default_rng(9192026)
    for n in (31, 32, 33, 97, 129):
        for d in (64, 512):
            qq = rng.integers(0, 127, (n, d), dtype=np.uint8)
            qq |= rng.integers(0, 2, (n, d), dtype=np.uint8)*np.uint8(128)
            ss = rng.integers(121, 130, (n, d//32), dtype=np.uint8)
            q = torch.from_numpy(qq).cuda(); s = torch.from_numpy(ss).cuda()
            for fmt in ('mx_e4m3', 'mx_e3m4'):
                z = reconstruction(q, s, fmt)
                got = engine.dots(q, s, fmt).cpu().numpy()
                rec = assert_dot(got, z)
                results['layout'].append(dict(n=n, d=d, format=fmt, **rec))
    return results


def producer_and_intervals(engine):
    rng = np.random.default_rng(20260919)
    x = rng.normal(0, .035, (97, 512)).astype(np.float32)
    x *= np.exp2(rng.integers(-16, 1, (97, 16))).repeat(32, axis=1).astype(np.float32)
    x[0] = 0; x[1] = np.finfo(np.float32).smallest_subnormal
    x[2] = np.linspace(-1, 1, 512, dtype=np.float32)
    x[3, ::32] = .95
    exact_norm = np.sum(x.astype(np.float64)**2, axis=1)
    distances = np.sum((x.astype(np.float64)[:, None, :]-x.astype(np.float64)[None, :, :])**2, axis=2)
    results = {}
    for fmt in ('mx_e4m3', 'mx_e3m4'):
        q, s, md = engine.prepare(torch.from_numpy(x).cuda(), fmt)
        z = reconstruction(q, s, fmt)
        got = engine.dots(q, s, fmt).cpu().numpy()
        rec = assert_dot(got, z)
        lo, hi, zu, eu = [m.cpu().numpy().astype(np.float64) for m in md]
        assert np.all(lo <= exact_norm) and np.all(hi >= exact_norm), (fmt, 'original norm')
        assert np.all(zu >= np.linalg.norm(z, axis=1)), (fmt, 'reconstructed norm')
        assert np.all(eu >= np.linalg.norm(x.astype(np.float64)-z, axis=1)), (fmt, 'residual norm')
        n = len(x); cap = n*n; scores = torch.from_numpy(got).cuda()
        out = torch.empty(cap, dtype=torch.int64, device='cuda'); amb = torch.empty_like(out)
        counts = torch.zeros(4, dtype=torch.int32, device='cuda')
        lower = torch.empty_like(scores); upper = torch.empty_like(scores)
        dump_kernel = half_classify[(triton.cdiv(cap, 1024),)](scores, *md, out, amb, counts, lower, upper, 0, 0,
            N=n, ROWS=n, COLS=n, CAP=cap, GAMMA=GAMMA, T=.3955230712890625,
            BLOCK=1024, DUMP=True, num_warps=8, enable_fp_fusion=False)
        engine.kernels[f'interval_dump_{fmt}_{n}'] = dump_kernel
        assert np.all(lower.cpu().numpy() <= distances) and np.all(upper.cpu().numpy() >= distances), (fmt, 'intervals')
        results[fmt] = dict(metadata_and_intervals=True, **rec)

        # Anchor each block to force a known scale. Place every midpoint and
        # its FP32 neighbors in the remaining 31 positions, on both signs.
        book = base.codebook(fmt[3:])[:127]
        mid = ((book[:-1]+book[1:])/2).astype(np.float32)
        values = np.concatenate([book, mid, np.nextafter(mid, np.float32(np.inf)),
                                 np.nextafter(mid, np.float32(-np.inf))]).astype(np.float32)
        values = np.concatenate([values, -values])
        div = 32 if fmt == 'mx_e3m4' else 512
        test = np.zeros((4, 16, 32), dtype=np.float32)
        test[:, :, 0] = book[-1]/div
        for i, val in enumerate(values): test.reshape(-1, 32)[i//31, 1+i%31] = val/div
        test = test.reshape(4, 512)
        tq, ts, _ = engine.prepare(torch.from_numpy(test).cuda(), fmt)
        codes = tq.cpu().numpy(); scales = np.repeat(np.exp2(ts.cpu().numpy().astype(np.float64)-127), 32, axis=1)
        assert np.all(scales == 1/div), (fmt, 'anchor scale')
        normalized = np.abs(test.astype(np.float64)/scales)
        error = np.abs(normalized[:, :, None]-book[None, None, :])
        for idx in np.ndindex(codes.shape):
            ties = np.flatnonzero(error[idx] == error[idx].min())
            expected = next((int(c) for c in ties if c % 2 == 0), int(ties[0]))
            assert (codes[idx] & 127) == expected, (fmt, 'ties-even', idx)
        results[fmt]['midpoint_and_neighbor_values'] = len(values)

        q, s, _ = engine.prepare(torch.from_numpy(x).cuda(), fmt)
        expected = engine.dots(q, s, fmt).cpu().numpy()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): captured = engine.dots(q, s, fmt)
        for _ in range(8): graph.replay()
        torch.cuda.synchronize()
        assert np.array_equal(captured.cpu().numpy(), expected), (fmt, 'Graph')
        for _ in range(2):
            stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream): other = engine.dots(q, s, fmt)
            stream.synchronize()
            assert np.array_equal(other.cpu().numpy(), expected), (fmt, 'stream')
        results[fmt].update(graph_replays=8, non_default_streams=2)
    return results
