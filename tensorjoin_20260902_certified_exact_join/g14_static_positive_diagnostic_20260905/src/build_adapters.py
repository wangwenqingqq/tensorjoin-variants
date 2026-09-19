"""Generate additive diagnostic copies, assert anchors, and retain exact diffs."""

import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parent
EXPECTED = {
    'g2b': '0283769705f91ec089dfebf8094ee7894e68f539523acbba48bef261409cf93c',
    'g5': 'a8653469a0e5bae82b64cd3d9a82c476df741783735c1b9d699e2f27b2db13a5',
}
KERNELS = {
    'g2b': ['triangular_int8_certificate_compact_i64',
            'filter_ambiguous_fp32_i64', 'refine_ambiguous_fp64_i64'],
    'g5': ['analytic_certificate_ragged_safe_i64',
           'certified_fp32_filter_i64', 'refine_ambiguous_fp64_i64'],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique(text, old, new):
    assert text.count(old) == 1, (text.count(old), old)
    return text.replace(old, new)


def create(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        handle.write(text)


def main():
    for method in EXPECTED:
        source = PROJECT / f'src/run_{method}_tensorjoin_public.py'
        assert sha(source) == EXPECTED[method]
        original = source.read_text()
        text = original
        index = text.index('\nEXPERIMENT_ID =')
        setup = '\nfrom g14_diagnostics import Diagnostics, KernelProxy\ndiag = Diagnostics()\n'
        setup += ''.join(f'{kernel} = KernelProxy({kernel}, "stage{i+1}", diag)\n'
                         for i, kernel in enumerate(KERNELS[method]))
        text = text[:index] + setup + text[index:]
        if method == 'g2b':
            text = unique(text, 'if visible != "0":',
                          'if visible != os.environ.get("G5_PHYSICAL_GPU", "2"):')
            text = unique(text, 'Expected CUDA_VISIBLE_DEVICES=0,',
                          'Expected campaign CUDA_VISIBLE_DEVICES,')
            old = 'f"results/g2b_public_{args.phase}_tensorjoin_{record_id}.json"'
            new = f'f"{HERE.name}/results/g2b_{{record_id}}.json"'
        else:
            old = 'f"results/g5_{args.phase}_tensorjoin_{args.record_id}.json"'
            new = f'f"{HERE.name}/results/g5_{{args.record_id}}.json"'
        text = unique(text, old, new)
        text = unique(text, '    torch.cuda.reset_peak_memory_stats()',
                      '    diag.prepare()\n    torch.cuda.reset_peak_memory_stats()')
        text = unique(text, '    public_started = time.perf_counter()',
                      '    public_started = time.perf_counter()\n    diag.start(public_started)')
        # Apply phase markers only inside the original public timer.
        start = text.index('    public_started = time.perf_counter()')
        end = text.index('    public_seconds = time.perf_counter() - public_started')
        body = text[start:end]
        anchor = ('    tile_rows, tile_columns = np.triu_indices(tile_extent)' if method == 'g2b'
                  else '    tile_extent = (N + BLOCK_M - 1) // BLOCK_M')
        body = unique(body, anchor, '    diag.mark("preprocess")\n' + anchor)
        anchor = '    device = torch.device("cuda:0")'
        body = unique(body, anchor, '    diag.mark("schedule")\n' + anchor)
        anchor = '    accepted_parts: list[np.ndarray] = []'
        body = unique(body, anchor, '    diag.mark("h2d_alloc")\n' + anchor)
        anchor = '    for batch_index, tile_start in enumerate('
        body = unique(body, anchor, '    diag.mark("loop_setup")\n' + anchor)
        anchor = '        tile_stop = min(tile_start + TILES_PER_BATCH, scheduled_tiles)'
        body = unique(body, anchor, '        diag.batch = batch_index\n' + anchor)
        anchor = f'        {KERNELS[method][0]}[(batch_tiles,)]('
        body = unique(body, anchor, '        diag.mark("batch_setup")\n' + anchor)
        counter_name = 'scan_counts' if method == 'g2b' else 'scan'
        anchor = f'        {counter_name} = counters.cpu().numpy().astype(np.int64)'
        body = unique(body, anchor, anchor + '\n        diag.mark("stage1_and_count")')
        anchor = '        if ambiguous_count:'
        body = unique(body, anchor, '        diag.mark("decode_s1")\n' + anchor)
        anchor = '        after_fp32 = counters.cpu().numpy().astype(np.int64)'
        body = unique(body, anchor, anchor + '\n        diag.mark("stage2_and_count")')
        anchor = '        if fp64_count:'
        body = unique(body, anchor, '        diag.mark("decode_s2")\n' + anchor)
        counter_name = 'final_counts' if method == 'g2b' else 'final'
        anchor = f'        {counter_name} = counters.cpu().numpy().astype(np.int64)'
        body = unique(body, anchor, anchor + '\n        diag.mark("stage3_and_count")')
        anchor = ('        total_direct += direct_count' if method == 'g2b'
                  else '        record = {')
        body = unique(body, anchor, '        diag.mark("accepted_readback")\n' + anchor)
        anchor = '    torch.cuda.synchronize()\n    accepted_upper = np.sort('
        body = unique(body, anchor,
                      '        diag.mark("bookkeeping_logging")\n\n'
                      '    torch.cuda.synchronize()\n    diag.mark("loop_drain")\n'
                      '    accepted_upper = np.sort(')
        text = text[:start] + body + text[end:]
        text = unique(text, '    public_seconds = time.perf_counter() - public_started',
                      '    public_stopped = time.perf_counter()\n'
                      '    public_seconds = public_stopped - public_started\n'
                      '    diagnostic = diag.finish(public_stopped)')
        text = unique(text, '        "public_seconds": public_seconds,',
                      '        "public_seconds": public_seconds,\n'
                      '        "g14_diagnostic": diagnostic,')
        ast.parse(text)
        target = HERE / f'src/run_{method}_diagnostic.py'
        create(target, text)
        create(HERE / f'artifacts/{method}_adapter.diff', ''.join(
            difflib.unified_diff(original.splitlines(True), text.splitlines(True),
                                 fromfile=str(source.relative_to(PROJECT)),
                                 tofile=str(target.relative_to(PROJECT)))))

    # Freeze the transitive local source imports without importing GPU modules.
    seeds = [PROJECT / f'src/run_{method}_tensorjoin_public.py' for method in EXPECTED]
    seeds += [PROJECT / 'src/run_g5_mistic_public.py', PROJECT / 'src/run_g5_guarded_process.py']
    visited = set()
    while seeds:
        path = seeds.pop()
        if path in visited:
            continue
        visited.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module
                       else [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            for module in modules:
                other = PROJECT / 'src' / (module.split('.')[0] + '.py')
                if other.is_file() and other not in visited:
                    seeds.append(other)
    visited.update(HERE.glob('src/*.py'))
    visited.add(HERE / 'PROTOCOL.md')
    visited.add(PROJECT / 'results/g5_screen_tensorjoin_r0_p2_a0.json')
    visited.add(PROJECT / 'results/g5_screen_mistic_r0_p1_a0.json')
    frozen = {str(p.relative_to(PROJECT)): sha(p) for p in sorted(visited)}
    create(HERE / 'artifacts/frozen_hashes.json', json.dumps(frozen, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'frozen_files': len(frozen), 'directory': str(HERE)}))


if __name__ == '__main__':
    main()
