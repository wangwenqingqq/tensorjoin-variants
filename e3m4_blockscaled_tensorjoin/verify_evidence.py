"""CPU-only consistency replay of the published block-scaled campaign evidence."""
import gzip
import hashlib
import json
from pathlib import Path

from summarize import METHODS, summarize

ROOT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(name):
    return json.loads((ROOT / 'results' / name).read_text())


def main():
    manifest = read('EVIDENCE_MANIFEST.json')
    for name, expected in manifest['files_sha256'].items():
        assert digest((ROOT / name).read_bytes()) == expected, name
    source = read('probe_v3.json')['source_sha256']
    for name, expected in source.items():
        assert digest((ROOT.parent / name).read_bytes()) == expected, name
    labels = ['probe_v3', 'validate_v3', 'stress_v3']
    labels += [f'san_{tool}_{case}_v3'
               for tool in ('memcheck', 'racecheck', 'initcheck', 'synccheck')
               for case in ('cifar4096', 'boundary_zero32')]
    labels += [f'bench_v1_{i}' for i in range(4)]
    observed_scaled = set()
    for label in labels:
        data = read(label + '.json')
        assert data['pass'] and data['source_sha256'] == source, label
        rows = [json.loads(s) for s in (ROOT / 'results' / (label + '.jsonl')).read_text().splitlines()]
        assert rows == data['records'], label
        reference = {r['case']: r for r in rows if r['role'] == 'reference'}
        for row in rows:
            if row['role'] not in ('reference', 'batch_wall'):
                ref = reference[row['case']]
                for key in ('input_sha256', 'output_sha256', 'output_count', 'n', 'threshold_d2'):
                    assert row[key] == ref[key], (label, row['case'], key)
        compiled = read(label + '.compiled.json')
        assert compiled and all('arch=120' in v['target'] for v in compiled.values())
        observed_scaled.update(v['cubin_sha256'] for v in compiled.values()
                               if v['name'].startswith('dot_mx_e3m4_'))
    probe = read('probe_v3.json')
    code = probe['native_layout']['codebook']
    assert code['mx_e3m4']['raw_0x10_k64'] == 4
    assert code['mx_e4m3']['raw_0x10_k64'] == .0625
    assert all(v['exact'] and v['checked_pairs'] == 65536 for v in code.values())
    assert len(probe['native_layout']['layout']) == 20
    assert all(v['max_error_over_bound'] <= 1 for v in probe['native_layout']['layout'])
    compiled = read('probe_v3.compiled.json')
    assert sum(v['name'].startswith('interval_dump_') for v in compiled.values()) == 2
    for label, repeats in [('validate_v3', 1), ('stress_v3', 16)]:
        rows = read(label + '.json')['records']
        cases = {r['case'] for r in rows}
        assert len(cases) == 14 and len(rows) == 14 * (1 + len(METHODS) * repeats)
        for case in cases:
            for method in METHODS:
                subset = [r for r in rows if r['case'] == case and r['format'] == method]
                assert len(subset) == repeats and all(r['exact_ids'] for r in subset)
    assert read('stress_v3.json')['distinct_gpu_input_addresses'] >= 16
    gates = read('GPU_SUPERVISION.json')
    for label in labels:
        row = gates['runs']['mx_' + label]
        assert row['pass'] and row['gpu'] == 7 and row['exit_code'] == 0, label
    for name, row in manifest['sanitizers'].items():
        log = (ROOT / 'results' / 'logs' / (name + '.log')).read_text()
        expected = ('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'
                    if 'racecheck' in name else 'ERROR SUMMARY: 0 errors')
        assert expected in log and row['pass'] and row['summaries'], name
    native = ROOT / 'results' / 'native'
    patched = set()
    for report in native.glob('*.patch.json'):
        r = json.loads(report.read_text()); stem = report.name.removesuffix('.patch.json')
        patched.add(r['candidate_sha256'])
        sections = json.loads((native / (stem + '.text.json')).read_text())
        a = gzip.decompress((native / (stem + '.baseline.cubin.text.gz')).read_bytes())
        b = gzip.decompress((native / (stem + '.e3m4.cubin.text.gz')).read_bytes())
        for data, suffix, key in [(a, '.baseline.cubin', 'baseline_sha256'),
                                  (b, '.e3m4.cubin', 'candidate_sha256')]:
            assert sections[suffix]['container_sha256'] == r[key]
            assert sections[suffix]['text_sha256'] == digest(data)
            assert sections[suffix]['text_size'] == len(data)
        start = sections['.baseline.cubin']['text_file_offset']
        assert start == sections['.e3m4.cubin']['text_file_offset']
        expected = bytearray(a)
        assert len(a) == len(b) and r['allowed_xor_bits'] == [82, 84]
        assert r['instruction_count'] == len(r['instructions']) > 0
        for insn in r['instructions']:
            off = insn['address']; assert insn['file_offset'] == start + off
            word = int.from_bytes(a[off:off + 16], 'little')
            assert hex(word) == insn['word'] and insn['fields'] == [0, 0]
            expected[off:off + 16] = (word ^ (1 << 82) ^ (1 << 84)).to_bytes(16, 'little')
        assert bytes(expected) == b
    assert patched == observed_scaled, 'native evidence/observed-specialization mismatch'
    assert summarize() == read('SUMMARY.json'), 'summary drift'
    print(json.dumps({'pass': True, 'final_runs': len(labels), 'source_files': len(source),
                      'hashed_evidence_files': len(manifest['files_sha256']),
                      'native_patch_pairs': len(list(native.glob('*.patch.json'))),
                      'scope': 'offline consistency, hashes and extracted .text diffs; full containers remain private; does not rerun a GPU'}))


if __name__ == '__main__':
    main()
