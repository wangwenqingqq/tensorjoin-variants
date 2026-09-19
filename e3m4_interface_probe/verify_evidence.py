"""Offline replay of every retained output; no CUDA or third-party dependency."""
import gzip
import itertools
import json
from pathlib import Path
import struct

from survey import FORMATS, ROOT, TYPE, digest, field, json_value, oracle, packed, records_for, same, selftest


def verify(root=ROOT):
    selftest()
    results = root / 'results'
    build = json.loads((results / 'BUILD_MANIFEST.json').read_text())
    for name, sha in build['source_sha256'].items():
        assert digest((root / name).read_bytes()) == sha, name
    guard = json.loads((results / 'GPU_SUPERVISION.json').read_text())
    assert digest((root.parent / 'precision_format_routing/src/guard.py').read_bytes()) == guard['guard_source_sha256']
    assert len(guard['runs']) == 6 and all(r['pass'] and r['exit_code'] == 0 for r in guard['runs'])
    static = json.loads((results / 'static/STATIC_CHECKS.json').read_text())
    assert static['pass'] and static['source_sha256'] == build['source_sha256']['probe.cu']
    assert len(static['low_optimization_controls']) == 12
    for control in static['low_optimization_controls']:
        assert [field(int(control['word'], 16), axis) for axis in (0, 1)] == [TYPE[control['a']], TYPE[control['b']]]
    assert len(static['decoders']) == 8
    assert len(static['direct_e3m4_ptx_rejection']) == 2
    assert all(r['exit_code'] != 0 for r in static['direct_e3m4_ptx_rejection'])
    total = 0
    inventory = {}
    for mode, required in [('validate', 72), ('stress', 22), ('memcheck', 22),
                           ('racecheck', 2), ('synccheck', 2), ('initcheck', 2)]:
        directory = results / mode
        summary = json.loads((directory / 'SUMMARY.json').read_text())
        paths = sorted(p for p in directory.glob('*.json') if p.name != 'SUMMARY.json')
        assert len(paths) == required and len(summary['probes']) == required
        seen = set()
        count = 0
        for p in paths:
            r = json.loads(p.read_text()); stem = p.stem
            key = (r['family'], r['a'], r['b']); assert key not in seen; seen.add(key)
            assert r['pass'] and r['exit_code'] == 0 and not r['mismatch_examples']
            assert r['repeats'] == (20 if mode == 'stress' else 1)
            assert r['sanitizer'] == (mode if mode.endswith('check') else None)
            assert r['records'] == list(map(list, records_for(r['a'], r['b'], r['family'])))
            blob = gzip.decompress((directory / (stem + '.output.bin.gz')).read_bytes())
            assert digest(blob) == r['output_sha256']
            vals = struct.unpack('<' + 'f' * (len(blob) // 4), blob)
            expected = [oracle(row, r['a'], r['b']) for row in r['records']]
            assert len(vals) == len(expected) * 128 * r['repeats'] == r['checked_outputs']
            assert all(same(v, expected[(i // 128) % len(expected)]) for i, v in enumerate(vals)), stem
            assert r['observed_first_lane'] == [json_value(vals[i * 128]) for i in range(len(expected))]
            inp = b''.join(struct.pack('<4I', packed(x, r['a']), packed(y, r['b']),
                                      (sa + 127) * 0x01010101, (sb + 127) * 0x01010101)
                           for x, y, sa, sb in r['records'])
            assert digest(inp) == r['input_sha256']
            group = 'patches' if 'e3m4' in (r['a'], r['b']) else 'controls'
            assert r['cubin_sha256'] == build['families'][r['family']][group][f'{r["a"]}_{r["b"]}']['sha256']
            if group == 'patches':
                negative = [oracle(row, 'e4m3' if r['a'] == 'e3m4' else r['a'],
                                   'e4m3' if r['b'] == 'e3m4' else r['b']) for row in r['records']]
                assert r['anti_e4m3_discriminating_records'] == sum(not same(x, y) for x, y in zip(expected, negative)) > 0
            if mode.endswith('check'):
                log = (directory / (stem + '.log')).read_text()
                assert ('ERROR SUMMARY: 0 errors' in log or
                        'RACECHECK SUMMARY: 0 hazards displayed' in log), stem
            summary_row = next(x for x in summary['probes']
                               if (x['family'], x['a'], x['b']) == key)
            assert summary_row == {k: v for k, v in r.items()
                                   if k not in ('records', 'observed_first_lane')}
            count += len(vals)
        assert summary['pass']
        if mode == 'validate':
            assert seen == set(itertools.product(('plain', 'scaled'), FORMATS, FORMATS))
        inventory[mode] = {'processes': required, 'checked_outputs': count}
        total += count
    print(json.dumps({'pass': True, 'checked_outputs': total, 'inventory': inventory}, indent=2))


if __name__ == '__main__':
    verify()
