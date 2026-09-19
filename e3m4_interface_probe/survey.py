"""Build, call, and independently validate a bounded SM120 instruction matrix.

The callable interface is run_probe(build, family, a, b, records, output_dir).
Each record is (raw_a_code, raw_b_code, scale_a_exponent, scale_b_exponent).
This broadcasts codes within an MMA tile; it is deliberately not a GEMM API.
"""
import argparse
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CUDA = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda-13.1'))
FORMATS = ('e4m3', 'e5m2', 'e3m4', 'e3m2', 'e2m3', 'e2m1')
DOC = tuple(f for f in FORMATS if f != 'e3m4')
# Absolute bit fields, low bit first, validated anew from 25 compiled controls.
BITS = ((78, 82, 83), (79, 84, 85))
TYPE = dict(zip(FORMATS, range(6)))
MASK = sum(1 << bit for group in BITS for bit in group)
CONFIG = {'e4m3': (4, 3, 7), 'e5m2': (5, 2, 15), 'e3m4': (3, 4, 3),
          'e3m2': (3, 2, 3), 'e2m3': (2, 3, 1), 'e2m1': (2, 1, 1)}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def command(args, timeout=60):
    p = subprocess.run(list(map(str, args)), capture_output=True, text=True,
                       timeout=timeout)
    if p.returncode:
        raise RuntimeError(f'{args[0]} exited {p.returncode}: {p.stdout}\n{p.stderr}')
    return p.stdout, p.stderr


def decode(code, fmt):
    eb, mb, bias = CONFIG[fmt]
    if not 0 <= code < (1 << (1 + eb + mb)):
        raise ValueError('raw code outside format')
    sign = -1 if code >> (eb + mb) else 1
    e, m = (code >> mb) & ((1 << eb) - 1), code & ((1 << mb) - 1)
    if fmt in ('e4m3', 'e3m4') and e == (1 << eb) - 1 and m == (1 << mb) - 1:
        return math.nan
    if fmt == 'e5m2' and e == 31:
        return sign * math.inf if m == 0 else math.nan
    return sign * math.ldexp(m if e == 0 else (1 << mb) + m,
                            (1 if e == 0 else e) - bias - mb)


def codes(fmt):
    e, m, _ = CONFIG[fmt]
    return range(1 << (e + m + 1))


def one(fmt):
    return next(x for x in codes(fmt) if decode(x, fmt) == 1)


def packed(code, fmt):
    decode(code, fmt)  # Domain check before constructing input words.
    # PTX f8f6f4 puts E2M1 in the middle four bits of each byte.
    byte = code << 2 if fmt == 'e2m1' else code
    return byte * 0x01010101


def oracle(record, a, b):
    x, y, sa, sb = record
    return struct.unpack('<f', struct.pack('<f',
        32 * decode(x, a) * decode(y, b) * math.ldexp(1, sa + sb)))[0]


def records_for(a, b, family):
    rows = [(x, one(b), 0, 0) for x in codes(a)]
    rows += [(one(a), y, 0, 0) for y in codes(b)]
    rng = random.Random(20260919)
    fa = [x for x in codes(a) if math.isfinite(decode(x, a))]
    fb = [x for x in codes(b) if math.isfinite(decode(x, b))]
    rows += [(rng.choice(fa), rng.choice(fb), 0, 0) for _ in range(128)]
    if a == 'e3m4': rows.append((0x10, one(b), 0, 0))
    if b == 'e3m4': rows.append((one(a), 0x10, 0, 0))
    if a == b == 'e3m4': rows += [(0x10, 0x10, 0, 0), (0x3c, 0x3c, 0, 0)]
    if family == 'scaled':
        anchors = [(one(a), one(b)), (fa[-2], fb[-2])]
        if a == 'e3m4': anchors.append((0x10, one(b)))
        if b == 'e3m4': anchors.append((one(a), 0x10))
        rows += [(x, y, sa, sb) for x, y in anchors
                 for sa, sb in [(-5, 0), (0, 4), (-2, 3), (3, -3)]]
    return rows


def instruction(path, family):
    sec, _ = command(['readelf', '-SW', path])
    sections = re.findall(r'\.text\.(\S+)\s+PROGBITS\s+[\da-fA-F]+\s+([\da-fA-F]+)', sec)
    assert len(sections) == 1 and sections[0][0] == 'probe', sections
    offset = int(sections[0][1], 16)
    sass, _ = command([CUDA / 'bin/cuobjdump', '-sass', path])
    lines = [l for l in sass.splitlines() if re.search(r'\bQMMA\.', l)]
    assert len(lines) == 1, lines
    line = lines[0]
    assert '.16832.F32.' in line and ('.SF.' in line) == (family == 'scaled'), line
    address = int(re.search(r'/\*([\da-fA-F]+)\*/', line)[1], 16)
    blob = path.read_bytes()
    word = int.from_bytes(blob[offset + address:offset + address + 16], 'little')
    return {'file_offset': offset + address, 'instruction_address': address,
            'word': hex(word), 'sass': line.strip(), 'sha256': digest(blob)}


def field(word, axis):
    return sum(((word >> bit) & 1) << i for i, bit in enumerate(BITS[axis]))


def build(out):
    out.mkdir(parents=True, exist_ok=False)
    version, _ = command([CUDA / 'bin/nvcc', '--version'])
    assert 'V13.1.115' in version, version
    command(['g++', '-std=c++17', '-O2', '-I' + str(CUDA / 'include'),
             ROOT / 'launch.cpp', '-L' + str(CUDA / 'lib64/stubs'), '-lcuda',
             '-o', out / 'launch'])
    data = {'nvcc_version': version, 'target': 'sm_120a',
            'source_sha256': {p.name: digest(p.read_bytes())
                              for p in (ROOT / 'probe.cu', ROOT / 'launch.cpp', ROOT / 'survey.py')},
            'families': {}, 'reserved_not_executed': [6, 7]}
    for fam in ('plain', 'scaled'):
        controls = {}
        for a, b in itertools.product(DOC, repeat=2):
            path = out / f'{fam}_{a}_{b}.cubin'
            args = [CUDA / 'bin/nvcc', '-cubin', '-arch=sm_120a', '-O3', '-std=c++17',
                    '-lineinfo', '-Xptxas=-v', f'-DATYPE="{a}"', f'-DBTYPE="{b}"']
            if fam == 'scaled': args.append('-DSCALED')
            _, stderr = command(args + [ROOT / 'probe.cu', '-o', path])
            info = instruction(path, fam)
            w = int(info['word'], 16)
            assert (field(w, 0), field(w, 1)) == (TYPE[a], TYPE[b]), info
            info['ptxas'] = stderr
            controls[f'{a}_{b}'] = info
        baseinfo = controls['e4m3_e4m3']
        base = (out / f'{fam}_e4m3_e4m3.cubin').read_bytes()
        baseword = int(baseinfo['word'], 16)
        # Fail closed if registers/control differ inside the selected instruction.
        for info in controls.values():
            assert (int(info['word'], 16) ^ baseword) & ~MASK == 0, info
        patches = {}
        for a, b in itertools.product(FORMATS, repeat=2):
            if 'e3m4' not in (a, b): continue
            mask = sum(((TYPE[fmt] >> i) & 1) << bit
                       for fmt, bits in zip((a, b), BITS) for i, bit in enumerate(bits))
            assert (baseword & MASK) == 0
            off = baseinfo['file_offset']
            buf = bytearray(base)
            buf[off:off+16] = (baseword | mask).to_bytes(16, 'little')
            assert len(buf) == len(base)
            assert all(x == y for i, (x, y) in enumerate(zip(base, buf))
                       if not off <= i < off + 16)
            assert int.from_bytes(buf[off:off+16], 'little') ^ baseword == mask
            path = out / f'{fam}_{a}_{b}.cubin'; path.write_bytes(buf)
            sass, _ = command([CUDA / 'bin/cuobjdump', '-sass', path])
            (out / f'{fam}_{a}_{b}.sass').write_text(sass)
            patches[f'{a}_{b}'] = {'sha256': digest(buf), 'base_sha256': digest(base),
                'instruction_address': baseinfo['instruction_address'], 'file_offset': off,
                'xor_bits': [i for i in range(128) if mask >> i & 1],
                'word': hex(baseword | mask), 'changed_bytes': sum(x != y for x, y in zip(base, buf)),
                'disassembly_sha256': digest(sass.encode())}
        data['families'][fam] = {'controls': controls, 'patches': patches}
    data['launcher_sha256'] = digest((out / 'launch').read_bytes())
    dump(out / 'manifest.json', data)
    print('Built 50 documented controls and 22 hidden-format candidates.', flush=True)


def same(x, y):
    return math.isnan(x) and math.isnan(y) or x == y


def json_value(x):
    return x if math.isfinite(x) else ('nan' if math.isnan(x) else ('inf' if x > 0 else '-inf'))


def launch_process(args, log, report_path, report, timeout=120):
    """Keep partial diagnostics if subprocess.run kills a timed-out child."""
    try:
        p = subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        def text(value):
            return value.decode(errors='replace') if isinstance(value, bytes) else value or ''
        log.write_text(text(exc.stdout) + text(exc.stderr))
        report.update({'pass': False, 'failure': 'timeout', 'timeout_seconds': timeout, 'exit_code': None})
        dump(report_path, report)
        raise RuntimeError('probe timed out; partial diagnostics retained') from exc
    log.write_text(p.stdout + p.stderr)
    return p


def run_probe(build_dir, family, a, b, records, output_dir, repeats=1, sanitizer=None):
    """Call one frozen format/family in a fresh CUDA process and check all outputs."""
    if family not in ('plain', 'scaled') or a not in FORMATS or b not in FORMATS:
        raise ValueError('unsupported family or format')
    if not 1 <= repeats <= 1000 or not 1 <= len(records) <= 65535:
        raise ValueError('invalid repeat or record count')
    meta = json.loads((build_dir / 'manifest.json').read_text())
    key = f'{a}_{b}'; stem = f'{family}_{key}'
    info = (meta['families'][family]['patches'] if 'e3m4' in (a, b)
            else meta['families'][family]['controls'])[key]
    cubin = build_dir / f'{stem}.cubin'; launcher = build_dir / 'launch'
    assert digest(cubin.read_bytes()) == info['sha256']
    assert digest(launcher.read_bytes()) == meta['launcher_sha256']
    binary = bytearray()
    for x, y, sa, sb in records:
        if family == 'plain' and (sa != 0 or sb != 0):
            raise ValueError('plain family cannot apply scales')
        if not -5 <= sa <= 4 or not -5 <= sb <= 4:
            raise ValueError('scale outside validated probe domain')
        binary.extend(struct.pack('<4I', packed(x, a), packed(y, b),
                                  (sa + 127) * 0x01010101, (sb + 127) * 0x01010101))
    output_dir.mkdir(parents=True, exist_ok=True)
    inp = output_dir / f'{stem}.input.bin'; out = output_dir / f'{stem}.output.bin'
    if inp.exists() or out.exists(): raise FileExistsError('refusing to overwrite probe evidence')
    inp.write_bytes(binary)
    args = [launcher, cubin, inp, out, str(repeats)]
    if sanitizer:
        if sanitizer not in ('memcheck', 'racecheck', 'synccheck', 'initcheck'):
            raise ValueError('unsupported sanitizer')
        args = [CUDA / 'bin/compute-sanitizer', '--tool', sanitizer, '--error-exitcode', '3'] + args
    report = {'family': family, 'a': a, 'b': b, 'records': records, 'repeats': repeats,
              'sanitizer': sanitizer,
              'input_sha256': digest(binary), 'cubin_sha256': info['sha256'], 'pass': False}
    p = launch_process(args, output_dir / f'{stem}.log', output_dir / f'{stem}.json', report)
    report['exit_code'] = p.returncode
    if p.returncode != 0:
        dump(output_dir / f'{stem}.json', report)
        raise RuntimeError(f'CUDA process failed: {stem}, see its log')
    report['resources'] = json.loads(next(l for l in p.stdout.splitlines() if l.startswith('{')))
    raw = out.read_bytes()
    assert len(raw) == len(records) * 128 * 4 * repeats
    report['output_sha256'] = digest(raw)
    vals = struct.unpack('<' + 'f' * (len(raw) // 4), raw)
    expected = [oracle(row, a, b) for row in records]
    bad = []
    for i, x in enumerate(vals):
        row = (i // 128) % len(records)
        if not same(x, expected[row]) and len(bad) < 20:
            bad.append({'output_index': i, 'record': row, 'actual': json_value(x),
                        'expected': json_value(expected[row])})
    report['observed_first_lane'] = [json_value(vals[i * 128]) for i in range(len(records))]
    report['checked_outputs'] = len(vals)
    report['mismatch_examples'] = bad
    report['pass'] = not bad
    # Independent anti-fallback oracle: reinterpret only hidden operands as E4M3.
    rejects = sum(not same(ex, oracle(row, 'e4m3' if a == 'e3m4' else a,
                                     'e4m3' if b == 'e3m4' else b))
                  for row, ex in zip(records, expected)) if 'e3m4' in (a, b) else 0
    report['anti_e4m3_discriminating_records'] = rejects
    if 'e3m4' in (a, b): assert rejects > 0
    dump(output_dir / f'{stem}.json', report)
    (output_dir / f'{stem}.output.bin.gz').write_bytes(gzip.compress(raw, mtime=0))
    if not report['pass']: raise AssertionError(f'oracle mismatch: {stem}: {bad[:2]}')
    return report


def run_all(build_dir, output_dir, mode):
    output_dir.mkdir(parents=True, exist_ok=False)
    reports = []
    for fam in ('plain', 'scaled'):
        for a, b in itertools.product(FORMATS, repeat=2):
            hidden = 'e3m4' in (a, b)
            if mode != 'validate' and not hidden: continue
            if mode in ('racecheck', 'synccheck', 'initcheck') and a != b: continue
            rows = records_for(a, b, fam)
            report = run_probe(build_dir, fam, a, b, rows, output_dir,
                               repeats=20 if mode == 'stress' else 1,
                               sanitizer=mode if mode.endswith('check') else None)
            reports.append({k: v for k, v in report.items()
                            if k not in ('records', 'observed_first_lane')})
            print(f'{fam} {a} x {b}: PASS ({report["checked_outputs"]} outputs)', flush=True)
    dump(output_dir / 'SUMMARY.json', {'mode': mode, 'pass': all(r['pass'] for r in reports),
                                      'probes': reports})


def selftest():
    assert decode(0x10, 'e3m4') == .25 and decode(0x10, 'e4m3') == 1/32
    assert decode(1, 'e3m4') == 1/64 and decode(0x7e, 'e3m4') == 30
    assert math.isnan(decode(0x7f, 'e3m4')) and math.isnan(decode(0xff, 'e3m4'))
    assert decode(0x7b, 'e5m2') == 57344 and math.isinf(decode(0x7c, 'e5m2'))
    assert decode(0x1f, 'e3m2') == 28 and decode(0x1f, 'e2m3') == 7.5
    assert decode(7, 'e2m1') == 6 and packed(2, 'e2m1') == 0x08080808
    assert oracle((0x10, 0x10, 0, 0), 'e3m4', 'e3m4') == 2
    assert oracle((0x3c, 0x3c, 0, 0), 'e3m4', 'e3m4') == 98
    for a, b, fam in itertools.product(FORMATS, FORMATS, ('plain', 'scaled')):
        for r in records_for(a, b, fam): oracle(r, a, b)
    print('CPU oracle / packing / record-generation checks passed.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('build', 'validate', 'stress', 'memcheck',
                                   'racecheck', 'synccheck', 'initcheck', 'selftest'))
    p.add_argument('--build', type=Path, default=ROOT / 'build')
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    if args.mode == 'selftest': selftest()
    elif args.mode == 'build': build(args.build.resolve())
    else:
        if args.out is None: p.error('--out is required and must not exist')
        run_all(args.build.resolve(), args.out.resolve(), args.mode)
