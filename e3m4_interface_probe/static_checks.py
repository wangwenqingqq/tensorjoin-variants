"""Supplementary low-optimization field controls and dual-decoder observations."""
import argparse
import json
from pathlib import Path
import subprocess

from survey import CUDA, ROOT, TYPE, command, digest, dump, field, instruction


def main(build, out):
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((build / 'manifest.json').read_text())
    version, _ = command([CUDA / 'bin/nvcc', '--version'])
    assert 'V13.1.115' in version
    rows, decoders, rejected = [], [], []
    for family in ('plain', 'scaled'):
        for a, b in [('e4m3', 'e4m3'), ('e5m2', 'e4m3'), ('e4m3', 'e5m2'),
                     ('e3m2', 'e3m2'), ('e2m3', 'e2m3'), ('e2m1', 'e2m1')]:
            path = out / f'{family}_{a}_{b}.cubin'
            args = [CUDA / 'bin/nvcc', '-cubin', '-arch=sm_120a', '-O0',
                    '-Xptxas=-O0', '-std=c++17', f'-DATYPE="{a}"', f'-DBTYPE="{b}"']
            if family == 'scaled': args.append('-DSCALED')
            command(args + [ROOT / 'probe.cu', '-o', path])
            info = instruction(path, family)
            word = int(info['word'], 16)
            assert (field(word, 0), field(word, 1)) == (TYPE[a], TYPE[b])
            rows.append({'family': family, 'a': a, 'b': b, **info})
        for fmt in ('e4m3', 'e3m4'):
            path = build / f'{family}_{fmt}_{fmt}.cubin'
            group = 'patches' if fmt == 'e3m4' else 'controls'
            sha = manifest['families'][family][group][f'{fmt}_{fmt}']['sha256']
            assert digest(path.read_bytes()) == sha
            for tool, flags in [('cuobjdump', ['-sass']), ('nvdisasm', ['-c'])]:
                p = subprocess.run([str(CUDA / 'bin' / tool), *flags, str(path)],
                                   capture_output=True, text=True, timeout=30)
                raw = p.stdout + p.stderr
                curated = raw.replace(str(ROOT), '${PROBE_ROOT}').replace(str(CUDA), '${CUDA_HOME}').rstrip() + '\n'
                name = f'{family}_{fmt}_{tool}.txt'; (out / name).write_text(curated)
                decoders.append({'family': family, 'format': fmt, 'tool': tool,
                                 'exit_code': p.returncode, 'cubin_sha256': sha,
                                 'raw_text_sha256': digest(raw.encode()), 'curated_file': name})
        args = [CUDA / 'bin/nvcc', '-cubin', '-arch=sm_120a', '-O3', '-std=c++17',
                '-DATYPE="e3m4"', '-DBTYPE="e3m4"']
        if family == 'scaled': args.append('-DSCALED')
        p = subprocess.run(list(map(str, args + [ROOT / 'probe.cu', '-o', out / f'{family}_direct.cubin'])),
                           capture_output=True, text=True, timeout=60)
        assert p.returncode != 0, 'E3M4 now compiles directly: revisit interface assumptions'
        # Keep the compiler's semantic error, excluding ephemeral temporary paths.
        lines = [line.split('; ', 1)[-1] for line in (p.stdout+p.stderr).splitlines()]
        rejected.append({'family': family, 'exit_code': p.returncode, 'diagnostic': lines})
    dump(out / 'STATIC_CHECKS.json', {'pass': True, 'low_optimization_controls': rows,
                                     'decoders': decoders, 'direct_e3m4_ptx_rejection': rejected,
                                     'source_sha256': digest((ROOT / 'probe.cu').read_bytes())})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); main(args.build.resolve(), args.out.resolve())
