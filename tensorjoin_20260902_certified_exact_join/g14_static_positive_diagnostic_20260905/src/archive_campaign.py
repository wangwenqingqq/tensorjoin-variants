"""Verify referenced receipts and archive sources; keep large pair files remote."""

import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    c = json.loads((HERE / 'results/campaign.json').read_text())
    assert c['status'] == 'complete_diagnostic_no_promotion'
    assert len(c['records']) == 12
    all_paths, large = set(), {}

    def verify(relative, digest):
        path = PROJECT / relative
        assert sha(path) == digest, relative
        all_paths.add(path)

    for relative, digest in c['frozen_hashes'].items():
        verify(relative, digest)
        source = PROJECT / relative
        target = HERE / 'artifacts/source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        shutil.copyfile(source, target)
        assert sha(target) == digest
    for slot in c['records']:
        assert slot['admitted']
        verify(slot['result_path'], slot['result_sha256'])
        verify(slot['guard_path'], slot['guard_sha256'])
        guard = json.loads((PROJECT / slot['guard_path']).read_text())
        assert guard['gpu_uuid'] == 'GPU-16f27f5a-dfcd-48e0-bb39-bebbe4009245'
        assert not guard['foreign_rows'] and not guard['postflight_compute_rows']
        for field in ('raw_log', 'preflight_log', 'occupancy_log'):
            verify(guard[field], guard[field + '_sha256'])
        if guard['triton_cache']:
            all_paths.update(p for p in (PROJECT / guard['triton_cache']).rglob('*') if p.is_file())
        result = json.loads((PROJECT / slot['result_path']).read_text())
        correct = result['correctness']
        assert correct['canonical_pair_count'] == 3926078
        assert correct['canonical_raw_u64_sha256'] == '13cae87e6a0f75a13f3e9a7881b33c42ab77680f6d13e99dcc00fa2de5963495'
        if slot['method'] == 'mistic':
            pair = PROJECT / result['pair_file']
            assert sha(pair) == result['pair_file_sha256']
            large[result['pair_file']] = {'bytes': pair.stat().st_size,
                                         'sha256': result['pair_file_sha256']}
    binary = PROJECT / 'adapters/mistic_g2b_public/build/main_d512'
    assert sha(binary) == c['mistic_binary_sha256']
    large[str(binary.relative_to(PROJECT))] = {'bytes': binary.stat().st_size,
                                             'sha256': sha(binary)}
    all_paths.update(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    manifest = {'verified_slots': 12, 'remote_only_large_artifacts': large,
                'files': {str(p.relative_to(PROJECT)): {'sha256': sha(p), 'bytes': p.stat().st_size}
                          for p in sorted(all_paths)}}
    path = HERE / 'artifacts/raw_evidence_manifest.json'
    with path.open('x') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')
    all_paths.add(path)
    with (HERE / 'artifacts/transfer_files.txt').open('x') as handle:
        for p in sorted(all_paths):
            handle.write(str(p.relative_to(PROJECT)) + '\n')
    print(json.dumps({'verified_slots': 12, 'files_to_sync': len(all_paths),
                      'transfer_bytes': sum(p.stat().st_size for p in all_paths),
                      'retained_large_artifacts': large}))


if __name__ == '__main__':
    main()
