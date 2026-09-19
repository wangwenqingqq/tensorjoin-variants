"""Summarize the predeclared G14 observations, retaining every process."""

import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parent


def quantile(values, probability):
    values = sorted(values)
    x = (len(values) - 1) * probability
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] + (values[hi] - values[lo]) * (x - lo)


def main():
    campaign = json.loads((HERE / 'results/campaign.json').read_text())
    rows = []
    for slot in campaign['records']:
        row = {k: slot.get(k) for k in (
            'block', 'placement', 'position', 'method', 'admitted',
            'public_seconds', 'result_path', 'guard_path', 'binary_identity_pass')}
        if slot.get('result_path'):
            result = json.loads((PROJECT / slot['result_path']).read_text())
            row['correctness'] = result['correctness']
            row['max_rss_kib'] = result.get('max_rss_kib_after_run')
            diag = result.get('g14_diagnostic')
            if diag:
                row['phase_totals'] = diag['phase_totals']
                row['phase_wall_sum_seconds'] = diag['phase_wall_sum_seconds']
                row['cuda_stream_span_totals_ms'] = {
                    stage: sum(e['cuda_stream_span_ms'] for e in diag['cuda_stream_spans']
                               if e['stage'] == stage)
                    for stage in ('stage1', 'stage2', 'stage3')}
                row['slowest_segments'] = sorted(diag['segments'],
                                                key=lambda s: s['wall_seconds'], reverse=True)[:8]
                row['work_counts'] = {k: result[k] for k in (
                    'batch_count', 'ambiguous_upper_pairs', 'fp64_refine_upper_pairs',
                    'g3b_ambiguous_upper_pairs', 'fp64_refined_upper_pairs',
                    'accepted_upper_pairs') if k in result}
                row['affinity'] = diag['host_before']['affinity']
        rows.append(row)
    distributions, paired = [], []
    for placement in ('default', 'node0'):
        for method in ('g2b', 'g5', 'mistic'):
            values = [r['public_seconds'] for r in rows
                      if r['placement'] == placement and r['method'] == method
                      and r['public_seconds'] is not None]
            if values:
                distributions.append({'placement': placement, 'method': method,
                                      'n': len(values), 'raw_seconds': values,
                                      'p10_seconds': quantile(values, .1),
                                      'median_seconds': statistics.median(values),
                                      'p90_seconds': quantile(values, .9)})
    for block in range(4):
        cells = {r['method']: r for r in rows if r['block'] == block}
        if all(m in cells and cells[m]['public_seconds'] is not None
               for m in ('g2b', 'g5', 'mistic')):
            times = {m: cells[m]['public_seconds'] for m in cells}
            paired.append({'block': block, 'placement': cells['g5']['placement'],
                           'g2b_over_g5_seconds_ratio': times['g2b'] / times['g5'],
                           'mistic_over_g5_seconds_ratio': times['mistic'] / times['g5'],
                           'mistic_over_g2b_seconds_ratio': times['mistic'] / times['g2b']})
    estimators = []
    for placement in ('default', 'node0'):
        group = [p for p in paired if p['placement'] == placement]
        for candidate, keeper in (('g5', 'mistic'), ('g2b', 'mistic'), ('g5', 'g2b')):
            key = f'{keeper}_over_{candidate}_seconds_ratio'
            ratios = [p[key] for p in group]
            if not ratios:
                continue
            candidate_times = [r['public_seconds'] for r in rows if r['method'] == candidate
                               and r['placement'] == placement]
            keeper_times = [r['public_seconds'] for r in rows if r['method'] == keeper
                            and r['placement'] == placement]
            estimators.append({'placement': placement, 'candidate': candidate, 'keeper': keeper,
                               'paired_ratios': ratios, 'process_wins': sum(v > 1 for v in ratios),
                               'n': len(ratios),
                               'paired_geomean_ratio': math.exp(statistics.mean(map(math.log, ratios))),
                               'marginal_median_ratio': statistics.median(keeper_times) /
                                                        statistics.median(candidate_times),
                               'confidence_interval': None,
                               'claim_limit': 'small instrumented diagnostic; no formal promotion'})
    summary = {'diagnostic_only': True, 'campaign_status': campaign['status'],
               'rows': rows, 'distributions': distributions, 'paired': paired,
               'estimators': estimators,
               'historical_g5_slow_mode_seconds': 5.284857193,
               'history_reclassified': False}
    destination = HERE / 'results/summary.json'
    with destination.open('x') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write('\n')
    lines = ['# G14 diagnostic observations', '',
             'Instrumented observations only. No confidence interval, novelty pass,',
             'production speedup claim, or historical result replacement.', '',
             '| Block | Placement | Position | Method | Public s | Admitted |',
             '|---|---|---|---|---:|---|']
    for r in rows:
        seconds = f"{r['public_seconds']:.6f}" if r['public_seconds'] is not None else 'missing'
        lines.append(f"| {r['block']} | {r['placement']} | {r['position']} | {r['method']} | {seconds} | {r['admitted']} |")
    lines += ['', '## Host-wall phase attribution (milliseconds)', '',
              '| Block | Method | Preprocess | Schedule | H2D/alloc | S1+count | S2+count | S3+count | Accepted D2H | Canonicalize | Other |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    keys = ['preprocess', 'schedule', 'h2d_alloc', 'stage1_and_count',
            'stage2_and_count', 'stage3_and_count', 'accepted_readback', 'canonicalize']
    for r in rows:
        if 'phase_totals' not in r:
            continue
        totals = r['phase_totals']
        values = [totals[k]['wall_seconds'] * 1000 for k in keys]
        other = sum(v['wall_seconds'] for k, v in totals.items() if k not in keys) * 1000
        lines.append(f"| {r['block']} | {r['method']} | " + ' | '.join(f'{v:.3f}' for v in values+[other]) + ' |')
    lines += ['', 'Full per-batch wall/CPU/fault/switch observations, stream-span events,',
              'process orders, distributions and paired estimators are in `results/summary.json`',
              'and the immutable child JSON records. Stream spans are not isolated kernel times.', '']
    with (HERE / 'OBSERVATIONS.md').open('x') as handle:
        handle.write('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
