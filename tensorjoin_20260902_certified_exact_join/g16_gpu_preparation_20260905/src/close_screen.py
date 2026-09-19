"""Cross-check all fixed public slots and write descriptive decision-grade facts."""
import json,math,statistics
from pathlib import Path
from g2b_public_common import atomic_json,sha256_file
HERE=Path(__file__).resolve().parents[1];PROJECT=HERE.parent;OLD=PROJECT/'g15_strong_control_20260905'

def main():
    screen=json.loads((HERE/'results/public_screen.json').read_text());assert screen['complete']
    precision=json.loads((HERE/'results/precision_audit.json').read_text());assert precision['precision_gate_pass']
    tc=json.loads((HERE/'results/gpu_prepare_compatibility_full_a0.json').read_text())
    fp=json.loads((HERE/'results/fp32_gpu_compatibility_full_a0.json').read_text())
    cpu=json.loads((OLD/'results/fp32_compatibility_full_a0.json').read_text())
    facts=[]
    for row in screen['records']:
        path=PROJECT/row['result_path'];assert sha256_file(path)==row['result_sha256']
        r=json.loads(path.read_text());g=json.loads((PROJECT/row['guard_path']).read_text())
        assert g['admitted'] and not g['foreign_rows'] and not g['postflight_compute_rows']
        c=r['correctness'];assert c['exact_contract_pass'] and c['canonical_pair_count']==3926078
        assert c['canonical_raw_u64_sha256']=='13cae87e6a0f75a13f3e9a7881b33c42ab77680f6d13e99dcc00fa2de5963495'
        if row['method']=='gpu_g5':
            for key in precision['g16_extra']['tc_work']: assert r[key]==tc[key],key
            key='selected_cache_artifacts_after_timer'
            for name,e in r[key].items():
                for suffix in ('cubin','ptx'):assert e[suffix+'_sha256']==tc[key][name][suffix+'_sha256']
            for suffix in ('cubin','ptx'):
                assert r['metadata_artifacts_after'][suffix]['sha256']==tc['metadata_artifacts_after'][suffix]['sha256']
        else:
            reference=fp if row['method']=='fp32_gpu' else cpu
            assert set(r['compiled_cache'].values())==set(reference['compiled_cache'].values())
            assert {k:v for k,v in r['library'].items() if k!='handle'}=={k:v for k,v in reference['library'].items() if k!='handle'}
            for key in ('computed_dot_pairs','ambiguous_upper_pairs','accepted_upper_pairs','overflow_events'):assert r[key]==reference[key],key
        facts.append(dict(block=row['block'],method=row['method'],output_and_code_identity_pass=True,result_sha256=sha256_file(path)))
    distributions={}
    for method in ('gpu_g5','fp32_cpu','fp32_gpu'):
        values=sorted(r['public_seconds'] for r in screen['records'] if r['method']==method)
        a,b=values;distributions[method]=dict(n=2,raw=values,p10=a+.1*(b-a),median=(a+b)/2,p90=a+.9*(b-a))
    result=dict(all_public_identity_gates_pass=True,slots=facts,distributions=distributions,
         screen_sha256=sha256_file(HERE/'results/public_screen.json'),precision_sha256=sha256_file(HERE/'results/precision_audit.json'),
         warning='n=2 interpolated descriptive quantiles only; no CI/tail/sustained/generalization claim')
    atomic_json(HERE/'results/closure_checks.json',result)
    print(json.dumps(result))
if __name__=='__main__':main()
