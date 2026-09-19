"""Serial bounded matrix; native mismatch is retained, never promoted as exact."""
import argparse,fcntl,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];PROJECT=HERE.parent
PYTHON='@TENSORJOIN_ROOT@/isaacsim6/env/bin/python'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['native','safety'],required=True);args=parser.parse_args()
    target=HERE/f'results/{args.phase}_matrix_a0.json';assert not target.exists()
    if args.phase=='native':
        names=[x['name'] for x in json.loads((HERE/'data/manifest.json').read_text())['inputs']]
        slots=[(name,'none',f'native_{name}_a0') for name in names]
    else:
        previous=json.loads((HERE/'results/native_matrix_a0.json').read_text());assert previous['all_structural_pass']
        slots=[(name,tool,f'{tool}_{name}_a0') for tool in ['memcheck','synccheck'] for name in ['boundary31','cifar4096']]
    records=[]
    with open('/tmp/tensorjoin_gpu2_campaign.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (HERE/f'results/{args.phase}_start_a0.json').write_text(json.dumps(dict(slots=slots,scope='compatibility/candidate characterization only'),indent=2)+'\n')
        for name,tool,rid in slots:
            path=HERE/f'results/case_{rid}.json';label='g17_'+rid
            command=[PYTHON,str(PROJECT/'src/run_g5_guarded_process.py'),'--label',label,'--expected-result',str(path.relative_to(PROJECT)),'--physical-gpu','2','--',PYTHON,str(HERE/'src/run_case.py'),'--case',name,'--record-id',rid,'--sanitizer',tool]
            print('SLOT_START '+json.dumps(command),flush=True)
            done=subprocess.run(command,check=False);gpath=PROJECT/f'results/g5_guard_{label}.json'
            guard=json.loads(gpath.read_text()) if gpath.exists() else {};result=json.loads(path.read_text()) if path.exists() else {}
            structural=result.get('correctness',{}).get('adapter_structure_pass',False)
            occupancy=bool(guard and not guard['foreign_rows'] and not guard['postflight_compute_rows'])
            row=dict(case=name,record_id=rid,tool=tool,guard_returncode=done.returncode,guard_admitted_exact=bool(guard.get('admitted')),
                result_path=str(path.relative_to(PROJECT)),guard_path=str(gpath.relative_to(PROJECT)),structure_pass=structural,occupancy_clean=occupancy,
                reference_pass=result.get('correctness',{}).get('exact_contract_pass',False),
                audit=result.get('audit'),error=result.get('error'))
            records.append(row);(HERE/f'results/matrix_slot_{rid}.json').write_text(json.dumps(row,indent=2)+'\n')
            print('SLOT_COMPLETE '+json.dumps({k:v for k,v in row.items() if k!='audit'}),flush=True)
            # Numerical mismatch permits only the predeclared bounded characterization.
            # No unsafe/missing artifact/decoder failure can reach larger input.
            if not structural or not occupancy:break
    complete=len(records)==len(slots)
    record=dict(complete=complete,all_structural_pass=complete and all(x['structure_pass'] and x['occupancy_clean'] for x in records),
        all_reference_pass=complete and all(x['reference_pass'] for x in records),records=records,
        native_numeric_failures_are_not_exact_admission=True,public_performance_admitted=False)
    target.write_text(json.dumps(record,indent=2)+'\n');print('MATRIX_COMPLETE '+json.dumps({k:v for k,v in record.items() if k!='records'}),flush=True)
    return 0 if record['all_structural_pass'] else 2
if __name__=='__main__':raise SystemExit(main())
