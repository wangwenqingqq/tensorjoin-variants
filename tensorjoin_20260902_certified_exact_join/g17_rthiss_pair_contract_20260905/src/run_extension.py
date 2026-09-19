"""Preserved extra endpoint probe and selected-function dispatch trace."""
import fcntl,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];PROJECT=HERE.parent;PYTHON='@TENSORJOIN_ROOT@/isaacsim6/env/bin/python'
def main():
    p=HERE/'results/safety_matrix_a0.json';assert json.loads(p.read_text())['all_structural_pass']
    target=HERE/'results/extension_a0.json';assert not target.exists();records=[]
    with open('/tmp/tensorjoin_gpu2_campaign.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for tool in ['none','memcheck','synccheck','nsys']:
            case='real31' if tool=='nsys' else 'boundary_zero32';rid=f'{tool}_{case}_a0'
            result=HERE/f'results/case_{rid}.json'
            child=[PYTHON,str(HERE/'src/run_case_r1.py'),'--case',case,'--record-id',rid,'--sanitizer','none' if tool=='nsys' else tool]
            if tool=='nsys':child=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--force-overwrite=false','-o',str(HERE/'artifacts/nsys_adapter_a0')]+child
            command=[PYTHON,str(PROJECT/'src/run_g5_guarded_process.py'),'--label','g17_'+rid,'--expected-result',str(result.relative_to(PROJECT)),'--physical-gpu','2','--']+child
            done=subprocess.run(command,check=False);gpath=PROJECT/f'results/g5_guard_g17_{rid}.json'
            guard=json.loads(gpath.read_text()) if gpath.exists() else {};r=json.loads(result.read_text()) if result.exists() else {}
            structure=r.get('correctness',{}).get('adapter_structure_pass',False)
            clean=bool(guard and not guard['foreign_rows'] and not guard['postflight_compute_rows'])
            records.append(dict(record_id=rid,case=case,tool=tool,command=command,returncode=done.returncode,guard_path=str(gpath.relative_to(PROJECT)),result_path=str(result.relative_to(PROJECT)),structure_pass=structure,occupancy_clean=clean,reference_pass=r.get('correctness',{}).get('exact_contract_pass',False)))
            if not structure or not clean:break
            if tool=='nsys':subprocess.run(['nsys','export','--type=sqlite','--force-overwrite=false','--output',str(HERE/'artifacts/nsys_adapter_a0.sqlite'),str(HERE/'artifacts/nsys_adapter_a0.nsys-rep')],check=True)
    target.write_text(json.dumps(dict(complete=len(records)==4,records=records,performance_admitted=False),indent=2)+'\n')
if __name__=='__main__':main()
