"""Bounded unchanged-native control for cross-run point permutation identity."""
import argparse,hashlib,json,os,subprocess,sys,time,threading
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parents[1];P=H.parent;G=P/'g17_rthiss_pair_contract_20260905'
sys.path.insert(0,str(G/'src'));from validate_export import validate
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=argparse.ArgumentParser();a.add_argument('--record-id',required=True);r=a.parse_args();assert os.environ['CUDA_VISIBLE_DEVICES']=='3'
case=next(x for x in json.loads((G/'data/manifest_r1.json').read_text())['inputs'] if x['name']=='zeros31')
target=H/'raw'/r.record_id;target.mkdir(exist_ok=False)
binary=G/'build_a0/RT-HiSS';expected=json.loads((G/'results/case_native_zeros31_a0.json').read_text());assert sha(binary)==expected['binary_sha256']
cmd=[str(binary),str(P/case['path']),format(case['epsilon'],'.17g'),'highest']
with (target/'program.log').open('x') as f:
 run=subprocess.Popen(cmd,cwd=target,env=dict(os.environ,G17_N='31',G17_OUTPUT_DIR=str(target),OMP_NUM_THREADS='8'),stdout=f,stderr=subprocess.STDOUT)
 watchdog=threading.Timer(180,run.kill);watchdog.start()
 try:os.waitid(os.P_PID,run.pid,os.WEXITED|os.WNOWAIT);time.sleep(1);run.wait()
 finally:watchdog.cancel()
assert run.returncode==0
audit=validate(target,case,P);assert audit['frozen_reference_pass']
current=np.fromfile(target/'point_map.u32',dtype='<u4');old=np.fromfile(G/'raw/native_zeros31_a0/point_map.u32',dtype='<u4');repaired=np.fromfile(H/'raw/correct_zeros31_g3_a2/point_map.u32',dtype='<u4')
x=np.fromfile(P/case['path'],dtype='<f4').reshape(31,512)
assert np.array_equal(x[current].view('<u4'),x[old].view('<u4')) and np.array_equal(x[current].view('<u4'),x[repaired].view('<u4'))
record=dict(record_id=r.record_id,command=cmd,binary_sha256=sha(binary),libowl_sha256=sha(G/'build_a0/OWL/owl/libowl.so'),audit=audit,
 same_point_map_as_g17_gpu2=bool(np.array_equal(current,old)),same_point_map_as_g18_gpu3=bool(np.array_equal(current,repaired)),
 old_map=old.tolist(),current_map=current.tolist(),repaired_map=repaired.tolist(),all_reordered_value_bytes_identical=True,
 correctness=dict(exact_contract_pass=True,meaning='unchanged native zeros31 map/ID diagnostic, not repaired gate'),public_seconds=None)
(H/'results'/f'{r.record_id}.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
