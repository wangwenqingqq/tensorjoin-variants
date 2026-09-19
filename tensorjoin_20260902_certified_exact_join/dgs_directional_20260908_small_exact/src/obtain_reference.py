"""Regenerate admitted complete IDs using an unchanged existing GPU operator."""
import hashlib,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
OLD=HERE.parent/'two_gate_campaign_20260908'
sys.path.insert(0,str(OLD/'src'))
from g2_operator_r1 import Matrix
from retained_ops import full_source,output_check
result={'mode':'reference regeneration, not paired latency','pid':os.getpid(),'started':time.time(),'pass':False}
try:
    x=full_source();op=Matrix();a,record=op.run('F16',x)
    result['output_sha256']=output_check(a);result['count']=len(a)
    result['record']=record;result['retained_programs']=op.capture()
    p=HERE/'artifacts/reference_ids_u64.bin'
    with p.open('xb') as f:a.astype('<u8',copy=False).tofile(f)
    result['file_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    assert result['file_sha256']==result['output_sha256']
    result['pass']=True
finally:
    result['ended']=time.time()
    (HERE/'results/reference_a0.json').open('x').write(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
