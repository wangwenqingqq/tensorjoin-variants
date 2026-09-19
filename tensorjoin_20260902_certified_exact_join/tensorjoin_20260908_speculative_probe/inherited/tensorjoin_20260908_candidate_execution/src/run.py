import argparse,gc,hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from engine import *
def write(p,r):
    with Path(p).open('x') as f:json.dump(r,f,indent=2)
def freeze():
    paths=list((HERE/'src').glob('*.py'))+[HERE/'PROTOCOL.md',HERE/'targets_r2.json']
    paths+=list((HERE/'artifacts').glob('*.npy'))+[HERE/'artifacts/preparation.json']
    paths+=list(Path('@TENSORJOIN_ROOT@/tensorjoin_20260908_block_feasibility/src').glob('*.py'))
    deps=json.loads((ROOT/'tensorjoin_20260908_output_audit/artifacts/timing_freeze.json').read_text())
    deps={str(ROOT/p):h for p,h in deps.items()}
    deps.update({str(p):output.sha(p) for p in paths})
    write(HERE/'artifacts/timing_freeze.json',deps)
def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--admit',action='store_true');p.add_argument('--process',type=int,default=0);args=p.parse_args()
    torch.set_num_threads(4)
    x,a,r=load();op=output.Matrix();radix=output.Radix()
    rec=dict(label=args.label,started=time.time(),samples=[],diagnostics=[],warmups=[],pass_=False)
    try:
        if not args.admit:
            for path,h in json.loads((HERE/'artifacts/timing_freeze.json').read_text()).items():assert output.sha(Path(path))==h,path
        configs=[(cell,m) for cell in output.CELLS for m in METHODS]
        for cell,m in configs:
            s=call(op,radix,x,a,r,cell,m,exact=args.admit)
            rec['warmups'].append(s);print(json.dumps(dict(warmup=True,**s)),flush=True)
            gc.collect();torch.cuda.empty_cache()
        if args.admit:
            for cell,m in configs:
                s=call(op,radix,x,a,r,cell,m,exact=True,diagnostic=True)
                rec['diagnostics'].append(s);print(json.dumps(s),flush=True)
                gc.collect();torch.cuda.empty_cache()
            for m in METHODS[1:]:
                cell=next(c for c in output.CELLS if c['name']=='original')
                s=call(op,radix,x,a,r,cell,m,exact=True,cold=True)
                rec['diagnostics'].append(s);print(json.dumps(s),flush=True)
                gc.collect();torch.cuda.empty_cache()
            rec['compiled_new']=capture();rec['compiled_inherited']=op.capture()
            freeze()
        else:
            for rep in range(3):
                seq=configs if (args.process+rep)%2==0 else list(reversed(configs))
                shift=(args.process*7+rep*5)%len(seq);seq=seq[shift:]+seq[:shift]
                for cell,m in seq:
                    s=call(op,radix,x,a,r,cell,m);s.update(repetition=rep,process=args.process)
                    rec['samples'].append(s);print(json.dumps(s),flush=True)
                    gc.collect();torch.cuda.empty_cache()
            rec['compiled_inherited']=op.capture()
        rec.update(pass_=True,ended=time.time(),torch=torch.__version__,triton=triton.__version__)
    except Exception:
        rec['exception']=traceback.format_exc();raise
    finally:write(HERE/'results'/f'{args.label}.json',rec)
if __name__=='__main__':main()
