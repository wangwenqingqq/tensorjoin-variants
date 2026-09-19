import argparse
import os
import traceback
from benchmark import *

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--index',type=int,required=True)
    args=parser.parse_args()
    assert args.index in range(3) and os.environ['CUDA_VISIBLE_DEVICES']=='4'
    configure()
    verify_timing_freeze()
    x=source()
    models=selected_models()
    plans=saved_plans(x)
    op,radix=retained.output.Matrix(),retained.output.Radix()
    result=dict(index=args.index,started=time.time(),samples=[],single_queries=[],mapping=[],pass_=False)
    order=list(range(6))
    order=order[args.index:]+order[:args.index]
    if args.index%2:order.reverse()
    for ci in order:
        cell=CELLS[ci]
        for method in METHODS:
            for config in CONFIGS:
                rec,_=run(op,radix,x,cell,method,config,plans,models)
                rec.update(phase='warmup',repeat=-1,process=args.index)
                result['samples'].append(rec)
        for repeat in range(4):
            methods=METHODS if (args.index+ci+repeat)%2==0 else METHODS[::-1]
            offset=(args.index+ci+repeat)%len(CONFIGS)
            configurations=CONFIGS[offset:]+CONFIGS[:offset]
            if repeat%2:configurations=configurations[::-1]
            for method in methods:
                for config in configurations:
                    rec,_=run(op,radix,x,cell,method,config,plans,models)
                    rec.update(phase='retained',repeat=repeat,process=args.index)
                    result['samples'].append(rec)
                    print(json.dumps(rec),flush=True)
        op.capture()
        gc.collect()
        torch.cuda.empty_cache()
    for ci in [0,3,5][::(-1 if args.index%2 else 1)]:
        for method in METHODS[::(-1 if args.index%2 else 1)]:
            families=FAMILIES[args.index:]+FAMILIES[:args.index]
            for family in families:
                rec,_=run(op,radix,x,CELLS[ci],method,family+'_geo',plans,models,build=True)
                rec.update(process=args.index)
                result['single_queries'].append(rec)
                print(json.dumps(rec),flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    q=np.load(HERE/'artifacts/features_f64.npy')
    result['mapping']=mapping_benchmark(q,models)
    verify_timing_freeze()
    result.update(pass_=True,ended=time.time(),compiled=op.capture())
    result['pass']=True
    save_json(HERE/'results'/f'confirm_{args.index:02d}.json',result)
    print(json.dumps(dict(process_complete=args.index,retained=240,single_queries=18)),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        save_json(HERE/'results'/f'runner_failed_{int(time.time())}.json',dict(exception=traceback.format_exc(),time=time.time()))
        raise
