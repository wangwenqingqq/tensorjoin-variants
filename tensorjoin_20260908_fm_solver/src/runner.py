import argparse
import os
import traceback
from engine import *


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--index',type=int,required=True);args=parser.parse_args()
    assert args.index in range(3) and os.environ['CUDA_VISIBLE_DEVICES']=='4';configure();verify_timing_freeze()
    x=source();plan=saved_plan(x);models=selected_models()
    expected=dict(np.load(HERE/'artifacts/admitted_masks.npz'))
    op,radix=retained.output.Matrix(),retained.output.Radix()
    result=dict(index=args.index,started=time.time(),samples=[],single_queries=[],mapping=[],pass_=False)
    order=list(range(6));order=order[args.index:]+order[:args.index]
    if args.index%2:order.reverse()
    for ci in order:
        cell=CELLS[ci]
        for method in METHODS:
            for config in CONFIGS:
                rec,_,_,mask=run(op,radix,x,cell,method,config,plan,models)
                assert np.array_equal(mask,expected[config+'_'+cell['name']])
                rec.update(phase='warmup',repeat=-1,process=args.index);result['samples'].append(rec)
        for repeat in range(4):
            methods=METHODS if (args.index+ci+repeat)%2==0 else METHODS[::-1]
            offset=(args.index+ci+repeat)%len(CONFIGS);configs=CONFIGS[offset:]+CONFIGS[:offset]
            if repeat%2:configs=configs[::-1]
            for method in methods:
                for config in configs:
                    rec,_,_,mask=run(op,radix,x,cell,method,config,plan,models)
                    assert np.array_equal(mask,expected[config+'_'+cell['name']])
                    rec.update(phase='retained',repeat=repeat,process=args.index);result['samples'].append(rec)
                    print(json.dumps(rec),flush=True)
        op.capture();gc.collect();torch.cuda.empty_cache()
    build_configs=['pca','all','heuristic25','direct25','fm25']
    for ci in [0,3,5][::(-1 if args.index%2 else 1)]:
        for method in METHODS[::(-1 if args.index%2 else 1)]:
            configs=build_configs[args.index:]+build_configs[:args.index]
            for config in configs:
                rec,_,_,mask=run(op,radix,x,CELLS[ci],method,config,plan,models,build=True)
                assert np.array_equal(mask,expected[config+'_'+CELLS[ci]['name']])
                rec.update(process=args.index);result['single_queries'].append(rec);print(json.dumps(rec),flush=True)
            gc.collect();torch.cuda.empty_cache()
    result['mapping']=mapping_benchmark(plan,models);verify_timing_freeze()
    result.update(pass_=True,ended=time.time(),compiled=op.capture());result['pass']=True
    save_json(HERE/'results'/f'confirm_{args.index:02d}.json',result)
    print(json.dumps(dict(process_complete=args.index,retained=384,build_queries=30)),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        save_json(HERE/'results'/f'runner_failed_{int(time.time())}.json',dict(exception=traceback.format_exc(),time=time.time()))
        raise
