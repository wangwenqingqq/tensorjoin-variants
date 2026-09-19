import argparse
import gc
import os
import traceback
from executor import *

def main():
    arg = argparse.ArgumentParser()
    arg.add_argument('--index',type=int,required=True)
    a = arg.parse_args()
    assert a.index in range(3) and os.environ['CUDA_VISIBLE_DEVICES'] == '0'
    verify_freeze()
    x = output.full_source()
    p = plan(x)
    op, radix = output.Matrix(),output.Radix()
    result = dict(index=a.index,started=time.time(),pass_=False,samples=[],single_query=[],ideal=[])
    try:
        order = list(range(6))
        order = order[a.index:]+order[:a.index]
        if a.index%2: order.reverse()
        for ci in order:
            cell = CELLS[ci]
            # Warm every configuration before any paired retained measurements.
            for method in METHODS:
                for mode in MODES:
                    rec, _ = call(op,radix,x,cell,method,mode,p)
                    rec.update(phase='warmup',repeat=-1,process=a.index)
                    result['samples'].append(rec)
            for repeat in range(4):
                methods = METHODS if (a.index+ci+repeat)%2 == 0 else METHODS[::-1]
                modes = MODES[(a.index+repeat)%3:]+MODES[:(a.index+repeat)%3]
                if repeat%2: modes = modes[::-1]
                for method in methods:
                    for mode in modes:
                        rec, _ = call(op,radix,x,cell,method,mode,p)
                        rec.update(phase='retained',repeat=repeat,process=a.index)
                        result['samples'].append(rec)
                        print(json.dumps(rec),flush=True)
            op.capture()
            gc.collect()
            torch.cuda.empty_cache()
        for ci in [0,3,5][::(-1 if a.index%2 else 1)]:
            cell = CELLS[ci]
            for method in METHODS[::(-1 if a.index%2 else 1)]:
                rec, _ = call(op,radix,x,cell,method,'geometric',p,cold=True)
                rec.update(process=a.index)
                result['single_query'].append(rec)
                print(json.dumps(rec),flush=True)
                rec, _ = call(op,radix,x,cell,method,'ideal_diagnostic',p)
                rec.update(process=a.index)
                result['ideal'].append(rec)
                print(json.dumps(rec),flush=True)
                gc.collect()
                torch.cuda.empty_cache()
        verify_freeze()
        result.update(pass_=True,ended=time.time(),compiled=op.capture())
        result['pass'] = True
        save_json(HERE/'results'/f'confirm_{a.index:02d}.json',result)
        print(json.dumps(dict(process_complete=a.index,retained=sum(r['phase']=='retained'
              for r in result['samples']),single_queries=len(result['single_query']))),flush=True)
    except Exception:
        result.update(ended=time.time(),exception=traceback.format_exc())
        result['pass'] = False
        save_json(HERE/'results'/f'confirm_{a.index:02d}_failed.json',result)
        raise

if __name__ == '__main__':
    main()
