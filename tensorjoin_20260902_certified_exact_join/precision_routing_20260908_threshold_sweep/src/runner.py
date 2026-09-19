from common import *
import argparse,traceback

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['explore','confirm'],required=True);p.add_argument('--index',type=int,required=True);args=p.parse_args()
 index=args.index;assert 0<=index<(2 if args.phase=='explore' else 6)
 assert os.environ['CUDA_VISIBLE_DEVICES']=='2'
 result=dict(phase=args.phase,index=index,block=('explore' if args.phase=='explore' else 'primary' if index<3 else 'repeat'),started=time.time(),pid=os.getpid(),pass_=False,samples=[])
 op=None
 try:
  verify_freeze();admit=json.loads((HERE/'artifacts/admission.json').read_text());assert admit['pass']
  expected={(s['cell']['name'],s['method']):s['stage_counts'] for s in admit['samples']}
  x=full_source();op=Matrix();order=CELLS[index:]+CELLS[:index]
  if index%2:order=order[::-1]
  result['cell_order']=[c['name'] for c in order]
  warmups=1 if args.phase=='explore' else 2;reps=3 if args.phase=='explore' else 6
  for ci,cell in enumerate(order):
   op.set_cell(cell)
   for phase,num in [('warmup',warmups),('retained',reps)]:
    for iteration in range(num):
     methods=METHODS if (index+ci+iteration)%2==0 else METHODS[::-1]
     for position,method in enumerate(methods):
      a,r=op.run(method,x);h=check_output(a,cell);del a
      assert r['stage_counts']==expected[(cell['name'],method)]
      r.update(cell=cell['name'],method=method,phase=phase,iteration=iteration,position=position,output_sha256=h)
      result['samples'].append(r)
      print(json.dumps({k:r[k] for k in ['cell','method','phase','iteration','position','seconds']}),flush=True)
   op.capture()
  result.update(ended=time.time(),compiled=op.capture(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),pass_=True);result['pass']=True
  verify_freeze()
 except Exception:result['pass']=False;result['exception']=traceback.format_exc();print(result['exception'],flush=True)
 finally:
  if op is not None:op.close()
  result['ended']=time.time();write_json(HERE/'results'/f'{args.phase}_{index:02d}.json',result)
 print(json.dumps(dict(phase=args.phase,index=index,pass_=result['pass'],calls=len(result['samples']))),flush=True)
 return 0 if result['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
