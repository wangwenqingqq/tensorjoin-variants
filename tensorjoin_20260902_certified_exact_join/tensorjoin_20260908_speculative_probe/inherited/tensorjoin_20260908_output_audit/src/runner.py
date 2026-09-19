from common import *
import argparse,traceback
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['explore','confirm'],required=True);p.add_argument('--index',type=int,required=True);args=p.parse_args();index=args.index
 assert 0<=index<(2 if args.phase=='explore' else 6) and os.environ['CUDA_VISIBLE_DEVICES']=='2'
 result=dict(phase=args.phase,index=index,block=('explore' if args.phase=='explore' else 'primary' if index<3 else 'repeat'),pid=os.getpid(),started=time.time(),pass_=False,samples=[])
 try:
  verify_freeze();admit=json.loads((HERE/'artifacts/admission.json').read_text());assert admit['pass']
  expected={(r['cell'],r['method']):r['stage_counts'] for r in admit['samples']}
  x=full_source();tc=Matrix();fp=FP32();radix=Radix();order=CELLS[index:]+CELLS[:index]
  if index%2:order=order[::-1]
  result['cell_order']=[c['name'] for c in order]
  if args.phase=='confirm':selection=json.loads((HERE/'artifacts/selection.json').read_text());result['selection']=selection;outs=['legacy',selection['challenger']]
  def one(cell,method,mode,phase,rep,position):
   rec=call(fp if method=='F32' else tc,radix,cell,method,mode,x);assert rec['stage_counts']==expected[cell['name'],method]
   rec.update(phase=phase,iteration=rep,position=position);result['samples'].append(rec)
   print(json.dumps({k:rec[k] for k in ['cell','method','output_mode','phase','iteration','position','seconds','peak_reserved_bytes']}),flush=True)
  for ci,cell in enumerate(order):
   if args.phase=='explore':
    combinations=[(m,o) for m in METHODS for o in OUTPUTS]
    for phase,reps in [('warmup',1),('retained',2)]:
     for rep in range(reps):
      shift=(index*3+ci+rep)%9;comb=combinations[shift:]+combinations[:shift]
      if (index+rep)%2:comb=comb[::-1]
      for pos,(m,o) in enumerate(comb):one(cell,m,o,phase,rep,pos)
   else:
    shift=(index+ci)%3;methods=METHODS[shift:]+METHODS[:shift]
    if index%2:methods=methods[::-1]
    for mi,m in enumerate(methods):
     for phase,reps in [('warmup',1),('retained',4)]:
      for rep in range(reps):
       modes=outs if (index+ci+mi+rep)%2==0 else outs[::-1]
       for pos,o in enumerate(modes):one(cell,m,o,phase,rep,pos)
   tc.capture();gc.collect();torch.cuda.empty_cache()
  result.update(compiled=tc.capture(),fp32_library=fp.library_record,cub_version=radix.version,pass_=True);result['pass']=True;verify_freeze()
 except Exception:result['pass']=False;result['exception']=traceback.format_exc();print(result['exception'],flush=True)
 result['ended']=time.time();write_json(HERE/'results'/f'{args.phase}_{index:02d}.json',result)
 print(json.dumps(dict(phase=args.phase,index=index,pass_=result['pass'],calls=len(result['samples']))),flush=True)
 return 0 if result['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
