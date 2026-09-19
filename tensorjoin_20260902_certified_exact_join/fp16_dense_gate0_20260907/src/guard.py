"""Exclusive bounded run; never terminate a foreign process."""
import argparse,fcntl,hashlib,json,os,platform,signal,subprocess,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]

def query(args):
    p=subprocess.run(['nvidia-smi']+args,text=True,capture_output=True,timeout=15,check=True)
    return p.stdout.strip()
def rows(uuid):
    lines=query(['--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits']).splitlines()
    return [x.split(', ') for x in lines if x.split(', ')[0]==uuid]
def state(gpu):
    return query(['-i',str(gpu),'--query-gpu=index,uuid,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu','--format=csv,noheader,nounits'])
def ours(pid,root):
    seen=set()
    while pid>1 and pid not in seen:
        if pid==root:return True
        seen.add(pid)
        try:pid=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[1])
        except (OSError,ValueError,IndexError):return False
    return False

def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--gpu',type=int,default=2);p.add_argument('--target',choices=['8p_gpu2'],default='8p_gpu2');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    target=json.loads((HERE/'targets.json').read_text())[a.target]
    assert a.label.replace('_','').isalnum() and a.gpu==target['gpu'] and platform.node()==target['host']
    cmd=a.command[1:] if a.command[0]=='--' else a.command
    for x in [HERE/'raw'/f'{a.label}.log',HERE/'raw'/f'{a.label}_occupancy.jsonl',HERE/'results'/f'{a.label}_guard.json']:
        assert not x.exists(),x
    result={'label':a.label,'command':cmd,'pid':None,'pass':False,'started':time.time(),'target':target,'target_config_sha256':hashlib.sha256((HERE/'targets.json').read_bytes()).hexdigest(),'source_sha256':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in (HERE/'src').glob('*.py')}}
    process=None; locks=[]
    try:
        for name in [f'/tmp/tensorjoin_gpu{a.gpu}_campaign.lock',f'/tmp/tensorjoin_g5_gpu{a.gpu}.lock']:
            h=open(name,'a+');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(h)
        uuid=query(['-i',str(a.gpu),'--query-gpu=uuid','--format=csv,noheader']);result['gpu_uuid']=uuid
        assert uuid==target['uuid']
        with (HERE/'raw'/f'{a.label}_occupancy.jsonl').open('x') as mon, (HERE/'raw'/f'{a.label}.log').open('x') as log:
            def sample(phase):
                active=rows(uuid); st=state(a.gpu)
                mon.write(json.dumps({'time':time.time(),'phase':phase,'gpu':st,'processes':active})+'\n');mon.flush()
                return active,st
            t=time.monotonic()
            while True:
                active,_=sample('quiescence');assert not active,active
                if time.monotonic()-t>=30:break
                time.sleep(1)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(a.gpu),TENSORJOIN_TARGET=a.target,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            log.write('COMMAND '+json.dumps(cmd)+'\n');log.flush()
            process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=HERE)
            result['pid']=process.pid
            print(json.dumps({'launched':cmd,'pid':process.pid,'log':str(log.name)}),flush=True)
            t=time.monotonic();maxmem=0
            while process.poll() is None:
                active,st=sample('running')
                foreign=[x for x in active if not ours(int(x[1]),process.pid)]
                assert not foreign,('foreign occupancy',foreign)
                maxmem=max(maxmem,int(st.split(', ')[7]));assert maxmem<4096,('memory budget MiB',maxmem)
                assert time.monotonic()-t<1800,'timeout'
                time.sleep(.5)
            result.update(exit_code=process.returncode,pass_=process.returncode==0,max_device_used_mib=maxmem)
            result['pass']=process.returncode==0
    except Exception:
        result['exception']=traceback.format_exc();print(result['exception'],flush=True)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
        result['ended']=time.time()
        with (HERE/'results'/f'{a.label}_guard.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
    return 0 if result['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
