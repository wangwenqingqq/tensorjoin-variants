"""Cooperative GPU lock and bounded ownership supervision; never stop foreign work."""
import argparse, fcntl, json, os, signal, subprocess, time
from pathlib import Path


def query(args):
    return subprocess.check_output(['nvidia-smi']+args,text=True,timeout=15).strip()


def descendant(pid,root):
    seen=set()
    while pid>1 and pid not in seen:
        if pid==root:return True
        seen.add(pid)
        try:pid=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[1])
        except (FileNotFoundError,ValueError,IndexError):return False
    return False


def main():
    a=argparse.ArgumentParser();a.add_argument('--gpu',type=int,default=1)
    a.add_argument('--label',required=True);a.add_argument('--timeout',type=int,default=1800)
    a.add_argument('command',nargs=argparse.REMAINDER);a=a.parse_args()
    cmd=a.command[1:] if a.command[0]=='--' else a.command
    root=Path(__file__).resolve().parents[1];(root/'raw').mkdir(exist_ok=True);(root/'results').mkdir(exist_ok=True)
    locks=[]
    for name in [f'/tmp/tensorjoin_gpu{a.gpu}_campaign.lock',f'/tmp/tensorjoin_g5_gpu{a.gpu}.lock',str(root.parent/f'.tensorjoin_gpu{a.gpu}_campaign.lock')]:
        f=open(name,'a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    uuid=query(['-i',str(a.gpu),'--query-gpu=uuid','--format=csv,noheader'])
    p=None;result={'gpu':a.gpu,'start':time.time(),'pass':False};seen={}
    with (root/'raw'/f'{a.label}.log').open('x') as log,(root/'raw'/f'{a.label}.occupancy.jsonl').open('x') as mon:
        def sample():
            state=query(['-i',str(a.gpu),'--query-gpu=index,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu','--format=csv,noheader,nounits'])
            active=[r.split(', ') for r in query(['--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits']).splitlines() if r.startswith(uuid)]
            mon.write(json.dumps({'time':time.time(),'state':state,'processes':active})+'\n');mon.flush()
            return active,state
        try:
            for _ in range(5):
                active,state=sample();assert not active,('GPU busy',active);time.sleep(1)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=uuid,TRITON_PTXAS_PATH='/usr/local/cuda-13.1/bin/ptxas',TRITON_PTXAS_BLACKWELL_PATH='/usr/local/cuda-13.1/bin/ptxas',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',TRITON_CACHE_DIR=str(root/'artifacts'/'triton_cache'),TENSORJOIN_ARTIFACTS=str(root/'artifacts'/'native'))
            p=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            result['pid']=p.pid;print(json.dumps({'pid':p.pid,'label':a.label}),flush=True);start=time.monotonic()
            while p.poll() is None:
                active,state=sample()
                for row in active:
                    pid=int(row[1]);stat=Path(f'/proc/{pid}/stat')
                    if stat.exists():
                        ticks=stat.read_text().rsplit(')',1)[1].split()[19]
                        assert descendant(pid,p.pid) and (pid not in seen or seen[pid][0]==ticks),('foreign process',row)
                        seen[pid]=(ticks,time.monotonic())
                    else:
                        assert pid in seen and row[2]=='[No data]' and time.monotonic()-seen[pid][1]<2,('unowned PID',row)
                assert int(state.split(', ')[6])<12000,'12 GiB memory budget exceeded'
                assert time.monotonic()-start<a.timeout,'timeout'
                time.sleep(.75)
            result['exit_code']=p.returncode;result['pass']=p.returncode==0
        except Exception as exc:
            result['error']=repr(exc)
        finally:
            if p is not None and p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
            result['end']=time.time()
            (root/'results'/f'{a.label}.guard.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    return 0 if result['pass'] else 1

if __name__=='__main__':raise SystemExit(main())
