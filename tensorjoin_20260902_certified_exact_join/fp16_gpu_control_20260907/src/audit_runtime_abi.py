"""Query actual CUDA ABI and resources for the already captured cubins."""
import argparse,ctypes as C,json,os,re,traceback
import torch
from frozen_driver import Driver,HERE,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);a=p.parse_args();dest=HERE/'results'/f'{a.label}.json';assert not dest.exists()
    r={'pass':False,'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'records':[]};d=None
    try:
        d=Driver();d.bind('cuModuleUnload',[C.c_void_p]);d.bind('cuFuncGetAttribute',[C.POINTER(C.c_int),C.c_int,C.c_void_p])
        for f in sorted((HERE/'artifacts/compiled').glob('*.cubin')):
            md=json.loads(f.with_suffix('.json').read_text());ptx=f.with_suffix('.ptx').read_text()
            name=md['name'];decl=ptx.split('.visible .entry '+name+'(')[1].split(')')[0]
            kinds=re.findall(r'\.param \.(u64|u32)\b',decl)
            expected=['u64']*9 if name=='half_metadata' else ['u64']*10+['u32']*2+['u64']*2
            assert kinds==expected and f'.reqntid {32*md["num_warps"]}' in ptx
            mod,fun=C.c_void_p(),C.c_void_p();d.check(d.lib.cuModuleLoad(C.byref(mod),str(f).encode()),'load')
            try:
                d.check(d.lib.cuModuleGetFunction(C.byref(fun),mod,name.encode()),'function');offset=0;params=[]
                for i,kind in enumerate(kinds):
                    size=8 if kind=='u64' else 4;offset=(offset+size-1)//size*size;off,got=C.c_size_t(),C.c_size_t()
                    d.check(d.lib.cuFuncGetParamInfo(fun,i,C.byref(off),C.byref(got)),'parameter');assert (off.value,got.value)==(offset,size)
                    params.append([off.value,got.value]);offset+=size
                attrs={}
                for key,code in [('max_threads',0),('static_shared_bytes',1),('local_bytes',3),('registers',4)]:
                    value=C.c_int();d.check(d.lib.cuFuncGetAttribute(C.byref(value),code,fun),'attribute');attrs[key]=value.value
                assert attrs['registers']==md['n_regs'] and attrs['local_bytes']==0 and attrs['max_threads']>=32*md['num_warps']
                r['records'].append({'kernel':f.stem,'cubin_sha256':sha(f),'parameters':params,'attributes':attrs,'dynamic_shared_bytes':md['shared'],'pass':True})
            finally:d.check(d.lib.cuModuleUnload(mod),'unload')
        assert len(r['records'])==9;r['pass']=True
    except Exception:r['exception']=traceback.format_exc();print(r['exception'],flush=True)
    finally:
        if d is not None:
            for mod in d.modules:d.check(d.lib.cuModuleUnload(mod),'original unload')
        d=None;torch.cuda.empty_cache()
        with dest.open('x') as h:json.dump(r,h,indent=2)
    print(json.dumps({'pass':r['pass'],'path':str(dest)}),flush=True)
    return 0 if r['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
