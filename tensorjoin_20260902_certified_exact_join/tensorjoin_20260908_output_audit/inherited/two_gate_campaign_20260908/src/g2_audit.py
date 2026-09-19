"""Actual compiled-code and Driver-API ABI audit; not a speed test."""
import argparse,collections,ctypes as C,hashlib,json,re,subprocess,traceback
from frozen_driver import Driver,HERE,sha

def main():
    a=argparse.ArgumentParser();a.add_argument('--label',required=True);a=a.parse_args();dest=HERE/'results'/f'{a.label}.json';assert not dest.exists()
    r=dict(pass_=False,records=[]);d=None
    try:
        d=Driver();d.bind('cuModuleUnload',[C.c_void_p]);d.bind('cuFuncGetAttribute',[C.POINTER(C.c_int),C.c_int,C.c_void_p])
        for f in sorted((HERE/'artifacts/g2_compiled_a0').glob('*.cubin')):
            md=json.loads(f.with_suffix('.json').read_text());ptx=f.with_suffix('.ptx').read_text();name=md['name']
            sass=subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','-sass',str(f)],text=True)
            resource=subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','-res-usage',str(f)],text=True)
            f.with_suffix('.sass.txt').write_text(sass);f.with_suffix('.resource.txt').write_text(resource)
            ops=collections.Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([\w.]+)',sass))
            norm=[]
            for line in sass.splitlines():
                m=re.search(r'/\*([0-9a-f]+)\*/\s+(.+?)\s*;',line)
                if m:norm.append(m[1]+' '+' '.join(m[2].split()))
            normalized=('\n'.join(norm)+'\n').encode();f.with_suffix('.selected.normalized.sass').write_bytes(normalized)
            assert md['n_spills']==0 and md['global_scratch_size']==md['profile_scratch_size']==0
            assert not any(k.split('.')[0] in ['LDL','STL'] for k in ops)
            assert 'STACK:0' in resource and 'LOCAL:0' in resource
            if f.stem.startswith('16_'):
                assert '.ftz' not in ptx and '.FTZ' not in sass and '.f64' not in ptx
                if f.stem.split('_')[1]!='1':assert any(k.startswith('HMMA') for k in ops)
                if f.stem.split('_')[1]!='0':assert ops['FADD.RP'] and ops['FADD.RM'] and ops['FMUL.RP']
            if f.stem.startswith('8_') and f.stem.split('_')[1]!='1':assert any(k.startswith('IMMA') for k in ops)
            decl=ptx.split('.visible .entry '+name+'(')[1].split(')')[0]
            kinds=re.findall(r'\.param \.(u64|u32|f32)\b',decl)
            expected=['u64']*9 if name=='half_metadata' else ['u64']*13+(['f32']*6 if name=='scan8' else [])+['u64']*2
            assert kinds==expected,(f.name,kinds);assert '.reqntid 128' in ptx
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
                assert attrs['registers']==md['n_regs'] and attrs['local_bytes']==0 and attrs['max_threads']>=128
                r['records'].append(dict(kernel=f.stem,cubin_sha256=sha(f),normalized_sass_sha256=hashlib.sha256(normalized).hexdigest(),parameters=params,attributes=attrs,opcodes=dict(ops),dynamic_shared_bytes=md['shared'],resource=resource,pass_=True))
            finally:d.check(d.lib.cuModuleUnload(mod),'unload')
        assert len(r['records'])==10,len(r['records']);r['pass']=True
    except Exception:r['pass']=False;r['exception']=traceback.format_exc();print(r['exception'],flush=True)
    finally:
        if d is not None:
            for mod in d.modules:d.check(d.lib.cuModuleUnload(mod),'unload original')
        dest.write_text(json.dumps(r,indent=2))
    print(json.dumps(dict(pass_=r['pass'],label=a.label)),flush=True);return 0 if r['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
