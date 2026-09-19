"""Bind observed CUDA dispatch to native/adapter selected instruction streams."""
import collections,hashlib,json,re,sqlite3,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];PROJECT=HERE.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def functions(text):
    result={}
    for block in re.split(r'\s*Function : ',text)[1:]:
        name=block.splitlines()[0].strip();instructions=[]
        for line in block.splitlines()[1:]:
            m=re.search(r'/\*[0-9a-fA-F]+\*/\s*(.*?)\s*;',line)
            if m:instructions.append(' '.join(m.group(1).split())+';')
        if instructions:result[name]=instructions
    return result

def main():
    assert json.loads((HERE/'results/extension_a0.json').read_text())['complete']
    db=sqlite3.connect(HERE/'artifacts/nsys_adapter_a0.sqlite')
    launches=db.execute('''select s.value,k.registersPerThread,k.gridX,k.blockX,k.staticSharedMemory,k.dynamicSharedMemory,
        k.localMemoryPerThread,count(*) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName
        group by s.value,k.registersPerThread,k.gridX,k.blockX,k.staticSharedMemory,k.dynamicSharedMemory,k.localMemoryPerThread''').fetchall()
    names=[row[0] for row in launches]
    targets=['identifyNeighborsGridPrimitiveSharedQueryShared','compressResultMask']
    for t in targets:assert len({name for name in names if t in name})==1,(t,names)
    all_functions={};containers={}
    for variant,directory in [('native','build_upstream_d512_a0'),('adapter','build_a0')]:
        binary=HERE/directory/'RT-HiSS';containers[variant]=dict(binary_sha256=sha(binary),libowl_sha256=sha(HERE/directory/'OWL/owl/libowl.so'))
        text=subprocess.check_output(['cuobjdump','--dump-sass',str(binary)],text=True)
        out=HERE/f'raw/{variant}_full.sass';assert not out.exists();out.write_text(text)
        all_functions[variant]=functions(text)
    selected=[]
    for target in targets:
        signatures=[]
        for variant in ['native','adapter']:
            match=[(k,v) for k,v in all_functions[variant].items() if target in k];assert len(match)==1
            symbol,instructions=match[0];text='\n'.join(instructions)+'\n';h=hashlib.sha256(text.encode()).hexdigest();signatures.append(h)
            opcodes=collections.Counter(re.sub(r'^@!?\w+\s+','',s).split()[0] for s in instructions)
            selected.append(dict(variant=variant,target=target,symbol=symbol,instruction_count=len(instructions),normalized_sass_sha256=h,opcodes=dict(opcodes)))
        assert signatures[0]==signatures[1],target
    result=dict(cuda_selected_code_identity_pass=True,containers=containers,selected=selected,nsys_launches=launches,
        nsys_sqlite_sha256=sha(HERE/'artifacts/nsys_adapter_a0.sqlite'),
        normalization='Remove cuobjdump address/encoding/metadata; retain instruction predicates, operands, constants, reuse flags and branch offsets; collapse whitespace only',
        scope='runtime-observed CUDA refinement/compression symbols matched to static functions in loaded adapter executable; native source/control build has same selected SASS',
        limit='OptiX driver-JIT identity not established; unchanged RT source recorded separately; not latency or all-input numerical proof')
    (HERE/'results/code_audit_a0.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
