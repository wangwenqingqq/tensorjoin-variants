"""Compile only; no GPU work, no package/environment mutation."""
import hashlib,json,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
nvcc=Path('/usr/local/cuda-13.1/bin/nvcc');out=HERE/'artifacts/output.so'
assert not out.exists()
args=[str(nvcc),'-std=c++17','-O3','-arch=sm_120','--shared','-Xcompiler=-fPIC','--cudart=shared','-I/usr/local/cuda-13.1/include/cccl',str(HERE/'src/output.cu'),'-o',str(out)]
subprocess.run(args,check=True)
dep=subprocess.check_output([str(nvcc),'-std=c++17','-arch=sm_120','-I/usr/local/cuda-13.1/include/cccl','-M',str(HERE/'src/output.cu')],text=True)
(HERE/'artifacts/output.dependencies.txt').write_text(dep)
paths=[Path(x) for x in dep.replace('\\\n',' ').split(':',1)[1].split()]
record=dict(created=time.time(),command=args,compiler_version=subprocess.check_output([str(nvcc),'--version'],text=True),compiler_sha256=sha(nvcc),source_sha256=sha(HERE/'src/output.cu'),library_sha256=sha(out),dependency_sha256={str(p):sha(p) for p in paths},protocol_sha256=sha(HERE/'PROTOCOL.md'))
(HERE/'artifacts/build.json').open('x').write(json.dumps(record,indent=2))
print(json.dumps({k:v for k,v in record.items() if k!='dependency_sha256'},indent=2))
