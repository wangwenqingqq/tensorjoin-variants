"""Own-process deallocation API A/B localization; not a safety/performance pass."""
import os
import subprocess
import time
from common import *
from common_r1 import check_r1
check_frozen();check_r1()
log=H/'raw/allocator_probe_r1.log'
with log.open('x') as f:
    p=subprocess.Popen([str(H/'artifacts/allocator_probe')],stdout=f,stderr=subprocess.STDOUT)
    os.waitid(os.P_PID,p.pid,os.WEXITED|os.WNOWAIT);time.sleep(1);p.wait()
value=json.loads(log.read_text())
passed=p.returncode==0 and value['wrong_free']==1 and value['correct_free']==0
write(H/'results/allocator_probe_r1.json',dict(observed=value,returncode=p.returncode,
    correctness=dict(exact_contract_pass=passed),scope='Intentional rejected-API diagnostic, not sanitizer admission',
    binary_sha256=sha(H/'artifacts/allocator_probe'),performance_admitted=False))
raise SystemExit(0 if passed else 2)
