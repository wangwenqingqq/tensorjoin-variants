from common import *
from common_r1 import check_r1
from common_r2 import check_r2
check_frozen();check_r1();check_r2()
files=[H/'ADDENDUM_OWNED_CUBLAS_R3.md',H/'artifacts/owned_cublas_r3.diff']
files += [H/'src'/n for n in ['prepare_r3.py','owned_cublas_r3.py','operators_r3.py','run_slot_r3.py','supervise_r3.py','common_r3.py','freeze_r3.py','admit_r3.py']]
write(H/'artifacts/frozen_r3.json',{str(p.relative_to(P)):sha(p) for p in files})
print('FROZEN_R3',len(files))
