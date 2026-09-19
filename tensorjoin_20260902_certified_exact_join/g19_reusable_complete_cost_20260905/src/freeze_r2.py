"""Freeze the explicit allocator-shutdown layer before renewed testing."""
from common import *
from common_r1 import check_r1
check_frozen();check_r1()
files=[H/'ADDENDUM_TORCH_SHUTDOWN_R2.md',H/'artifacts/host_allocator.so',H/'artifacts/host_allocator_build.json']
files += [H/'src'/n for n in ['host_allocator.cpp','build_host_allocator.py','prepare_r2.py','operators_r2.py',
          'run_slot_r2.py','supervise_r2.py','common_r2.py','freeze_r2.py','admit_r2.py']]
write(H/'artifacts/frozen_r2.json',{str(p.relative_to(P)):sha(p) for p in files})
print('FROZEN_R2',len(files))
