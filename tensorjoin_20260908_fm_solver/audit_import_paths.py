"""Explicit entry-file import audit without changing any frozen solver inputs."""
import json
import os
from pathlib import Path
import runpy
import sys
import time

HERE=Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(HERE/'src'))
records={}
for name in ['admit','runner']:
    path=HERE/'src'/(name+'.py')
    module=runpy.run_path(str(path),run_name='explicit_import_audit_'+name)
    assert Path(module['main'].__code__.co_filename)==path
    records[name]=dict(entry_file=module['main'].__code__.co_filename,
        engine_run_file=module['run'].__code__.co_filename)
    assert Path(module['run'].__code__.co_filename)==HERE/'src/engine.py'
    if name=='admit':
        assert Path(module['prediction_metrics'].__code__.co_filename)==HERE/'src/train.py'
import torch
assert not torch.cuda.is_initialized()
record=dict(pass_=True,time=time.time(),entries=records,cuda_initialized=False,
    note='Initial imports.json verified the engine/output import, but bare admit/runner names resolved to inherited modules after sys.path changes. Actual workers execute explicit current entry paths. This audit checks those same entry files explicitly without running main or changing frozen sources.')
with (HERE/'analysis/import_paths.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
