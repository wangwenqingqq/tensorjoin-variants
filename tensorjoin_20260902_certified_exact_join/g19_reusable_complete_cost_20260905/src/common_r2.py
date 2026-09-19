"""Additive R2 closing-only source and bridge gate."""
from common import *
def check_r2():
    for path,value in json.loads((H/'artifacts/frozen_r2.json').read_text()).items():assert sha(P/path)==value,path
