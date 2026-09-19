"""Additive immutable R1 execution gate."""
from common import *
def check_r1():
    for path,value in json.loads((H/'artifacts/frozen_r1.json').read_text()).items():
        assert sha(P/path)==value,path
