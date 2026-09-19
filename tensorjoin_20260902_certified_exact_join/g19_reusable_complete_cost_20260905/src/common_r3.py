from common import *
def check_r3():
    for path,value in json.loads((H/'artifacts/frozen_r3.json').read_text()).items():assert sha(P/path)==value,path
