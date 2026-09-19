"""Fail-closed SM120 E4M3-to-E3M4 patch on owned, not shared-cache, bytes."""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

CUOBJDUMP=os.environ.get('CUOBJDUMP','/usr/local/cuda-13.1/bin/cuobjdump')
ART=Path(os.environ.get('TENSORJOIN_ARTIFACTS','artifacts'))


def sha(b):
    return hashlib.sha256(b).hexdigest()


def inspect_binary(binary):
    with tempfile.TemporaryDirectory() as temp:
        p=Path(temp)/'kernel.cubin'; p.write_bytes(binary)
        sec=subprocess.check_output(['readelf','-SW',str(p)],text=True)
        sections=re.findall(r'\.text\.\S+\s+PROGBITS\s+[0-9a-fA-F]+\s+([0-9a-fA-F]+)',sec)
        assert len(sections)==1,sections
        offset=int(sections[0],16)
        sass=subprocess.check_output([CUOBJDUMP,'-sass',str(p)],text=True)
    addresses=[]
    for line in sass.splitlines():
        if re.search(r'\bQMMA\.',line):
            assert '16832' in line and '.F32' in line and '.SF' not in line,line
            m=re.search(r'/\*([0-9a-fA-F]+)\*/',line); assert m,line
            addresses.append(int(m[1],16))
    assert addresses,'No plain QMMA.16832 found; do not patch another instruction family.'
    return offset,addresses,sass


def field_words(binary):
    offset,addresses,sass=inspect_binary(binary)
    return [int.from_bytes(binary[offset+a:offset+a+16],'little') for a in addresses]


def patch(kernel):
    assert kernel.metadata.target.arch==120,kernel.metadata.target
    binary=kernel.kernel; offset,addresses,sass=inspect_binary(binary)
    buf=bytearray(binary); changed=set()
    for a in addresses:
        w=int.from_bytes(binary[offset+a:offset+a+16],'little')
        # Relative to freshly compiled E4M3, not an absolute selector number.
        for bit in (82,84):
            pos=offset+a+bit//8; buf[pos]^=1<<(bit%8); changed.add(pos)
        assert w ^ int.from_bytes(buf[offset+a:offset+a+16],'little') == (1<<82)|(1<<84)
    observed={i for i,(a,b) in enumerate(zip(binary,buf)) if a!=b}
    assert observed==changed and len(buf)==len(binary)
    clone=copy.copy(kernel); clone.asm=dict(kernel.asm); clone.kernel=bytes(buf)
    clone.asm['cubin']=bytes(buf); clone.module=None; clone.function=None; clone._run=None
    ART.mkdir(parents=True,exist_ok=True)
    key=sha(binary); (ART/(key+'.original.cubin')).write_bytes(binary)
    (ART/(key+'.e3m4.cubin')).write_bytes(buf); (ART/(key+'.sass')).write_text(sass)
    report={'original_sha256':key,'patched_sha256':sha(buf),'instruction_count':len(addresses),
            'allowed_xor_bits':[82,84],'changed_byte_count':len(changed),
            'metadata':kernel.metadata._asdict(),'scope':'plain QMMA.16832 F32, SM120 only'}
    (ART/(key+'.patch.json')).write_text(json.dumps(report,indent=2,default=str))
    return clone
