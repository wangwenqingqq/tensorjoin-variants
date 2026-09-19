"""Make two bounded copies with explicit FP64 threshold constant transport."""
import difflib
import hashlib
import json
from pathlib import Path
H=Path(__file__).resolve().parents[1];P=H.parent
old1=(P/'g15_strong_control_20260905/src/fp32_first_kernels.py').read_text()
new1=old1.replace('row_start, column_start, threshold,','row_start, column_start, threshold: tl.constexpr,')
assert new1!=old1
new1=new1.replace('upper <= threshold','upper <= tl.full((), threshold, tl.float64)')
new1=new1.replace('lower > threshold','lower > tl.full((), threshold, tl.float64)')
s=(P/'src/run_g2a_tensorjoin.py').read_text()
old2=s[s.index('@triton.jit\ndef refine_ambiguous_fp64_i64'):s.index('\ndef parse_args()',s.index('def refine_ambiguous_fp64_i64'))]
new2=old2.replace('    threshold_d2,','    threshold_d2: tl.constexpr,')
assert new2!=old2
new2=new2.replace('distance_d2 <= threshold_d2','distance_d2 <= tl.full((), threshold_d2, tl.float64)')
out=H/'src/precision_kernels.py';assert not out.exists()
out.write_text(new1+'\n\n'+new2)
art=H/'artifacts'
(art/'precision.diff').write_text(''.join(difflib.unified_diff((old1+'\n\n'+old2).splitlines(True),out.read_text().splitlines(True),fromfile='prior functions',tofile='G19 explicit FP64 constants')))
(art/'precision_changes.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
 change='Only threshold scalar annotation and explicit FP64 constant construction. No arithmetic reduction, tile, ownership or barrier change.'),indent=2)+'\n')
