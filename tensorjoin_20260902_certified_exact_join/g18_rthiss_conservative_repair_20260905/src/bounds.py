"""Exact rational error constants and outward FP32 cutoffs for G18."""
from fractions import Fraction as F
import numpy as np
D=512
u32=F(1,2**24);u64=F(1,2**53)
g=(2*D+4)*u32/(1-(2*D+4)*u32)
h=(3*D+4)*u64/(1-(3*D+4)*u64)
B=D*F(1,2**140)
def outward(q,up):
 v=np.float32(float(q));assert np.isfinite(v)
 step=np.float32(np.inf if up else -np.inf)
 while (F(float(v))<q if up else F(float(v))>q):v=np.nextafter(v,step,dtype=np.float32)
 assert F(float(v))>=q if up else F(float(v))<=q
 return v

def cuts(t):
 t=F(float(t));assert 0<=t<=2048
 high=B+(1+g)*t/(1-h);low=(1-g)*t/(1+h)-B
 return outward(low,False),outward(high,True)
