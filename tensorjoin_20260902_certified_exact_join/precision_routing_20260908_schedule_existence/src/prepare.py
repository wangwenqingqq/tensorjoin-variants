"""Freeze diagnostic schedules from existing original-order census; no timing."""
import json,time
import numpy as np
from common import HERE,ROOT,sha
source=ROOT/'precision_routing_20260908_layout_existence/artifacts/original_F8_census.npz'
h=np.load(source)['hist1'];o=np.arange(len(h),dtype=np.int64);c=np.argsort(-h,kind='stable')
groups=np.array_split(c,4096);interleaved=np.array([g[j] for j in range(108) for g in groups if len(g)>j],dtype=np.int64)
assert np.array_equal(np.sort(interleaved),o)
np.savez(HERE/'artifacts/schedules.npz',original=o,clustered=c,interleaved=interleaved)
(HERE/'artifacts/schedule_provenance.json').open('x').write(json.dumps({'time':time.time(),'source':str(source),'source_sha256':sha(source),'schedules_sha256':sha(HERE/'artifacts/schedules.npz'),'uses_oracle_information':True,'included_in_operator_cost':False},indent=2))
