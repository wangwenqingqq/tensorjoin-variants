import os
from problem import *


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    start=time.perf_counter();x=source()
    p,q,score,scale2,feature_time=make_features(x)
    old=Path('@TENSORJOIN_ROOT@/tensorjoin_20260908_fm_control/artifacts')
    assert digest(q)==json.loads((old/'features.json').read_text())['features_sha256']
    perm=np.argsort(score,kind='stable').astype(np.int64)
    assert np.array_equal(perm,np.load(old/'pca_permutation.npy'))
    b,btime=projected_bounds(q,perm,scale2)
    assert np.array_equal(b['combined'],np.load(old/'pca_bounds.npz')['combined'])
    features,coarse,stime=summaries(x,q,perm,b)
    # Last, ragged block stays in test. All other split blocks have 64 rows.
    order=np.random.default_rng(2026090901).permutation(NB-1)
    groups=dict(train=order[:625],validation=order[625:781],test=np.r_[order[781:],NB-1])
    split={}
    for name,blocks in groups.items():
        member=np.zeros(NB,bool);member[blocks]=True
        split[name]=member[b['tr']] & member[b['tc']]
    assert all(not np.any(split[a]&split[c]) for a,c in [('train','validation'),('train','test'),('validation','test')])
    mu=features[split['train']].astype(np.float64).mean(0)
    sd=features[split['train']].astype(np.float64).std(0);sd=np.maximum(sd,1e-5)
    thresholds=np.asarray([c['T'] for c in CELLS],np.float64)
    normalizer=dict(mean=mu.tolist(),scale=sd.tolist(),threshold_mean=float(thresholds.mean()),threshold_scale=float(thresholds.std()))
    label_start=time.perf_counter()
    counts=exact_counts(perm,split['train']|split['validation'])
    label_seconds=time.perf_counter()-label_start
    assert np.all(counts[:,~(split['train']|split['validation'])]==-1)
    for name,a in [('projector',p),('features_f64',q),('pca_scores_f64',score),('permutation',perm),('pair_features',features),('coarse_state',coarse),('counts_train_validation',counts)]:
        np.save(HERE/'artifacts'/f'{name}.npy',a,allow_pickle=False)
    np.savez_compressed(HERE/'artifacts/bounds.npz',**b)
    np.savez_compressed(HERE/'artifacts/block_splits.npz',**groups)
    np.savez_compressed(HERE/'artifacts/pair_splits.npz',**split)
    save_json(HERE/'artifacts/normalizer.json',normalizer)
    margin=numerical_margin(q)
    save_json(HERE/'artifacts/numerical_margin.json',margin)
    meta=dict(pass_=True,created=time.time(),scale2=scale2,feature_time=feature_time,bound_seconds=btime,
        summary_seconds=stime,labels_aggregation_seconds=label_seconds,total_seconds=time.perf_counter()-start,
        input_sha256=sha(ROOT/'data/g2b_cifar60000/vectors_f32.npy'),
        split_blocks={k:len(v) for k,v in groups.items()},split_rows=dict(train=40000,validation=9984,test=10016),
        split_pairs={k:int(v.sum()) for k,v in split.items()},condition_dimensions=21,base_features=20,
        labels_scope='Only training and validation block pairs aggregated from an existing exact oracle; no test labels computed. Prior same-dataset results already seen.',
        oracle_generation_included=False,oracle_source=str(SWEEP/'artifacts/reference_manifest.json'),
        old_pca_permutation_and_bounds_bitwise_equal=True,numerical_margin=margin)
    save_json(HERE/'artifacts/preparation.json',meta)
    print(json.dumps(meta),flush=True)

if __name__=='__main__':main()
