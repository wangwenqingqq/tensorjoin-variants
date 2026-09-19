import os
from layout_common import *

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    x = source()
    p,q,score,scale2,timing = make_features(x)
    order = np.random.default_rng(2026090813).permutation(len(x)).astype(np.int64)
    np.savez(HERE/'artifacts/splits.npz',train=order[:40000],validation=order[40000:50000],test=order[50000:])
    np.save(HERE/'artifacts/projector.npy',p,allow_pickle=False)
    np.save(HERE/'artifacts/features_f64.npy',q,allow_pickle=False)
    np.save(HERE/'artifacts/pca_scores_f64.npy',score,allow_pickle=False)
    save_json(HERE/'artifacts/features.json',dict(created=time.time(),scale2=scale2,timing=timing,
        projector_sha256=digest(p),features_sha256=digest(q),pca_scores_sha256=digest(score),
        input_sha256=sha(ROOT/'data/g2b_cifar60000/vectors_f32.npy')))
    # This checks the common executor optimization, before evaluating a model.
    previous = np.load(OLD/'artifacts/cifar60000_pca_sort_permutation.npy')
    perm = np.argsort(score,kind='stable').astype(np.int64)
    assert np.array_equal(perm,previous)
    b,bs = projected_bounds(q,perm,scale2)
    prevb = dict(np.load(OLD/'artifacts/cifar60000_pca_sort_bounds.npz'))
    checks=[]
    for cell in CELLS:
        now=b['combined']<=cell['T']+SQ_PAD
        old=prevb['combined']<=cell['T']+SQ_PAD
        assert np.array_equal(now,old)
        checks.append(dict(cell=cell['name'],same_mask=True,retained_tiles=int(now.sum())))
    save_json(HERE/'artifacts/common_bound_admission.json',dict(pass_=True,checks=checks,
        bound_seconds=bs,source_scope=str(OLD/'NUMERICAL_SCOPE.md')))
    previous=Path('@TENSORJOIN_ROOT@/tensorjoin_20260908_learned_layout')
    for name in ['features_f64.npy','pca_scores_f64.npy','projector.npy']:
        assert np.array_equal(np.load(HERE/'artifacts'/name),np.load(previous/'artifacts'/name)),name
    prior_split=dict(np.load(previous/'artifacts/splits.npz'))
    for name,ids in dict(np.load(HERE/'artifacts/splits.npz')).items():
        assert np.array_equal(ids,prior_split[name]),name
    chosen=json.loads((HERE/'inherited/previous_model_selection.json').read_text())['models']['mlp']
    assert sha(HERE/'artifacts/proxy.pt')==chosen['checkpoint_sha256']
    save_json(HERE/'artifacts/inheritance_check.json',dict(pass_=True,
        numeric_features_and_splits_identical=True,prior_proxy_sha256=chosen['checkpoint_sha256'],
        scope='Adaptive follow-up; previous test rows are not a fresh holdout.'))
    print(json.dumps(dict(prepared=True,features=timing,bound_seconds=bs,checks=checks)),flush=True)

if __name__=='__main__':main()
