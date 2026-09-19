"""Recompute all structural metrics from retained per-tile integer counts."""
import csv,json,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
N=60000;TOTAL=N*(N+1)//2
LAYOUTS=['original','random','kd_variance','pca_sort']
KINDS=['oracle','r16','r32','r64']
CELLS=['k4','k16','k64','original','k256','k1024']
SIZES=[16,32,64,256]

def dump(p,v):p.write_text(json.dumps(v,indent=2))
def quantiles(a):
    return np.quantile(a,[0,.25,.5,.75,.9,.99,1]).tolist() if len(a) else None
def ratio(a,b):return float(a/b) if b else None

def aggregate(a,f):
    if f==1:return a
    starts=np.arange(0,len(a),f)
    return np.add.reduceat(np.add.reduceat(a,starts,axis=0,dtype=np.int64),starts,axis=1,dtype=np.int64)

def maximum(a,f):
    starts=np.arange(0,len(a),f)
    return np.maximum.reduceat(np.maximum.reduceat(a,starts,axis=0),starts,axis=1)

def indices(size):
    ns=np.minimum(size,N-np.arange(0,N,size));r,c=np.triu_indices(len(ns))
    caps=ns[r].astype(np.int64)*ns[c]
    diagonal=r==c;caps[diagonal]=ns[r[diagonal]]*(ns[r[diagonal]]+1)//2
    assert int(caps.sum())==TOTAL
    full=(r!=c)&(ns[r]==size)&(ns[c]==size)
    return r,c,caps,full

def main():
    started=time.perf_counter();idx={s:indices(s) for s in SIZES}
    r16,c16,_,_=idx[16]
    rows=[];native=[];branches=[];histograms={};pair_counts={}
    for layout in LAYOUTS:
        oracle=np.load(HERE/'artifacts'/f'counts_{layout}_oracle.npz')['counts16']
        for kind in KINDS:
            data=np.load(HERE/'artifacts'/f'counts_{layout}_{kind}.npz')
            small=data['counts16']
            supports_r=data['active_rows64'] if kind!='oracle' else None
            supports_c=data['active_cols64'] if kind!='oracle' else None
            assert small.shape==oracle.shape
            assert np.all(small>=oracle)
            for ci,cell in enumerate(CELLS):
                a=np.zeros((N//16,N//16),np.uint16);a[r16,c16]=small[ci]
                o=np.zeros_like(a);o[r16,c16]=oracle[ci]
                true_pairs=int(oracle[ci].sum());candidate_pairs=int(small[ci].sum())
                key=(kind,cell)
                if key in pair_counts:assert pair_counts[key]==candidate_pairs,(layout,key,'layout-dependent pair mask')
                pair_counts[key]=candidate_pairs
                grids={s:aggregate(a,s//16) for s in SIZES}
                oracles={s:aggregate(o,s//16) for s in SIZES}
                work={}
                for s in SIZES:
                    r,c,caps,full=idx[s];v=grids[s][r,c];ov=oracles[s][r,c]
                    assert int(v.sum())==candidate_pairs and int(ov.sum())==true_pairs
                    assert (v<=caps).all() and (v>=ov).all()
                    hit=v>0;work[s]=int(caps[hit].sum())
                    rows.append(dict(layout=layout,kind=kind,cell=cell,size=s,
                        total_pairs=TOTAL,candidate_pairs=candidate_pairs,true_pairs=true_pairs,
                        false_candidates=candidate_pairs-true_pairs,
                        candidate_pair_fraction=candidate_pairs/TOTAL,
                        false_candidate_share=(candidate_pairs-true_pairs)/candidate_pairs,
                        total_tiles=len(v),occupied_tiles=int(hit.sum()),
                        rejected_tile_fraction=float((~hit).mean()),
                        oracle_empty_tiles=int((ov==0).sum()),
                        empty_tile_recall=ratio(np.count_nonzero(~hit),np.count_nonzero(ov==0)),
                        occupied_pair_capacity=work[s],occupied_capacity_fraction=work[s]/TOTAL,
                        work_amplification=work[s]/candidate_pairs,
                        information_limited=candidate_pairs/TOTAL>.5))
                    if s==64:
                        selected=full&hit;empty_selected=selected&(ov==0)
                        record=dict(layout=layout,kind=kind,cell=cell,
                            full_offdiag_occupied_tiles=int(selected.sum()),
                            candidate_count_quantiles=quantiles(v[selected]),
                            candidate_density_median=float(np.median(v[selected])/4096),
                            few6_fraction=ratio(np.count_nonzero(selected&(v<=6)),selected.sum()),
                            few16_fraction=ratio(np.count_nonzero(selected&(v<=16)),selected.sum()),
                            few64_fraction=ratio(np.count_nonzero(selected&(v<=64)),selected.sum()),
                            oracle_empty_survivor_tiles=int(empty_selected.sum()),
                            oracle_empty_candidate_count_quantiles=quantiles(v[empty_selected]),
                            oracle_empty_few64_fraction=ratio(np.count_nonzero(empty_selected&(v<=64)),empty_selected.sum()))
                        if kind!='oracle':
                            nr,nc=supports_r[ci],supports_c[ci]
                            assert np.array_equal(nr>0,hit) and np.array_equal(nc>0,hit)
                            cover=np.minimum(nr,nc)
                            record.update(active_rows_quantiles=quantiles(nr[selected]),active_cols_quantiles=quantiles(nc[selected]),
                                cover8_fraction=ratio(np.count_nonzero(selected&(cover<=8)),selected.sum()),
                                cover16_fraction=ratio(np.count_nonzero(selected&(cover<=16)),selected.sum()),
                                both_full64_fraction=ratio(np.count_nonzero(selected&(nr==64)&(nc==64)),selected.sum()),
                                oracle_empty_cover8_fraction=ratio(np.count_nonzero(empty_selected&(cover<=8)),empty_selected.sum()))
                            record['sparse_exception_gate']=record['few64_fraction']>=.2 or record['cover8_fraction']>=.2
                        native.append(record)
                        histograms[f'{layout}/{kind}/{cell}/pairs64']=np.bincount(v[selected].astype(np.int64),minlength=4097).tolist()
                for parent,child in [(256,64),(64,16)]:
                    factor=parent//child
                    child_grid=grids[child];parent_grid=grids[parent]
                    active=aggregate((child_grid>0).astype(np.uint16),factor)
                    maxmass=maximum(child_grid,factor)
                    r,c,_,full=idx[parent];pv=parent_grid[r,c]
                    selected=full&(pv>0)
                    av=active[r,c][selected];mass=maxmass[r,c][selected]/pv[selected]
                    branches.append(dict(layout=layout,kind=kind,cell=cell,parent=parent,child=child,
                        full_offdiag_occupied_parents=int(selected.sum()),
                        active_child_quantiles=quantiles(av),mean_active_children=float(av.mean()),
                        active_child_fraction=float(av.mean()/(factor*factor)),
                        top_child_mass_quantiles=quantiles(mass),
                        at_most_quarter_children_fraction=float(np.mean(av<=factor*factor//4)),
                        capacity_reduction=1.-work[child]/work[parent],
                        subtile_opportunity_gate=1.-work[child]/work[parent]>=.2))
                    histograms[f'{layout}/{kind}/{cell}/children{parent}_{child}']=np.bincount(av.astype(np.int64),minlength=factor*factor+1).tolist()
            print(json.dumps(dict(analyzed=layout,kind=kind,seconds=time.perf_counter()-started)),flush=True)
            data.close()
    for name,records in [('summary',rows),('native64',native),('branches',branches)]:
        dump(HERE/'analysis'/f'{name}.json',records)
        with (HERE/'analysis'/f'{name}.csv').open('w') as f:
            keys=list(dict.fromkeys(k for row in records for k in row))
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)
    dump(HERE/'analysis/histograms.json',histograms)
    dump(HERE/'results/analysis_verification.json',dict(pass_=True,configurations=len(rows),native_records=len(native),
        branch_records=len(branches),layout_invariant_pair_counts=True,all_counts_dominate_reference=True,
        all_aggregates_preserve_pair_count=True,seconds=time.perf_counter()-started))

if __name__=='__main__':main()
