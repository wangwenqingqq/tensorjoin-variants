#!/usr/bin/env python3
"""G9 R2 fixed-kernel CUDA Graph submission attribution."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from run_g8_layout_probe import build_cases, THRESHOLD
from g9_host_contract import encode, pack, dot_reference, exact_truth_grid

ROOT=Path(__file__).resolve().parents[1]
METHODS=("P16","TC16physical","TC16diagonal")
UUID="GPU-e5a0e7f5-9917-c2a1-ee8e-7282cbba2603"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rawsha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def dump(path,data):
    path.write_text(json.dumps(data,indent=2,sort_keys=True,default=str)+"\n")


def reference_distance(x,ids):
    n=len(x)
    result=np.empty(len(ids),dtype=np.float64)
    for b in range(0,len(ids),512):
        p=ids[b:b+512]
        d=x[p//n].astype(np.float64)-x[p%n].astype(np.float64)
        result[b:b+len(p)]=np.sum(d*d,axis=1)
    return result


def fixtures():
    rng=np.random.default_rng(20260904)
    x=(rng.integers(-(1<<20),1<<20,size=(64,512)).astype(np.float64)*2.0**-20).astype(np.float32)
    x[0:2]=0
    x[2]=np.where(np.arange(512)%2,0.5,-0.5)
    x[3]=x[2]
    x[4:7]=0
    x[4,0]=.75
    x[5,0]=np.nextafter(np.float32(.75),np.float32(-np.inf))
    x[6,0]=np.nextafter(np.float32(.75),np.float32(np.inf))
    i,j=np.triu_indices(64)
    ids=(i*64+j)[:-1].astype(np.int64)
    for label,t in (("below",np.nextafter(.5625,-np.inf)),("equal",.5625),("above",np.nextafter(.5625,np.inf))):
        yield dict(info=dict(name="fixture_"+label,synthetic=True),x=x,ids=ids,logical=ids,threshold=float(t),fixture=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--label",required=True)
    p.add_argument("--mode",choices=("check","timing"),required=True)
    p.add_argument("--reverse",action="store_true")
    args=p.parse_args()
    assert platform.node()=="gpu-host-8" and os.environ["CUDA_VISIBLE_DEVICES"]==UUID
    outpath=ROOT/f"results/{args.label}.json"
    assert not outpath.exists()
    art=ROOT/f"artifacts/{args.label}"
    art.mkdir(exist_ok=False)
    import torch
    import triton
    from g8_layout_kernels import pair_filter
    from g9_tc_kernels import limb_dot, tc_certificate, fp32_subset, fp64_subset
    result=dict(experiment_id="tensorjoin_20260904_g9_int14_tc_refinement_optimistic",
        label=args.label,mode=args.mode,reverse=args.reverse,graph_mode=True,graph_protocol_sha256=sha(ROOT/"PROTOCOL_G9_GRAPH_DIAGNOSTIC_R2.md"),pid=os.getpid(),host=platform.node(),
        python=sys.version,torch=torch.__version__,triton=triton.__version__,numpy=np.__version__,
        cuda_runtime=torch.version.cuda,device=torch.cuda.get_device_name(0),gpu_uuid=UUID,
        source_hashes={f:sha(ROOT/f) for f in ("src/run_g9_graph_diagnostic.py","src/g9_tc_kernels.py",
            "src/g9_host_contract.py","src/g8_layout_kernels.py","src/run_g8_layout_probe.py")},
        protocol_sha256=sha(ROOT/"PROTOCOL_G9_TC_REFINEMENT_A0.md"),
        numerics_sha256=sha(ROOT/"NUMERICS_G9_INT14.md"),gpu_revision_sha256=sha(ROOT/"PROTOCOL_G9_GPU7_R1.md"),cases=[],compiled={},
        scope="TC+certificate+all actual FP32/FP64 repair; free preprocessing and intermediate queues")
    _,cases=build_cases()
    cases=list(fixtures())+cases
    if args.reverse:
        cases.reverse()
    canonical_results={}
    def record(tag,kernel):
        cubin=kernel.asm["cubin"]
        key=hashlib.sha256(cubin).hexdigest()
        if key not in result["compiled"]:
            (art/(key+".cubin")).write_bytes(cubin)
            (art/(key+".ptx")).write_text(kernel.asm["ptx"])
            result["compiled"][key]=dict(tags=[],metadata=kernel.metadata._asdict(),
                n_regs=kernel.n_regs,n_spills=kernel.n_spills,cubin_sha256=key,
                ptx_sha256=sha(art/(key+".ptx")))
        if tag not in result["compiled"][key]["tags"]:
            result["compiled"][key]["tags"].append(tag)
        assert kernel.n_spills==0, (tag,"spilled kernel: retain and stop")
        return key
    for case in cases:
        info=dict(case["info"])
        name=info["name"]
        fixture=case.get("fixture",False)
        x,ids=case["x"],case["ids"]
        n,d=x.shape
        q=len(ids)
        threshold=case.get("threshold",THRESHOLD)
        print("CASE",name,"N",n,"Q",q,flush=True)
        encoding=encode(x)
        reference=reference_distance(x,ids)
        truth=exact_truth_grid(encoding["xi"],ids,n,threshold) if fixture else (reference<=threshold).astype(np.uint8)
        expected_dot=dot_reference(encoding["q"],ids,n)
        gpu={k:torch.from_numpy(np.ascontiguousarray(encoding[k])).cuda() for k in ("h","l","s","norm2","hup","rup")}
        gx=torch.from_numpy(x).cuda()
        gi=torch.from_numpy(ids).cuda()
        rounded=np.float32(threshold)
        lo=float(np.nextafter(rounded,np.float32(-np.inf))) if float(rounded)>threshold else float(rounded)
        hi=float(np.nextafter(rounded,np.float32(np.inf))) if float(rounded)<threshold else float(rounded)
        info.update(fixture=fixture,threshold=threshold,count=q,encoding_audit=encoding["audit"],
                    encoding_host_seconds=encoding["host_seconds"],methods={})
        paths={}
        for method in METHODS:
            ctx=dict(out=torch.full((q+64,),73,device="cuda",dtype=torch.uint8))
            stage_functions=[]
            stage_hashes=[]
            md=dict()
            if method=="P16":
                def initial(ctx=ctx):
                    return pair_filter[(triton.cdiv(q,16),)](gx,gi,ctx["out"][32:32+q],
                        Q=q,N=n,D=d,LO=lo,HI=hi,B=16,K=256,num_warps=4,num_stages=1,enable_fp_fusion=False)
                stage_functions.append(("fp32_full",initial))
                stage_hashes.append(record(name+"/"+method+"/fp32_full",initial()))
                torch.cuda.synchronize()
                f32slots=np.array([],dtype=np.int32)
                md["tc_pending_count"]=None
                md["padding"]=1.0
            else:
                layout=pack(ids,n,method=="TC16diagonal")
                grow,gcol,gslots=[torch.from_numpy(layout[k]).cuda() for k in ("rows","cols","slots")]
                gdot=torch.full((q+64,),123456789,device="cuda",dtype=torch.int64)
                low=torch.empty(q,device="cuda",dtype=torch.float64)
                high=torch.empty_like(low)
                def initial(grow=grow,gcol=gcol,gslots=gslots,gdot=gdot,groups=layout["groups"]):
                    return limb_dot[(groups,)](gpu["h"],gpu["l"],grow,gcol,gslots,gdot[32:32+q],
                        D=d,K=64,num_warps=4,num_stages=2,enable_fp_fusion=False)
                def cert(ctx=ctx,gdot=gdot,low=low,high=high):
                    return tc_certificate[(triton.cdiv(q,256),)](gi,gdot[32:32+q],gpu["s"],gpu["norm2"],
                        gpu["hup"],gpu["rup"],ctx["out"][32:32+q],low,high,
                        Q=q,N=n,T=threshold,B=256,num_warps=4,num_stages=1,enable_fp_fusion=False)
                stage_functions.extend((("tc_dot",initial),("tc_certificate",cert)))
                stage_hashes.extend((record(name+"/"+method+"/tc_dot",initial()),record(name+"/"+method+"/certificate",cert())))
                torch.cuda.synchronize()
                dd=gdot.cpu().numpy()
                assert np.all(dd[:32]==123456789) and np.all(dd[-32:]==123456789)
                assert np.array_equal(dd[32:32+q],expected_dot), (name,method,"integer dot mismatch")
                lower,upper=low.cpu().numpy(),high.cpu().numpy()
                if fixture:
                    for k,pair in enumerate(ids):
                        delta=encoding["xi"][pair//n]-encoding["xi"][pair%n]
                        exact=sum(int(v)*int(v) for v in delta)
                        a,b=float(lower[k]).as_integer_ratio()
                        c,e=float(upper[k]).as_integer_ratio()
                        assert a*(1<<80)<=exact*b and exact*e<=c*(1<<80), (name,"exact enclosure")
                else:
                    assert np.all(lower<=reference) and np.all(reference<=upper), (name,"reference enclosure")
                states=ctx["out"][32:32+q].cpu().numpy()
                assert np.all(states<=2)
                assert np.all(states[states!=2]==truth[states!=2]), (name,method,"TC false decision")
                f32slots=np.flatnonzero(states==2).astype(np.int32)
                md.update(tc_pending_count=len(f32slots),tc_initial_counts=np.bincount(states,minlength=3).tolist(),
                    tc_state_sha256=rawsha(states),tc_queue_sha256=rawsha(f32slots),
                    quantized_dot_sha256=rawsha(expected_dot),integer_dot_exact=True,interval_enclosure_pass=True,
                    groups=layout["groups"],padding=layout["padding"],packing_host_seconds=layout["host_seconds"])
                if len(f32slots):
                    queue=torch.from_numpy(f32slots).cuda()
                    def repair32(ctx=ctx,queue=queue,count=len(f32slots)):
                        return fp32_subset[(triton.cdiv(count,16),)](gx,gi,queue,ctx["out"][32:32+q],
                            M=count,N=n,D=d,LO=lo,HI=hi,B=16,K=256,num_warps=4,num_stages=1,enable_fp_fusion=False)
                    stage_functions.append(("fp32_repair",repair32))
                    stage_hashes.append(record(name+"/"+method+"/fp32_repair",repair32()))
            torch.cuda.synchronize()
            before64=ctx["out"][32:32+q].cpu().numpy()
            assert np.all(before64<=2)
            assert np.all(before64[before64!=2]==truth[before64!=2])
            f64slots=np.flatnonzero(before64==2).astype(np.int32)
            md.update(fp64_pending_count=len(f64slots),fp64_queue_sha256=rawsha(f64slots),
                after_fp32_counts=np.bincount(before64,minlength=3).tolist())
            if len(f64slots):
                queue64=torch.from_numpy(f64slots).cuda()
                def repair64(ctx=ctx,queue64=queue64,count=len(f64slots)):
                    return fp64_subset[(count,)](gx,gi,queue64,ctx["out"][32:32+q],M=count,
                        N=n,D=d,T=threshold,K=256,num_warps=4,num_stages=1,enable_fp_fusion=False)
                stage_functions.append(("fp64_repair",repair64))
                stage_hashes.append(record(name+"/"+method+"/fp64_repair",repair64()))
            def full(stages=tuple(stage_functions)):
                for _,fn in stages:
                    fn()
            def validate(ctx=ctx,method=method):
                state=ctx["out"].cpu().numpy()
                assert np.all(state[:32]==73) and np.all(state[-32:]==73), (name,method,"guard")
                state=state[32:32+q]
                assert np.all(state<=2), (name,method,"unwritten")
                assert np.all(state[state!=2]==truth[state!=2]), (name,method,"false final")
                if not fixture:
                    assert not np.any(state==2), (name,method,"unresolved final")
                    family="synthetic" if info["synthetic"] else "real"
                    canonical=state[np.argsort(case["logical"])]
                    hs=rawsha(canonical)
                    if family in canonical_results:
                        assert canonical_results[family]==hs
                    canonical_results[family]=hs
                return dict(counts=np.bincount(state,minlength=3).tolist(),sha256=rawsha(state))
            full()
            torch.cuda.synchronize()
            md["correctness_before"]=validate()
            md["compiled_stage_hashes"]=stage_hashes
            md["stage_names"]=[s for s,_ in stage_functions]
            paths[method]=dict(ctx=ctx,full=full,validate=validate,stages=stage_functions)
            info["methods"][method]=md
            print(method,"TCpending",md["tc_pending_count"],"FP64",len(f64slots),"padding",md["padding"],flush=True)
        # Capture the unchanged warmed kernels; no queue discovery in the graph.
        for m in METHODS:
            paths[m]["eager_full"]=paths[m]["full"]
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                paths[m]["eager_full"]()
            paths[m]["graph"]=graph
            paths[m]["full"]=graph.replay
            graph.replay()
            torch.cuda.synchronize()
            paths[m]["validate"]()
        if args.mode=="timing" and not fixture:
            schedule=list(METHODS)[::-1] if args.reverse else list(METHODS)
            for method in schedule:
                for _ in range(20):
                    paths[method]["full"]()
            torch.cuda.synchronize()
            def measure(fn,count):
                torch.cuda.synchronize()
                begin,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                begin.record()
                for _ in range(count):
                    fn()
                end.record()
                end.synchronize()
                return float(begin.elapsed_time(end))/count
            samples=[]
            for r in range(40):
                order=schedule[r%3:]+schedule[:r%3]
                for position,m in enumerate(order):
                    samples.append(dict(round=r,position=position,method=m,ms=measure(paths[m]["full"],20)))
            info["samples"]=samples
            info["sustained_ms"]={m:measure(paths[m]["full"],500) for m in schedule}
            info["stage_diagnostic_ms"]={m:{s:measure(fn,20) for s,fn in paths[m]["stages"]} for m in schedule}
        for m in METHODS:
            paths[m]["ctx"]["out"]=torch.full((q+64,),73,device="cuda",dtype=torch.uint8)
            # A captured graph owns old pointers; recapture after allocation churn.
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                paths[m]["eager_full"]()
            paths[m]["graph"]=graph
            paths[m]["full"]=graph.replay
            for _ in range(3):
                paths[m]["full"]()
            torch.cuda.synchronize()
            info["methods"][m]["correctness_after"]=paths[m]["validate"]()
        info["correctness_pass"]=True
        result["cases"].append(info)
        dump(outpath,result)
        print("PASS",name,flush=True)
    result["correctness_pass"]=True
    result["canonical_final_state_hashes"]=canonical_results
    dump(outpath,result)
    print("COMPLETE",args.label,flush=True)


if __name__=="__main__":
    main()
