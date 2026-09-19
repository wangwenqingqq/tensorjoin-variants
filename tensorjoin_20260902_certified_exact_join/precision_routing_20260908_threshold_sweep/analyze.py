"""Analysis of all predeclared cells; never feeds back into experiment selection."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
CELLS=json.loads((HERE/'artifacts/thresholds.json').read_text())['cells']
NAMES=[c['name'] for c in CELLS]
REF=json.loads((HERE/'artifacts/reference_manifest.json').read_text())
METHODS=['F8','F16']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def interval(x):return np.quantile(x,[.025,.975],axis=0).T.tolist()
def block(paths,seed):
 records=[json.loads(p.read_text()) for p in paths];P=len(records)
 reps=3 if records[0]['phase']=='explore' else 6
 data=np.full((P,6,reps,2),np.nan);order=np.zeros((P,6,reps),np.int8)
 raw=[]
 for pi,record in enumerate(records):
  assert record['pass'];label=f"{record['phase']}_{record['index']:02d}_a0"
  guard=json.loads((HERE/'results'/f'{label}_guard.json').read_text());assert guard['pass']
  for s in record['samples']:
   assert s['output_sha256']==REF['cells'][s['cell']]['sha256']
   raw.append(dict(block=record['block'],process=record['index'],**s))
   if s['phase']!='retained':continue
   ci=NAMES.index(s['cell']);mi=METHODS.index(s['method']);ri=s['iteration']
   assert np.isnan(data[pi,ci,ri,mi]);data[pi,ci,ri,mi]=s['seconds']
   if mi==0:order[pi,ci,ri]=s['position']
 assert np.isfinite(data).all()
 medians=np.median(data,axis=2);means=np.mean(medians,axis=0);ratios=means[:,1]/means[:,0]
 rng=np.random.default_rng(seed);B=20000
 ps=rng.integers(0,P,(B,P));rs=rng.integers(0,reps,(B,P,6,reps))
 sampled=data[ps[:,:,None,None],np.arange(6)[None,None,:,None],rs,:]
 bm=np.median(sampled,axis=3).mean(axis=1);br=bm[:,:,1]/bm[:,:,0]
 best_static=float(means.sum(axis=0).min());oracle=float(means.min(axis=1).sum())
 bh=1-bm.min(axis=2).sum(axis=1)/bm.sum(axis=1).min(axis=1)
 rows=[]
 for ci,name in enumerate(NAMES):
  order_ratios={}
  for pos,label in [(0,'F8_first'),(1,'F16_first')]:
   m=np.mean([np.median(data[pi,ci,order[pi,ci]==pos,:],axis=0) for pi in range(P)],axis=0)
   order_ratios[label]=float(m[1]/m[0])
  rows.append(dict(cell=name,mean_degree=REF['cells'][name]['nonself_mean_degree'],seconds_F8=float(means[ci,0]),seconds_F16=float(means[ci,1]),ratio=float(ratios[ci]),ratio_ci95=interval(br[:,ci]),process_ratios=(medians[:,ci,1]/medians[:,ci,0]).tolist(),process_medians_seconds=medians[:,ci,:].tolist(),order_ratios=order_ratios))
 return dict(processes=P,repetitions=reps,retained_calls=P*6*reps*2,warmup_calls=sum(s['phase']=='warmup' for s in raw),cells=rows,oracle_headroom=1-oracle/best_static,oracle_headroom_ci95=interval(bh),best_static=METHODS[int(np.argmin(means.sum(axis=0)))],best_static_mean_seconds=best_static/6,oracle_mean_seconds=oracle/6,bootstrap_draws=B,bootstrap_seed=seed),raw

def main():
 out=HERE/'analysis';out.mkdir(exist_ok=True)
 blocks={};raw=[]
 for name,paths,seed in [
  ('explore',[HERE/'results'/f'explore_{i:02d}.json' for i in range(2)],2026090802),
  ('primary',[HERE/'results'/f'confirm_{i:02d}.json' for i in range(3)],2026090803),
  ('repeat',[HERE/'results'/f'confirm_{i:02d}.json' for i in range(3,6)],2026090804),
  ('pooled',[HERE/'results'/f'confirm_{i:02d}.json' for i in range(6)],2026090805)]:
  blocks[name],r=block(paths,seed)
  if name!='pooled':raw+=r
 f8=[];f16=[]
 for i,name in enumerate(NAMES):
  rows=[blocks[b]['cells'][i] for b in ['primary','repeat']]
  if all(r['ratio_ci95'][0]>1.05 and min(r['process_ratios'])>1 for r in rows):f8.append(name)
  if all(r['ratio_ci95'][1]<1/1.05 and max(r['process_ratios'])<1 for r in rows):f16.append(name)
 gate=dict(F8_material_cells=f8,F16_material_cells=f16,material_crossing=bool(f8 and f16),oracle_headroom_pass=all(blocks[b]['oracle_headroom_ci95'][0]>=.05 for b in ['primary','repeat']))
 gate['pass']=gate['material_crossing'] and gate['oracle_headroom_pass']
 env=json.loads((HERE/'artifacts/environment.json').read_text());assert sha(HERE/'ANALYSIS_PLAN.md')==env['analysis_plan_sha256']
 all_guards=[json.loads(p.read_text()) for p in (HERE/'results').glob('*_guard.json')]
 assert len(all_guards)==10 and all(g['pass'] for g in all_guards)
 local_verified=[];remote_only=[]
 for rel,h in json.loads((HERE/'artifacts/timing_freeze.json').read_text()).items():
  prefix=HERE.name+'/'
  if rel.startswith(prefix):p=HERE/rel[len(prefix):]
  elif rel.startswith('two_gate_campaign_20260908/'):p=HERE/'inherited'/rel.split('/',1)[1]
  else:p=HERE/'inherited'/rel
  if p is not None and p.exists():assert sha(p)==h;local_verified.append(rel)
  else:remote_only.append(rel)
 summary=dict(blocks=blocks,gate=gate,hardware=env,correctness=dict(reference_pairs=REF['pairs_covered'],thresholds=6,admission_calls=24,timing_calls=len(raw),retained_calls=sum(r['phase']=='retained' for r in raw),all_output_hashes_match=True,guards=len(all_guards),all_guards_pass=True,max_device_used_mib=max(g['max_device_used_mib'] for g in all_guards)),local_frozen_files_verified=local_verified,remote_frozen_files_verified_by_each_process=remote_only)
 (out/'summary.json').write_text(json.dumps(summary,indent=2))
 fields=['block','process','cell','method','phase','iteration','position','seconds','output_sha256']
 with (out/'raw_times.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(raw)
 with (out/'summary.csv').open('w') as f:
  fields=['block','cell','mean_degree','seconds_F8','seconds_F16','ratio','ci_low','ci_high'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
  for name,b in blocks.items():
   for r in b['cells']:w.writerow(dict(block=name,**r,ci_low=r['ratio_ci95'][0],ci_high=r['ratio_ci95'][1]))
 with (out/'diagnostics.csv').open('w') as f:
  fields=['cell','method','stage1_accepted','fp32_queue','stage2_accepted','stage2_rejected','fp64_queue','upper_output','stage1_gpu_ms','stage2_gpu_ms','terminal_gpu_ms','preparation_seconds','canonicalization_seconds','output_download_seconds','diagnostic_seconds']
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in json.loads((HERE/'artifacts/admission.json').read_text())['samples']:
   diag=r['diagnostic'];values=[r['cell']['name'],r['method'],*r['stage_counts'],*diag['stage_gpu_ms'],diag['preparation_seconds'],diag['canonicalization_seconds'],diag['output_download_seconds'],diag['seconds']]
   w.writerow(dict(zip(fields,values,strict=True)))
 plot(blocks,out)
 report(summary,out)
 print(json.dumps({'gate':gate,'primary':blocks['primary'],'repeat':blocks['repeat'],'correctness':summary['correctness']},indent=2))

def plot(blocks,out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,2,figsize=(11.6,4.3),layout='constrained');x=np.arange(6)
 labels=[f"{REF['cells'][n]['nonself_mean_degree']:.2f}"+('\noriginal' if n=='original' else '') for n in NAMES]
 ax=axs[0];ax.axhspan(1/1.05,1.05,color='#eeeeee',label='5% materiality band');ax.axhline(1,color='#777777',lw=1)
 for name,offset,color in [('primary',-.09,'#2563eb'),('repeat',.09,'#d97706')]:
  rows=blocks[name]['cells'];v=np.array([r['ratio'] for r in rows]);ci=np.array([r['ratio_ci95'] for r in rows]);ax.errorbar(x+offset,v,yerr=np.stack([v-ci[:,0],ci[:,1]-v]),marker='o',capsize=3,label=name,color=color,lw=1.5)
 ax.set_xticks(x,labels);ax.set_xlabel('Realized mean nonself degree');ax.set_ylabel('Complete-operator time: F16 / F8');ax.set_title('Threshold sweep with paired 95% intervals');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
 ax=axs[1]
 for m,key,color in [('F8','seconds_F8','#2563eb'),('F16','seconds_F16','#d97706')]:
  ax.plot(x,[1000*r[key] for r in blocks['pooled']['cells']],marker='o',label=m,color=color)
 ax.set_yscale('log');ax.set_xticks(x,labels);ax.set_xlabel('Realized mean nonself degree');ax.set_ylabel('Complete-operator latency (ms, log scale)');ax.set_title('Input upload through sorted CPU output');ax.legend();ax.grid(axis='y',alpha=.2)
 for ext in ['png','svg','pdf']:fig.savefig(out/f'threshold_sweep.{ext}',dpi=180)
 plt.close(fig)

def report(s,out):
 b=s['blocks'];admit=json.loads((HERE/'artifacts/admission.json').read_text());d={(r['cell']['name'],r['method']):r for r in admit['samples']}
 lines=['# 阈值扫描实验结果','',f"结论：预登记的两路径路由门槛{'通过' if s['gate']['pass'] else '未通过'}。",'',
 '固定原始 60,000 × 512 输入和行顺序，比较现有融合 F8（INT8）与 F16（FP16）完整流程。阈值由固定的 4,194,304 个非自身点对样本选定；全部六个工作点均进入确认和独立复测。', '',
 '## 端到端结果','',
 '下表耗时为主确认组内各进程中位数的均值。R = F16 / F8；大于 1 表示 F8 更快。置信区间按进程和配对重复分层 bootstrap，主确认与复测各 3 个独立进程，每进程每阈值 6 对。', '',
 '| 工作点 | 半径 ε | 实际平均邻居数 | F8 (ms) | F16 (ms) | 主确认 R [95% CI] | 独立复测 R [95% CI] |',
 '|---|---:|---:|---:|---:|---|---|']
 for i,c in enumerate(CELLS):
  r=b['primary']['cells'][i];rr=b['repeat']['cells'][i]
  fmt=lambda a:f"{a['ratio']:.4f} [{a['ratio_ci95'][0]:.4f}, {a['ratio_ci95'][1]:.4f}]"
  lines.append(f"| {c['name']} | {c['eps']:.8f} | {r['mean_degree']:.4f} | {r['seconds_F8']*1000:.3f} | {r['seconds_F16']*1000:.3f} | {fmt(r)} | {fmt(rr)} |")
 lines+=['','![阈值扫描](analysis/threshold_sweep.png)','','## 预登记门槛','',
 '需要同一对工作点在主确认和复测中分别支持 F8、F16 至少 1.05× 的优势，95% 区间均越过门槛，各进程方向一致；六阈值等查询权重混合下，理想按阈值选择路径的节省还须至少 5%，且区间下界通过。这里的 oracle 假定提前知道每个阈值的最佳路径，未扣除实际选路开销；只衡量整查询选择，不是 tile 级路由。','',
 f"- 稳定通过 F8 优势门槛的工作点：{', '.join(s['gate']['F8_material_cells']) or '无'}。",
 f"- 稳定通过 F16 优势门槛的工作点：{', '.join(s['gate']['F16_material_cells']) or '无'}。",
 f"- 双向实质性交叉：{'通过' if s['gate']['material_crossing'] else '未通过'}。"]
 for name,label in [('primary','主确认'),('repeat','独立复测')]:
  r=b[name];lo,hi=r['oracle_headroom_ci95'];lines.append(f"- {label}：最佳固定路径 {r['best_static']}，理想按阈值选路的端到端节省 {r['oracle_headroom']*100:.3f}% [95% CI {lo*100:.3f}%, {hi*100:.3f}%]。")
 if not s['gate']['pass']:
  lines+=['','本扫描观察到低密度偏向 INT8、高密度偏向 FP16 的排名交叉，但高密度端优势不足 5%，理想选路总收益也不足 0.5%。依照预登记停止条件，这组输入、阈值和完整输出流程不支持继续将当前两路径 planner 作为研究主线。']
 lines+=['','## 数值核验和边界','',
 f"冻结 FP64 终端核实际遍历了全部 {REF['pairs_covered']:,} 个含自身的无序点对，生成最大阈值参考集，再用同一核筛出较小阈值。原阈值参考集为 3,926,078 个有向 ID，哈希与旧实验一致。新旧原阈值 F16 核及 metadata 核的 cubin 完全相同。",'',
 f"24 次准入/诊断调用与参考数组逐项相等；另有 {s['correctness']['timing_calls']} 次计时阶段调用（包含预热，其中 {s['correctness']['retained_calls']} 次保留测量），每次输出数和完整 SHA256 均匹配参考。全部 10 个独占守护进程通过，峰值显存 {s['correctness']['max_device_used_mib']} MiB。",'',
 '结果只支持此数据、硬件和阈值集合上的冻结 FP64 参考语义一致性；不是 exact-real join，也不消除继承的 FP16 Tensor Core 误差前提（[原数值边界说明](inherited/NUMERICAL_SCOPE.md)）。每个新 constexpr 阈值均捕获 cubin/PTX/IR，检查 HMMA、定向舍入、无 FTZ、无 spill/local/scratch，保存 CUDA 参数 ABI。','',
 '## 耗时与队列诊断','',
 '下表来自独立诊断调用，GPU 事件时间及阶段拆分不用于性能门槛。U1 为进入 FP32 精化的无序点对数；两路径的最终 FP64 队列数在各阈值均一致。','',
 '| 工作点 | F8 U1 | F16 U1 | FP64 队列 | F8 排序占比 | F16 排序占比 |',
 '|---|---:|---:|---:|---:|---:|']
 for c in CELLS:
  a=d[c['name'],'F8'];z=d[c['name'],'F16'];assert a['stage_counts'][4]==z['stage_counts'][4]
  portion=lambda r:100*r['diagnostic']['canonicalization_seconds']/r['diagnostic']['seconds']
  lines.append(f"| {c['name']} | {a['stage_counts'][1]:,} | {z['stage_counts'][1]:,} | {a['stage_counts'][4]:,} | {portion(a):.1f}% | {portion(z):.1f}% |")
 lines+=['','在 k1024 工作点，F16 的 FP32 候选队列约缩小 19 倍，但完整输出达 61,458,806 个有向 ID，CPU 镜像和排序约占诊断端到端耗时的 89%–91%。这解释了为何精化阶段的改善只带来很小的整体优势。此诊断不证明排序实现无法改进；本扫描严格保留现有的共同输出流程。','',
 '完整计时包含 tile 列表构造、输入上传、metadata、队列、计数器同步、结果下载和 CPU 镜像排序；不含 JIT、输入磁盘读取、哈希核验及预热。行未重排，因此未添加旧布局实验的额外 identity remap，不能直接与旧布局实验绝对耗时相除。','',
 '## 进程与顺序敏感性','',
 '| 工作点 | 主确认三个进程 R | 复测三个进程 R | 主确认 F8先/F16先 | 复测 F8先/F16先 |',
 '|---|---|---|---|---|']
 for i,n in enumerate(NAMES):
  a=b['primary']['cells'][i];z=b['repeat']['cells'][i]
  fmt=lambda r:', '.join(f'{v:.4f}' for v in r['process_ratios'])
  order=lambda r:'/'.join(f"{r['order_ratios'][k]:.4f}" for k in ['F8_first','F16_first'])
  lines.append(f'| {n} | {fmt(a)} | {fmt(z)} | {order(a)} | {order(z)} |')
 lines+=['','不按耗时删除样本；粗扫结果单独留存，不并入确认估计。每个确认区块只有三个进程簇，因此置信区间的解释限于本次重复测量。', '',
 '## 文件与复现','',
 '- [冻结方案](PROTOCOL.md) 和 [统计细节](ANALYSIS_PLAN.md)。',
 '- [完整统计 JSON](analysis/summary.json)、[汇总 CSV](analysis/summary.csv)、[原始计时 CSV](analysis/raw_times.csv)、[独立阶段诊断 CSV](analysis/diagnostics.csv)。',
 '- [阈值及样本哈希](artifacts/thresholds.json)、[参考集清单](artifacts/reference_manifest.json)、[准入与二进制审计](artifacts/admission.json)、[环境与原核比对](artifacts/environment.json)。',
 '- 远程完整实验目录：`@TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join/precision_routing_20260908_threshold_sweep`；参考 NPY 与上三角分片保留在该目录。',
 '- 本地同步源码、编译产物、日志、结果、守护记录；未复制约 1 GiB 的参考数组。`python analyze.py` 可从同步的结果重新生成统计和图。',
 '- 在原服务器使用相同环境和输入，从新目录按 `select_thresholds.py → reference.py → admit.py → run_campaign.py` 执行；GPU 调用必须经 `guard_r2.py`。现有结果禁止覆盖。','']
 (HERE/'REPORT.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
