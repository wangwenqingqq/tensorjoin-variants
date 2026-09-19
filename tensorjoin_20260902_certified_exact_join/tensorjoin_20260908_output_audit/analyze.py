"""Frozen shared-output comparison; exploratory output selection is separate."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
SWEEP=HERE.parent/'precision_routing_20260908_threshold_sweep'
CELLS=json.loads((SWEEP/'artifacts/thresholds.json').read_text())['cells'];NAMES=[c['name'] for c in CELLS]
REF=json.loads((SWEEP/'artifacts/reference_manifest.json').read_text())['cells'];METHODS=['F8','F16','F32'];OUTPUTS=['legacy','cpu_single','gpu_radix']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ci(x):return np.quantile(x,[.025,.975],axis=0).T.tolist()
def estimate(indices,seed,challenger):
 P=len(indices);data=np.full((P,6,3,4,2),np.nan);raw=[]
 for pi,index in enumerate(indices):
  p=HERE/'results'/f'confirm_{index:02d}.json';r=json.loads(p.read_text());assert r['pass'] and r['selection']['challenger']==challenger
  assert json.loads((HERE/'results'/f'confirm_{index:02d}_a0_guard.json').read_text())['pass']
  for s in r['samples']:
   assert s['output_sha256']==REF[s['cell']]['sha256'];raw.append(dict(block=r['block'],process=index,**s))
   if s['phase']!='retained':continue
   c=NAMES.index(s['cell']);m=METHODS.index(s['method']);o=['legacy',challenger].index(s['output_mode']);i=s['iteration']
   assert np.isnan(data[pi,c,m,i,o]);data[pi,c,m,i,o]=s['seconds']
 assert np.isfinite(data).all();pm=np.median(data,axis=3);means=pm.mean(axis=0)
 rng=np.random.default_rng(seed);B=20000;ps=rng.integers(0,P,(B,P));rs=rng.integers(0,4,(B,P,6,3,4))
 sampled=data[ps[:,:,None,None,None],np.arange(6)[None,None,:,None,None],np.arange(3)[None,None,None,:,None],rs,:]
 bm=np.median(sampled,axis=4).mean(axis=1);speed=means[:,:,0]/means[:,:,1];bs=bm[:,:,:,0]/bm[:,:,:,1]
 rows=[]
 for c,name in enumerate(NAMES):
  for m,method in enumerate(METHODS):
   rows.append(dict(cell=name,method=method,mean_degree=REF[name]['nonself_mean_degree'],legacy_seconds=float(means[c,m,0]),new_seconds=float(means[c,m,1]),speedup=float(speed[c,m]),speedup_ci95=ci(bs[:,c,m]),process_legacy_seconds=pm[:,c,m,0].tolist(),process_new_seconds=pm[:,c,m,1].tolist(),process_speedups=(pm[:,c,m,0]/pm[:,c,m,1]).tolist()))
 saving=1-means[:,:,1].sum()/means[:,:,0].sum();bsv=1-bm[:,:,:,1].sum(axis=(1,2))/bm[:,:,:,0].sum(axis=(1,2))
 comparisons=[]
 for c,name in enumerate(NAMES):
  comparisons.append(dict(cell=name,ratio_F16_F8=float(means[c,1,1]/means[c,0,1]),ratio_F16_F8_ci95=ci(bm[:,c,1,1]/bm[:,c,0,1]),speedup_F32_F8=float(means[c,2,1]/means[c,0,1]),speedup_F32_F8_ci95=ci(bm[:,c,2,1]/bm[:,c,0,1]),speedup_F32_F16=float(means[c,2,1]/means[c,1,1]),speedup_F32_F16_ci95=ci(bm[:,c,2,1]/bm[:,c,1,1])))
 tc=means[:,:2,1];btc=bm[:,:,:2,1];headroom=1-tc.min(axis=1).sum()/tc.sum(axis=0).min();bh=1-btc.min(axis=2).sum(axis=1)/btc.sum(axis=1).min(axis=1)
 return dict(process_indices=indices,bootstrap_seed=seed,bootstrap_draws=B,rows=rows,mixture_time_saving=float(saving),mixture_time_saving_ci95=ci(bsv),mixture_speedup=float(means[:,:,0].sum()/means[:,:,1].sum()),legacy_mean_seconds=float(means[:,:,0].mean()),new_mean_seconds=float(means[:,:,1].mean()),comparisons=comparisons,descriptive_F8_F16_oracle_headroom=float(headroom),descriptive_F8_F16_oracle_headroom_ci95=ci(bh),best_static_TC=METHODS[int(tc.sum(axis=0).argmin())]),raw

def main():
 out=HERE/'analysis';out.mkdir(exist_ok=True)
 selection=json.loads((HERE/'artifacts/selection.json').read_text());challenger=selection['challenger'];blocks={};raw=[]
 for name,ids,seed in [('primary',list(range(3)),2026090811),('repeat',list(range(3,6)),2026090812),('pooled',list(range(6)),2026090813)]:
  blocks[name],r=estimate(ids,seed,challenger)
  if name!='pooled':raw+=r
 exploration=[]
 for index in range(2):
  r=json.loads((HERE/'results'/f'explore_{index:02d}.json').read_text());assert r['pass']
  assert sha(HERE/'results'/f'explore_{index:02d}.json')==selection['exploration_sha256'][f'explore_{index:02d}.json']
  for s in r['samples']:assert s['output_sha256']==REF[s['cell']]['sha256'];raw.append(dict(block='explore',process=index,**s))
 for c,name in enumerate(NAMES):
  for m in METHODS:
   vals={}
   for mode in OUTPUTS:
    vals[mode]=float(np.mean([np.median([r['seconds'] for r in raw if r['block']=='explore' and r['process']==i and r['phase']=='retained' and r['cell']==name and r['method']==m and r['output_mode']==mode]) for i in range(2)]))
   exploration.append(dict(cell=name,method=m,**vals))
 for mode in OUTPUTS:assert np.isclose(np.mean([r[mode] for r in exploration]),selection['scores'][mode],rtol=1e-13)
 regressions=[]
 for i,r in enumerate(blocks['primary']['rows']):
  rr=blocks['repeat']['rows'][i]
  if all(x['speedup_ci95'][1]<1/1.05 and max(x['process_speedups'])<1 for x in [r,rr]):regressions.append([r['cell'],r['method']])
 gate=dict(mixture_saving_pass=all(blocks[b]['mixture_time_saving_ci95'][0]>=.05 for b in ['primary','repeat']),reproducible_regressions_gt5pct=regressions)
 gate['replacement_recommended']=gate['mixture_saving_pass'] and not regressions
 guards=[json.loads(p.read_text()) for p in (HERE/'results').glob('*_guard.json')];assert len(guards)==9 and all(g['pass'] for g in guards)
 assert len(raw)==1404 and sum(r['phase']=='retained' for r in raw)==1080
 env=json.loads((HERE/'artifacts/environment.json').read_text());assert sha(HERE/'ANALYSIS_PLAN.md')==env['analysis_plan_sha256']
 local_verified=[];remote_only=[]
 for rel,want in json.loads((HERE/'artifacts/timing_freeze.json').read_text()).items():
  p=HERE/rel.split('/',1)[1] if rel.startswith(HERE.name+'/') else HERE/'inherited'/rel
  if p.exists():assert sha(p)==want,rel;local_verified.append(rel)
  else:remote_only.append(rel)
 summary=dict(selection=selection,blocks=blocks,exploration=exploration,gate=gate,correctness=dict(timing_calls=len(raw),retained_calls=sum(r['phase']=='retained' for r in raw),admission_calls=72,all_output_hashes_match=True,guards=len(guards),all_guards_pass=True,max_device_used_mib=max(g['max_device_used_mib'] for g in guards)),max_allocated_bytes=max(r['peak_allocated_bytes'] for r in raw),max_reserved_bytes=max(r['peak_reserved_bytes'] for r in raw),max_cub_scratch_bytes=max(r['cub_scratch_bytes'] for r in raw),environment=env)
 summary.update(local_frozen_files_verified=local_verified,remote_frozen_files_verified_by_each_process=remote_only)
 (out/'summary.json').write_text(json.dumps(summary,indent=2))
 fields=['block','process','cell','method','output_mode','phase','iteration','position','seconds','output_finalize_seconds','output_download_bytes','output_gpu_copy_bytes','cub_scratch_bytes','peak_allocated_bytes','peak_reserved_bytes','output_sha256']
 with (out/'raw_times.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(raw)
 fields=['block','cell','method','mean_degree','legacy_seconds','new_seconds','speedup','ci_low','ci_high']
 with (out/'summary.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
  for block,r in blocks.items():
   for row in r['rows']:w.writerow(dict(block=block,**row,ci_low=row['speedup_ci95'][0],ci_high=row['speedup_ci95'][1]))
 fields=['cell','method','seconds','preparation_seconds','pre_finalize_seconds','output_finalize_seconds','output_collect_seconds','output_download_bytes','output_gpu_copy_bytes','cub_scratch_bytes','stage_gpu_ms','stage_counts']
 with (out/'diagnostics.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(json.loads((HERE/'artifacts/admission.json').read_text())['diagnostics'])
 plot(summary,out);report(summary,out)
 print(json.dumps(dict(gate=gate,correctness=summary['correctness'],blocks={b:{k:v for k,v in r.items() if k not in ['rows','comparisons']} for b,r in blocks.items()}),indent=2))

def plot(s,out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,2,figsize=(11.6,4.3),layout='constrained');x=np.arange(6);rows=s['blocks']['pooled']['rows'];labels=[f"{REF[n]['nonself_mean_degree']:.2f}"+('\noriginal' if n=='original' else '') for n in NAMES]
 colors=['#2563eb','#d97706','#168b70']
 for m,color in zip(METHODS,colors):
  r=[r for r in rows if r['method']==m];v=np.array([r['speedup'] for r in r]);c=np.array([r['speedup_ci95'] for r in r]);axs[0].errorbar(x,v,yerr=np.stack([v-c[:,0],c[:,1]-v]),marker='o',capsize=3,label=m,color=color)
  axs[1].plot(x,[r['new_seconds']*1000 for r in r],marker='o',label=m,color=color)
 axs[0].axhline(1,color='#777777',lw=1);axs[0].set_ylabel('Complete-operator speedup: legacy / GPU output');axs[0].set_title('Benefit of the common output implementation')
 axs[1].set_yscale('log');axs[1].set_ylabel('Complete-operator latency (ms, log scale)');axs[1].set_title('Fixed methods with the same GPU output')
 for ax in axs:ax.set_xticks(x,labels);ax.set_xlabel('Realized mean nonself degree');ax.legend();ax.grid(axis='y',alpha=.2)
 for ext in ['png','svg','pdf']:fig.savefig(out/f'output_audit.{ext}',dpi=180)
 plt.close(fig)

def report(s,out):
 b=s['blocks'];selected=s['selection']['challenger'];admit=json.loads((HERE/'artifacts/admission.json').read_text())
 rmap=lambda block:{(r['cell'],r['method']):r for r in b[block]['rows']}
 primary=rmap('primary');repeat=rmap('repeat')
 lines=['# 共享输出优化与基线审计','',
 f"结论：{'建议采用' if s['gate']['replacement_recommended'] else '尚不建议统一替换为'} `{selected}` 公共输出实现。该结果属于共同执行流程的工程改进，未建立新的精度规划器或算法新颖性。",'',
 '原始数据为 60,000 × 512 FP32；沿用已冻结六个阈值、输入顺序、全部 FP64 参考结果。F8、F16 的数值核逐字节保持不变；F32 使用已有 pedantic cuBLAS + 冻结分类核 + FP64 终端。三条路径共享同一个输出实现，完整结果仍是 CPU 上升序排列的全部有向 uint64 pair ID。','',
 '## 输出优化收益','',
 '| 独立区块 | 原公共输出平均耗时 | 新公共输出平均耗时 | 完整加速 | 耗时节省 [95% CI] |',
 '|---|---:|---:|---:|---|']
 for name,label in [('primary','主确认'),('repeat','独立复测')]:
  r=b[name];lo,hi=r['mixture_time_saving_ci95']
  lines.append(f"| {label} | {r['legacy_mean_seconds']*1000:.3f} ms | {r['new_mean_seconds']*1000:.3f} ms | {r['mixture_speedup']:.3f}× | {100*r['mixture_time_saving']:.2f}% [{100*lo:.2f}%, {100*hi:.2f}%] |")
 lines+=['','平均值按六个阈值 × 三种算子的 18 个组合等调用权重计算；每格先取各进程保留测量的中位数，再平均。主确认与复测各三个新进程，每格四对测量。置信区间来自成对输出测量和进程簇的分层 bootstrap。', '',
 f"超过 5% 的可复现退化组合：{s['gate']['reproducible_regressions_gt5pct'] or '无'}。各组合的原始计时均保留，不按速度剔除。",'',
 '| 工作点 | 方法 | 原输出 ms | 新输出 ms | 主确认加速 [95% CI] | 复测加速 [95% CI] |',
 '|---|---|---:|---:|---|---|']
 fmt=lambda r:f"{r['speedup']:.3f}× [{r['speedup_ci95'][0]:.3f}, {r['speedup_ci95'][1]:.3f}]"
 for c in NAMES:
  for m in METHODS:
   r=primary[c,m];z=repeat[c,m];lines.append(f"| {c} | {m} | {r['legacy_seconds']*1000:.3f} | {r['new_seconds']*1000:.3f} | {fmt(r)} | {fmt(z)} |")
 lines+=['','![公共输出改进](analysis/output_audit.png)','','图使用六个确认进程合并的描述性估计；替换门槛仍分别按主确认和复测区块判断。','','## 共享新输出后的计算路径比较','',
 '以下均使用相同的新输出实现；主确认与复测分开报告。F32 是本机已有的 pedantic FP32 强对照，不代表已复现所有外部系统。F32/F16 大于 1 表示 F16 更快；F16/F8 大于 1 表示 F8 更快。','',
 '| 工作点 | F8 ms | F16 ms | F32 ms | F32/F16 主确认 / 复测 | F16/F8 主确认 / 复测 |',
 '|---|---:|---:|---:|---|---|']
 for i,c in enumerate(NAMES):
  a=b['primary']['comparisons'][i];z=b['repeat']['comparisons'][i]
  lines.append(f"| {c} | {primary[c,'F8']['new_seconds']*1000:.3f} | {primary[c,'F16']['new_seconds']*1000:.3f} | {primary[c,'F32']['new_seconds']*1000:.3f} | {a['speedup_F32_F16']:.3f} / {z['speedup_F32_F16']:.3f} | {a['ratio_F16_F8']:.3f} / {z['ratio_F16_F8']:.3f} |")
 lines+=['','各路径比值的 95% 区间和逐进程耗时见 [完整统计](analysis/summary.json)。不同计算方法在独立方法块测量，因此统计配对依赖进程簇；不把不相关的方法调用冒充逐次配对。','',
 '作为描述性敏感性结果，六阈值等概率、提前知道最佳路径且不扣选择开销时：','']
 for name,label in [('primary','主确认'),('repeat','复测')]:
  r=b[name];lo,hi=r['descriptive_F8_F16_oracle_headroom_ci95'];lines.append(f"- {label}最佳固定 TC 路径为 {r['best_static_TC']}；F8/F16 理想选路节省 {r['descriptive_F8_F16_oracle_headroom']*100:.3f}% [{lo*100:.3f}%, {hi*100:.3f}%]。")
 lines+=['','该描述性结果未实现选择器，也不重开上一轮已停止的 planner 研究主张。','',
 '## 改动和对照','',
 '- `legacy`：每批上三角结果下载，排序上三角 ID，镜像，再排序完整 ID。',
 '- `cpu_single`：相同下载流程，删除第一遍上三角排序，镜像后统一排序。',
 '- `gpu_radix`：每批结果保留 GPU 副本，拼接后以整数 ID 镜像，调用 NVIDIA CUB 64 位无符号 radix sort，再下载最终有序结果。自身的额外镜像用最大 uint64 哨兵表示，排序到末尾后按实际自身计数删除。',
 '',
 'GPU 改进包含结果暂存、镜像、排序和传输流程的整体变化，不能把全部收益归因于单个排序核。结果规模、设备临时空间和复制成本都进入完整计时；未使用参考结果规模预分配。','',
 '两轮粗测的平均完整耗时如下；这部分只用于按已写明的统一规则选输出实现，不并入确认估计：','',
 '| 输出实现 | 粗测平均耗时 |',
 '|---|---:|']
 for o in OUTPUTS:lines.append(f"| {o} | {s['selection']['scores'][o]*1000:.3f} ms |")
 lines+=['','## 数值核验和资源代价','',
 f"54 个正常配置和 18 次单独诊断均与已有完整 FP64 参考数组逐项相等；空结果、自身、乱序、非整块长度、超过有符号 32 位范围的 ID、越界输入检测和前后哨兵检查均通过。另有 {s['correctness']['timing_calls']} 次计时阶段调用，其中 {s['correctness']['retained_calls']} 次保留测量，每次输出数量和完整 SHA256 均匹配参考。",'',
 f"所有 {s['correctness']['guards']} 个守护进程通过；守护采样的最大设备用量 {s['correctness']['max_device_used_mib']} MiB。计时调用的最大 PyTorch 已分配显存 {s['max_allocated_bytes']/2**20:.1f} MiB、保留显存 {s['max_reserved_bytes']/2**20:.1f} MiB；最大 CUB scratch 为 {s['max_cub_scratch_bytes']/2**20:.1f} MiB。",'',
 'GPU 版本在 k1024 下载完整有向结果 491,670,448 字节；旧流程仅下载上三角 246,075,224 字节再由 CPU 镜像。因此改进需要更多 D2H 字节和 GPU 临时空间，并非通过减少最终答案或压缩输出合同取得。','',
 f"CUB 版本宏为 {admit['cub_version']}，nvcc 为 CUDA 13.1。完整编译命令、编译器/源码/共享库和全部编译依赖哈希保存在 [build.json](artifacts/build.json)。F16 计算核和 metadata 均与原阈值扫描捕获产物字节一致；FP32 的库配置、库哈希和冻结核身份保存在准入证据中。",'',
 '数值结论限于此输入、阈值、硬件和冻结 FP64 判定过程；不主张 exact-real 语义，也不消除继承的 FP16 数值前提。复测每块仅三个进程簇，区间不代表跨硬件或跨数据分布泛化。','',
 '## 基线和研究结论','',
 '本轮证明了共同输出实现存在可修复的性能问题，并建立了更快且公平共享的完整输出对照。没有增加新的距离算法或精度选择机制。外部 FaSTED 与 RT-HiSS 的机制和合同核查见 [基线范围](BASELINE_SCOPE.md)；本轮未运行它们的实现。', '',
 '后续算法研究必须相对这个改进后的公共执行器衡量净收益。若探索块级剪枝，应单独固定覆盖 FP64 参考判定的界、真实数据集合、判界开销和停止条件；现有输出加速不能充当剪枝有效或新颖性的证据。','',
 '## 复现文件','',
 '- [协议](PROTOCOL.md)、[统计细节](ANALYSIS_PLAN.md)、[选型证据](artifacts/selection.json)。',
 '- [汇总 CSV](analysis/summary.csv)、[全部原始计时 CSV](analysis/raw_times.csv)、[单独阶段诊断 CSV](analysis/diagnostics.csv)、[完整统计](analysis/summary.json)。',
 '- [准入及阶段诊断](artifacts/admission.json)、[冻结文件](artifacts/timing_freeze.json)、[环境](artifacts/environment.json)、[独立统计交叉核对](analysis/independent_verification.json)、[最终资源检查](results/final_resource_check.json)。',
 '- 源码在 `src/`；GPU 输出实现为 `src/output.cu` 和 `src/common.py`。`python analyze.py` 从结果重新生成统计与图。',
 '- 原服务器实验目录：`@TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join/tensorjoin_20260908_output_audit`。同级旧阈值扫描目录保留全部大型参考 NPY；本轮直接核验其哈希并复用。','']
 (HERE/'REPORT.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
