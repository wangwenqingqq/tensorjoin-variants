"""Post-freeze analysis of actual OT-CFM and matched one-pass layout control."""
import csv
import json
import math
import sys
import platform
import importlib.metadata
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
CELLS=['k4','k16','k64','original','k256','k1024']
METHODS=['F8','F16']
CONFIGS=['original_full','pca_geo','direct_geo','fm_geo','fm_full']
FAMILIES=['pca','direct','fm']
COLORS={'pca':'#666666','direct':'#e69f00','fm':'#0072b2'}

def read(rel):return json.loads((HERE/rel).read_text())
def ci(values):return np.quantile(values,[.025,.975]).tolist()
def csv_write(name,rows):
    with (HERE/'analysis'/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def amortize(extra,saving):return max(0,math.ceil(extra/saving)) if saving>0 else None

def main():
    (HERE/'analysis').mkdir(exist_ok=True)
    (HERE/'analysis/statistics_environment.json').write_text(json.dumps(dict(python=sys.version,
        platform=platform.platform(),packages={n:importlib.metadata.version(n) for n in ['numpy','matplotlib']}),indent=2))
    t=read('results/training.json');ad=read('artifacts/admission.json')
    runs=[read(f'results/confirm_{i:02d}.json') for i in range(3)]
    guards=[read(f'results/{name}_guard.json') for name in
        ['training_a0','admission_a0']+[f'confirm_{i:02d}_a0' for i in range(3)]]
    assert t['pass_'] and ad['pass'] and all(r['pass'] for r in runs+guards)
    samples=[s for r in runs for s in r['samples'] if s['phase']=='retained']
    assert len(samples)==720
    a=np.full((3,4,6,2,5),np.nan)
    for s in samples:
        idx=(s['process'],s['repeat'],CELLS.index(s['cell']),METHODS.index(s['method']),CONFIGS.index(s['configuration']))
        assert np.isnan(a[idx]);a[idx]=s['seconds']
    assert np.isfinite(a).all()
    proc=np.median(a,axis=1);point=proc.mean(axis=0)
    rng=np.random.default_rng(2026090838)
    pp=rng.integers(0,3,(20000,3));rr=rng.integers(0,4,(20000,3,4))
    boot=np.median(a[pp[:,:,None],rr],axis=2).mean(axis=1)
    build_calls=[s for r in runs for s in r['single_queries']]
    builds={f:float(np.mean([s['build_wall_seconds'] for s in build_calls if s['family']==f])) for f in FAMILIES}
    costs={'pca':dict(selected_training_seconds=0.,all_seed_training_and_selection_seconds=0.)}
    for family in ['direct','fm']:
        sel=t['selection']['models'][family]
        models=[m for m in t['models'] if m['family']==family]
        costs[family]=dict(selected_training_seconds=sel['training_seconds']+sel['initialization_seconds'],
            selected_updates_seconds=sel['update_seconds'],selected_pairing_seconds=sel['coupling_seconds'],
            all_seed_training_seconds=sum(m['training_seconds']+m['initialization_seconds'] for m in models),
            all_seed_evaluation_seconds=sum(m['evaluation_seconds'] for m in models),
            all_seed_training_and_selection_seconds=sum(m['training_seconds']+m['initialization_seconds']+m['evaluation_seconds'] for m in models),
            note='Each standalone family pays its full pairing cost. Selected checkpoint is a lower-bound replay cost, without search. Accounted components exclude minor Python/setup overhead.')
    primary=[];comparisons=[];mixtures=[]
    for cj,cell in enumerate(CELLS):
        for mj,method in enumerate(METHODS):
            row=dict(cell=cell,method=method)
            for j,config in enumerate(CONFIGS):row[config+'_ms']=float(point[cj,mj,j]*1000)
            primary.append(row)
            for baseline,bj in [('pca',1),('direct',2)]:
                saving=1-point[cj,mj,3]/point[cj,mj,bj]
                interval=ci(1-boot[:,cj,mj,3]/boot[:,cj,mj,bj])
                ps=(1-proc[:,cj,mj,3]/proc[:,cj,mj,bj]).tolist()
                gain=point[cj,mj,bj]-point[cj,mj,3]
                comparisons.append(dict(cell=cell,method=method,baseline=baseline,
                    saving_fraction=float(saving),saving_ci95=interval,process_savings=ps,
                    speedup=float(point[cj,mj,bj]/point[cj,mj,3]),
                    speedup_ci95=ci(boot[:,cj,mj,bj]/boot[:,cj,mj,3]),
                    passes_5pct_gate=bool(interval[0]>=.05 and min(ps)>0),
                    selected_cost_break_even_queries=amortize(costs['fm']['selected_training_seconds']-costs[baseline]['selected_training_seconds']+builds['fm']-builds[baseline],gain),
                    all_seed_cost_break_even_queries=amortize(costs['fm']['all_seed_training_and_selection_seconds']-costs[baseline]['all_seed_training_and_selection_seconds']+builds['fm']-builds[baseline],gain)))
    for mj,method in enumerate(METHODS):
        times=point[:,mj,:].mean(axis=0);bb=boot[:,:,mj,:].mean(axis=1)
        for baseline,bj in [('pca',1),('direct',2)]:
            saving=1-times[3]/times[bj];interval=ci(1-bb[:,3]/bb[:,bj])
            ps=(1-proc[:,:,mj,3].mean(axis=1)/proc[:,:,mj,bj].mean(axis=1)).tolist()
            gain=times[bj]-times[3]
            mixtures.append(dict(method=method,baseline=baseline,baseline_ms=float(times[bj]*1000),
                fm_ms=float(times[3]*1000),saving_fraction=float(saving),saving_ci95=interval,
                process_savings=ps,passes_5pct_gate=bool(interval[0]>=.05 and min(ps)>0),
                selected_cost_break_even_queries=amortize(costs['fm']['selected_training_seconds']-costs[baseline]['selected_training_seconds']+builds['fm']-builds[baseline],gain),
                all_seed_cost_break_even_queries=amortize(costs['fm']['all_seed_training_and_selection_seconds']-costs[baseline]['all_seed_training_and_selection_seconds']+builds['fm']-builds[baseline],gain)))
    mapping=[]
    for family,steps in [('direct',0),('fm',8),('fm',16),('fm',32)]:
        rows=[next(s for s in r['mapping'] if s['family']==family and s['solver_steps']==steps) for r in runs]
        med=[s['median_seconds'] for s in rows]
        assert len({s['coordinates_sha256'] for s in rows})==1
        mapping.append(dict(family=family,solver_steps=steps,nfe=rows[0]['nfe'],selected=rows[0]['selected'],
            parameters=rows[0]['parameters'],mean_process_median_ms=float(np.mean(med)*1000),
            process_median_ms=[x*1000 for x in med],dense_layer_flops=rows[0]['dense_layer_flops'],
            input_rows=60000,features=32,device='cuda'))
    cold=[]
    for family in FAMILIES:
        for cell in ['k4','original','k1024']:
            for method in METHODS:
                rows=[s for s in build_calls if s['family']==family and s['cell']==cell and s['method']==method]
                assert len(rows)==3
                cold.append(dict(family=family,cell=cell,method=method,
                    build_plus_query_seconds_mean=float(np.mean([r['seconds'] for r in rows])),
                    build_seconds_mean=float(np.mean([r['build_wall_seconds'] for r in rows])),training_included=False))
    full_calls=len(ad['samples'])+len(ad['diagnostics'])+len(ad['build_admission'])+sum(len(r['samples'])+len(r['single_queries']) for r in runs)
    assert full_calls==1029
    s=dict(primary=primary,comparisons=comparisons,mixtures=mixtures,mapping=mapping,
        training_costs=costs,actual_joint_training_and_selection_wall_seconds=t['ended']-t['started'],
        build_seconds=builds,cold_queries=cold,geometry=ad['geometry'],selection=t['selection'],
        flow_diagnostics=ad['flow_diagnostics'],gaussianization_baseline=t['gaussianization_baseline'],
        retained_calls=720,warmups=180,admission_calls=75,build_plus_query_calls=len(build_calls),full_gpu_calls=full_calls,
        reference_occupancy_checks=sum(len(g['reference_checks']) for f in ad['geometry'].values() for g in f.values()),
        max_device_used_mib=max(g['max_device_used_mib'] for g in guards),
        bootstrap=dict(draws=20000,seed=2026090838,unit='process then paired repetitions'),
        scope='Actual Gaussianizing minibatch OT-CFM; reused prior test split and transductive PCA. Same original bound axes. Only within-campaign GPU4 ratios.')
    (HERE/'analysis/summary.json').write_text(json.dumps(s,indent=2))
    for name,rows in [('summary',primary),('comparisons',comparisons),('mixtures',mixtures),('mapping',mapping),('cold_queries',cold)]:csv_write(name+'.csv',rows)
    keys=['process','repeat','cell','method','configuration','seconds','executed_tiles','output_count','output_sha256']
    csv_write('raw_times.csv',[{k:r[k] for k in keys} for r in samples])
    csv_write('candidate_evaluations.csv',[dict(family=c['family'],seed=c['seed'],updates=c['step'],
        rule=c['rule'],solver_steps=c['solver_steps'],nfe=c['nfe'],
        validation_pruning=c['validation']['mean_pair_pruning'],
        gaussian_sliced_w2_squared_normalized=c['distribution']['sliced_w2_squared_normalized'],
        selected=c==t['selection']['models'][c['family']],checkpoint_sha256=c['checkpoint_sha256']) for c in t['candidates']])
    plot(s,t);report(s,t,ad)
    print(json.dumps({k:s[k] for k in ['mixtures','mapping','training_costs','build_seconds','full_gpu_calls']},indent=2))

def plot(s,t):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
    for family in ['direct','fm']:
        histories=[r['history'] for r in t['models'] if r['family']==family]
        steps=[r['step'] for r in histories[0]];v=np.array([[r['loss'] for r in h] for h in histories])
        axes[0,0].plot(steps,v.mean(0),label=family,color=COLORS[family])
        axes[0,0].fill_between(steps,v.min(0),v.max(0),alpha=.15,color=COLORS[family])
    axes[0,0].set(title='Training objectives (different regression targets)',xlabel='Updates',ylabel='Normalized MSE');axes[0,0].legend()
    base=t['gaussianization_baseline']
    vals=[base['sliced_w2_squared_normalized']]+[t['selection']['models'][f]['distribution']['sliced_w2_squared_normalized'] for f in ['direct','fm']]
    bars=axes[0,1].bar(FAMILIES,vals,color=[COLORS[f] for f in FAMILIES]);axes[0,1].bar_label(bars,fmt='%.4f',padding=3)
    axes[0,1].axhline(base['gaussian_sampling_floor'],color='red',ls=':',label='Gaussian sampling floor')
    axes[0,1].set_yscale('log');axes[0,1].set(title='Selected models: Gaussian fit on validation',ylabel='Normalized sliced-W2 squared (lower better)');axes[0,1].legend(fontsize=8)
    xx=np.arange(3)
    for offset,split,color in [(-.18,'test','#56b4e9'),(.18,'full','#0072b2')]:
        vals=[100*s['geometry'][f][split]['mean_pair_pruning'] for f in FAMILIES]
        bars=axes[0,2].bar(xx+offset,vals,.34,label='reused test' if split=='test' else split,color=color)
        axes[0,2].bar_label(bars,fmt='%.2f',padding=3)
    axes[0,2].set_xticks(xx,FAMILIES);axes[0,2].set(title='Same certified bound, changed row grouping',ylabel='Mean safely pruned pairs (%)');axes[0,2].legend()
    for j,method in enumerate(METHODS):
        rows=[r for r in s['comparisons'] if r['method']==method and r['baseline']=='pca']
        x=np.arange(6)+(j-.5)*.1;cc=np.asarray([r['speedup_ci95'] for r in rows]);color=['#0072b2','#e69f00'][j]
        axes[1,0].vlines(x,cc[:,0],cc[:,1],color=color)
        axes[1,0].plot(x,[r['speedup'] for r in rows],'o-',label=method,color=color)
    axes[1,0].axhline(1,color='black',lw=.8);axes[1,0].set_xticks(np.arange(6),CELLS)
    axes[1,0].set(title='Repeated complete queries: PCA time / FM time',ylabel='Speedup (>1 favors FM)');axes[1,0].legend()
    rows=s['mapping'];labels=['Direct\n1 eval']+[f"FM {r['solver_steps']} steps\n{r['nfe']} evals" for r in rows[1:]]
    bars=axes[1,1].bar(np.arange(4),[r['mean_process_median_ms'] for r in rows],color=[COLORS[r['family']] for r in rows])
    axes[1,1].bar_label(bars,fmt='%.1f',padding=3);axes[1,1].set_xticks(np.arange(4),labels)
    axes[1,1].set(title='Measured 60,000-row GPU mapping',ylabel='Host-to-host milliseconds')
    tr=np.load(HERE/'artifacts/flow_trajectory.npz');states=tr['states']
    for j in range(48):axes[1,2].plot(states[:,j,0],states[:,j,1],color=COLORS['fm'],alpha=.35,lw=.8)
    axes[1,2].scatter(states[0,:48,0],states[0,:48,1],s=10,color='#666666',label='t=0')
    axes[1,2].scatter(states[-1,:48,0],states[-1,:48,1],s=10,color='#e69f00',label='t=1')
    axes[1,2].set(title='Actual ODE trajectories: first 48 sampled rows',xlabel='Normalized coordinate 1',ylabel='Normalized coordinate 2');axes[1,2].legend()
    for ax in axes.ravel():ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(HERE/'analysis/fm_control.png',dpi=180);fig.savefig(HERE/'analysis/fm_control.pdf');plt.close(fig)

def report(s,t,ad):
    pca_mixture=[r for r in s['mixtures'] if r['baseline']=='pca']
    lead='本轮高斯化 FM '+('在部分查询路径达到预定收益门槛。' if any(r['passes_5pct_gate'] for r in pca_mixture) else '没有达到相对 PCA 的预定查询收益门槛。')
    lead+='六阈值等查询权重下，FM 相对 PCA 的耗时变化为 '+ '，'.join(f"{r['method']} {-100*r['saving_fraction']:+.3f}%" for r in pca_mixture)+'。'
    lines=['# 实际 Flow Matching 布局对照','',lead,'',
        '目录：`gpu-host-8:@TENSORJOIN_ROOT@/tensorjoin_20260908_fm_control`。所有性能比值来自本轮 GPU4；旧实验只读。','',
        '## 本轮回答的问题','',
        '本轮实际训练时间相关的 32 维速度场，并用 Heun ODE 积分得到布局坐标。'
        '比较对象是同一批 OT 配对、相近参数量、全部参数重训的一次前向 MLP，以及 PCA 排序。'
        '此前小网络的负结果不能推出 FM 的结果；以下结论来自本轮直接测量。','',
        '本轮选择“把数据输运到各向同性高斯，再分组”的 FM 策略。分布拟合成功与原始距离界的剪枝效率是两个指标。'
        '模型坐标只决定排列，可靠判界仍使用原始 PCA32 包围盒；因此没有把非线性输运后的距离当作原空间距离。','',
        '| 布局 | 参数数 | 选定分组规则 | 验证剪枝率 | 复用测试行剪枝率 | 全量剪枝率 |',
        '|---|---:|---|---:|---:|---:|']
    for f in FAMILIES:
        sel=t['selection']['pca'] if f=='pca' else t['selection']['models'][f]
        lines.append(f"| {f} | {sel.get('parameters','—')} | {sel['rule']} | {100*sel['validation']['mean_pair_pruning']:.3f}% | {100*s['geometry'][f]['test']['mean_pair_pruning']:.3f}% | {100*s['geometry'][f]['full']['mean_pair_pruning']:.3f}% |")
    lines+=['','比例是六个既定阈值的等权平均；每个阈值按含自对的无序点对数计算。'
        '40,000/10,000/10,000 划分沿用前两轮，测试集已经在此前报告中出现；这是适应性后续实验，不能称为全新独立留出。'
        'PCA 也使用全部无标签数据。此次选型没有查看测试/全量几何或查询计时。','',
        '## FM 确实训练并执行了什么','',
        '设归一化原 PCA32 坐标为 u、目标高斯为 g，批大小 256。每批用 FP32 距离矩阵和 CPU 最小费用一对一指派得到 minibatch OT 配对；'
        '不是全数据全局 OT。采样 t∈[0,1]，令 z_t=(1−t)u+t g，最小化 ||vθ(z_t,t)−(g−u)||²/σ_g²。'
        '部署时从观测 u 出发求解 dz/dt=vθ(z,t)，得到 32 维终点。路径条件噪声取 0。'
        '公式依据 [Flow Matching](https://arxiv.org/abs/2210.02747) 与 [minibatch OT-CFM](https://arxiv.org/abs/2302.00482)；具体布局选择是本实验假设。','',
        'FM 为 33→128→128→32 SiLU，24,992 参数；直接映射为 u+rθ(u)，32→128→128→32 SiLU，24,864 参数。'
        '两者初始隐藏层匹配、输出层为零；FM 的时间输入列初始为零。全部层参与 Adam 训练，而不是仅优化读出层。'
        '每类三个种子，每种子 4,000 步，学习率 0.001；双方逐步消费完全相同的 OT 配对。','',
        '每个检查点允许第一坐标排序或按最大方差坐标递归平衡分块，每块 64 行。'
        'PCA 也允许两种规则。FM 另试 8/16/32 步 Heun；验证剪枝率选型。'
        '直接映射有 18 个候选，FM 有 54 个，后者搜索预算更大。18 份真实训练权重及全部 72 次候选评估均保留。','',
        '| 模型 | 选定种子 / 更新数 | 固定验证回归损失：初始→选定 | 高斯拟合 sliced-W2²/σ_g² |',
        '|---|---|---:|---:|']
    for f in ['direct','fm']:
        sel=t['selection']['models'][f];ck=next(c for c in t['checkpoints'] if c['name']==sel['name'])
        lines.append(f"| {f} | {sel['seed']} / {sel['step']} | {ck['initial_validation_loss']:.4f} → {ck['validation_loss']:.4f} | {sel['distribution']['sliced_w2_squared_normalized']:.6f} |")
    base=t['gaussianization_baseline']
    lines+=['',f"原始归一化特征的 sliced-W2²/σ_g² 为 {base['sliced_w2_squared_normalized']:.6f}，两批独立高斯样本的有限采样基线为 {base['gaussian_sampling_floor']:.6f}。"
        '该诊断使用 64 个固定投影方向，不等于完整 32 维 W2。两种回归损失预测目标不同，不用其数值大小判定哪种模型更好。'
        'FM 在此分布指标上优于直接回归，但其选定布局剪枝率仍低于 PCA。'
        '在固定 8 步、第一坐标排序下，三个 FM 种子训练到 4000 步的分布误差进一步降至 0.00214–0.00285，'
        '验证剪枝率却降至 3.19%–4.31%；完整候选数据保存在 candidate_evaluations.csv。'
        '这一观察仍不足以排除更长训练或其他模型设计的可能。','',
        '## ODE 数值与实际映射成本','',
        '| 模型 | Heun 步数 | 实际网络求值 NFE | 6 万行映射 ms | 三个进程中位数 ms | 稠密层 GFLOP |',
        '|---|---:|---:|---:|---|---:|']
    for r in s['mapping']:
        lines.append(f"| {r['family']} | {r['solver_steps']} | {r['nfe']} | {r['mean_process_median_ms']:.3f} | "+' / '.join(f'{v:.3f}' for v in r['process_median_ms'])+f" | {r['dense_layer_flops']/1e9:.3f} |")
    sel=t['selection']['models']['fm']
    lines+=['',f"实际选定 {sel['solver_steps']} 步、{sel['nfe']} 次速度场调用。以上是 GPU4 上真实完整 32 维映射，"
        '包含 FP64 特征转 FP32、H2D、归一化、全部 ODE 调用和终点 D2H；模型已加载，不含 PCA/分组/下界构建。'
        '每个进程预热后保留七次，不删除慢样本。FLOP 只计稠密层乘加，不代表实际吞吐。','',
        '| 步数 | 对 64 步终点的相对 RMS 差 | 正向再反向的相对 RMS 差 |',
        '|---|---:|---:|']
    for r in ad['flow_diagnostics']['solvers']:lines.append(f"| {r['steps']} | {r['endpoint_relative_rms_vs_64']:.6g} | {r['reverse_relative_rms']:.6g} |")
    lines+=['','数值诊断使用固定 1,024 行；64 步只是更细步长对照，不是精确解析解，也不据此重新选型。'
        f"同一输入在 t=0.25 与 0.75 的速度差 RMS 为 {ad['flow_diagnostics']['time_dependence_rms']:.6g}，说明模型使用了时间输入。"
        '零速度场恢复 PCA 验证剪枝率；Heun 另通过常向量场和线性场解析解、误差收敛和逆向积分检查。','',
        '## 完整查询计时','',
        '| 阈值 | 路径 | PCA ms | 直接映射 ms | FM ms | 原始全扫描 ms | FM 布局无剪枝 ms |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in s['primary']:
        lines.append(f"| {r['cell']} | {r['method']} | {r['pca_geo_ms']:.3f} | {r['direct_geo_ms']:.3f} | {r['fm_geo_ms']:.3f} | {r['original_full_ms']:.3f} | {r['fm_full_ms']:.3f} |")
    lines+=['','这些查询复用 CPU 排列、重排数据及下界矩阵；包含阈值掩码、tile 列表、GPU 输入/metadata、全部计算与精化、'
        '原始 ID 恢复、CUB 完整结果排序和 CPU 下载。查询不重复执行网络或 ODE，因此这一比较已经允许学习布局充分复用。','',
        '| 路径 | FM 相对基线 | 六阈值平均耗时节省 [95% CI] | 三进程节省 | 预定 5% 门槛 |',
        '|---|---|---:|---|---|']
    for r in s['mixtures']:
        lo,hi=r['saving_ci95'];ps=' / '.join(f'{100*v:.2f}%' for v in r['process_savings'])
        lines.append(f"| {r['method']} | {r['baseline']} | {100*r['saving_fraction']:.3f}% [{100*lo:.3f}%, {100*hi:.3f}%] | {ps} | "+('通过' if r['passes_5pct_gate'] else '未通过')+' |')
    lines+=['','负节省表示 FM 更慢。点估计是三个进程各自四次重复中位数的均值；六阈值等查询次数加权。'
        '20000 次分层 bootstrap 成对重采样进程和重复。预定门槛为节省的 95% CI 下界至少 5%，并且三个进程方向都为正。'
        '只有三个进程簇，区间只反映本次实验内重复性；各阈值全部区间见 comparisons.csv，不挑最好单格下结论。','',
        '## 训练、建造与摊销','',
        '| 模型 | 选定检查点训练秒数 | 其中 OT 配对秒数 | 三种子全部训练＋初始化秒数 | 检查点评估秒数 |',
        '|---|---:|---:|---:|---:|']
    for f in ['direct','fm']:
        c=s['training_costs'][f]
        lines.append(f"| {f} | {c['selected_training_seconds']:.3f} | {c['selected_pairing_seconds']:.3f} | {c['all_seed_training_seconds']:.3f} | {c['all_seed_evaluation_seconds']:.3f} |")
    lines+=['','每类独立部署都应付全额 OT 配对成本，不能因为实验共享配对就除以二。'
        f"本实验两个模型共享配对的实际联合训练/评估阶段耗时 {s['actual_joint_training_and_selection_wall_seconds']:.3f} 秒。"
        '该墙钟口径不含公共 CPU 准备、导入和守护等待；分项账单不含少量 Python/设置开销。'
        '“选定检查点训练”是事后已知选择的重训成本下限，不代表完成全部搜索的成本。','',
        '| 布局 | 实际完整布局建造平均秒数 |',
        '|---|---:|']
    for f in FAMILIES:lines.append(f"| {f} | {s['build_seconds'][f]:.4f} |")
    lines+=['','建造包含新做 PCA/特征、真实模型映射（FM 含 ODE）、分组、原 PCA32 保守下界及原始输入重排；模型已训练。'
        '每个确认进程实际重建 18 次，并逐位核对排列和下界。以下是完整重建后查询的实测总耗时。','',
        '| 布局 | 阈值 | 路径 | 实测建造＋查询秒数 |',
        '|---|---|---|---:|']
    for r in s['cold_queries']:lines.append(f"| {r['family']} | {r['cell']} | {r['method']} | {r['build_plus_query_seconds_mean']:.4f} |")
    lines+=['','摊销按“训练成本差＋建造成本差”除以每次查询节省，分别保存选定检查点重训和全部种子/评估两种账单。'
        '没有正查询节省时记为 null；增加复用次数不会消除负的查询收益。估计见 summary.json 和 mixtures.csv。'
        '这些是基于实测分项的代数估算，不是重新执行数千次查询的成本测量。','',
        '## 解释范围与诊断','',
        '| 布局 | 全量平均 PC1 块宽度 | 全量平均 PCA32 包围盒对角线 |',
        '|---|---:|---:|']
    for f in FAMILIES:
        g=s['geometry'][f]['full'];lines.append(f"| {f} | {g['mean_pc1_block_width']:.6f} | {g['mean_block_bbox_diagonal']:.6f} |")
    lines+=['','事后几何诊断表明，FM 确实改变了分组，但这些块在原始判界坐标中的范围更宽。'
        '这与剪枝率下降相符；它不是对训练目标与性能之间因果关系的独立证明。'
        '直接回归的输出方差收缩且分布拟合差；FM 改善了这一点，却没有使固定原坐标 AABB 更容易分离。','',
        '本轮结果只覆盖这套高斯化 OT-CFM、32 维公共特征、有限网络/训练预算和两种分组规则。'
        '它不能否定直接优化可证剪枝率的 FM、其他目标分布、更大模型/训练预算、不同数据或其他可靠判界结构。'
        '尤其不能把“FM 可以表达比小网络复杂的输运”直接等同于“高斯化终点必然带来更好的块剪枝”。','',
        '## 输出正确性与归档','',
        f"- {s['reference_occupancy_checks']} 个测试/全量布局阈值组合均未剪掉 FP64 参考命中。",
        '- 75 次准入完整输出逐项匹配参考数组；180 次预热、720 次保留计时、54 次实际重建查询均核对完整输出数量与 SHA256，合计 1029 次。',
        '- 算术内核、原始 ID 映射及 CUB 输出保持继承的冻结身份；数值范围见 inherited/NUMERICAL_SCOPE.md，不主张无条件 exact-real。',
        f"- 训练、准入和三个确认进程共五次 GPU4 守护均通过，最大设备占用 {s['max_device_used_mib']} MiB。",
        '- 训练前冻结 PROTOCOL.md、METHODS.md、源代码和准备输入；选型后保存模型哈希，再冻结全部计时依赖。',
        '- 结果文件：results/training.json、artifacts/model_selection.json、artifacts/admission.json、results/confirm_*.json、raw/。',
        '- analysis/summary.json、comparisons.csv、mixtures.csv、raw_times.csv、mapping.csv、cold_queries.csv 可逐项复核统计。',
        '- finalize_audit.py 在 CPU 上复核冻结依赖、18 份权重、全部输出和守护记录，生成 output_audit.json、环境/最终 GPU 状态和 archive_manifest.json。',
        '- CPU src/prepare.py → CPU src/preflight.py → 守护下 src/train.py → 守护下 src/admit.py → run_campaign.py。复现使用新目录，禁止覆盖既有结果。',
        '- python analyze.py 重建报告与独立 PNG/PDF；最终归档文件可对照 archive_manifest.json 校验。','',
        '![FM 对照](analysis/fm_control.png)','']
    (HERE/'REPORT.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
