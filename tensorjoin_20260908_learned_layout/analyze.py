"""Reproducible statistics for the frozen one-pass learned-layout experiment."""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
CELLS=['k4','k16','k64','original','k256','k1024']
METHODS=['F8','F16']
CONFIGS=['original_full','pca_geo','linear_geo','mlp_geo','mlp_full']
FAMILIES=['pca','linear','mlp']

def read(rel):return json.loads((HERE/rel).read_text())
def ci(values):return np.quantile(values,[.025,.975]).tolist()

def main():
    out=HERE/'analysis';out.mkdir(exist_ok=True)
    training=read('results/training.json')
    admission=read('artifacts/admission.json')
    runs=[read(f'results/confirm_{i:02d}.json') for i in range(3)]
    guards=[read('results/training_a0_guard.json'),read('results/admission_a0_guard.json')]+[
        read(f'results/confirm_{i:02d}_a0_guard.json') for i in range(3)]
    assert training['pass_'] and admission['pass'] and all(r['pass'] for r in runs+guards)
    samples=[s for r in runs for s in r['samples'] if s['phase']=='retained']
    assert len(samples)==720
    a=np.full((3,4,6,2,5),np.nan)
    for s in samples:
        ix=(s['process'],s['repeat'],CELLS.index(s['cell']),METHODS.index(s['method']),CONFIGS.index(s['configuration']))
        assert np.isnan(a[ix]);a[ix]=s['seconds']
    assert np.isfinite(a).all()
    proc=np.median(a,axis=1)
    point=proc.mean(axis=0)
    rng=np.random.default_rng(2026090817)
    pp=rng.integers(0,3,(20000,3))
    rr=rng.integers(0,4,(20000,3,4))
    boot=np.median(a[pp[:,:,None],rr],axis=2).mean(axis=1)
    build_calls=[s for r in runs for s in r['single_queries']]
    builds={f:float(np.mean([s['build_wall_seconds'] for s in build_calls if s['family']==f])) for f in FAMILIES}
    table=[]
    for cj,cell in enumerate(CELLS):
        for mj,method in enumerate(METHODS):
            row=dict(cell=cell,method=method)
            for j,config in enumerate(CONFIGS):row[config+'_ms']=float(point[cj,mj,j]*1000)
            for family,j in [('linear',2),('mlp',3)]:
                ratio=point[cj,mj,1]/point[cj,mj,j]
                interval=ci(boot[:,cj,mj,1]/boot[:,cj,mj,j])
                process_ratios=(proc[:,cj,mj,1]/proc[:,cj,mj,j]).tolist()
                row[family+'_speedup_vs_pca']=float(ratio)
                row[family+'_speedup_ci95']=interval
                row[family+'_process_speedups']=process_ratios
                row[family+'_passes_5pct_gate']=bool(interval[0]>=1/.95 and min(process_ratios)>1)
                saving=point[cj,mj,1]-point[cj,mj,j]
                selected=training['selection']['models'][family]
                extra=training['guide_seconds']+selected['training_seconds']+selected['initialization_seconds']+builds[family]-builds['pca']
                row[family+'_modeled_break_even_vs_pca']=max(0,math.ceil(extra/saving)) if saving>0 else None
            table.append(row)
    mixtures=[]
    for mj,method in enumerate(METHODS):
        times=point[:,mj,:].mean(axis=0)
        bb=boot[:,:,mj,:].mean(axis=1)
        for family,j in [('linear',2),('mlp',3)]:
            saving=1-times[j]/times[1]
            interval=ci(1-bb[:,j]/bb[:,1])
            process_savings=(1-proc[:,:,mj,j].mean(axis=1)/proc[:,:,mj,1].mean(axis=1)).tolist()
            mixtures.append(dict(method=method,family=family,pca_ms=float(times[1]*1000),
                learned_ms=float(times[j]*1000),saving_fraction=float(saving),saving_ci95=interval,
                process_savings=process_savings,passes_5pct_gate=bool(interval[0]>=.05 and min(process_savings)>0)))
    mapping=[]
    for family in ['linear','mlp']:
        for device in ['cpu','cuda']:
            rows=[next(s for s in r['mapping'] if s['family']==family and s['device']==device) for r in runs]
            med=[s['median_seconds'] for s in rows]
            mapping.append(dict(family=family,device=device,parameters=rows[0]['parameters'],
                mean_process_median_ms=float(np.mean(med)*1000),process_median_ms=[x*1000 for x in med],
                dense_layer_flops=rows[0]['dense_layer_flops'],input_rows=60000,features=32))
    cold=[]
    for family in FAMILIES:
        for cell in ['k4','original','k1024']:
            for method in METHODS:
                rows=[s for s in build_calls if s['family']==family and s['cell']==cell and s['method']==method]
                assert len(rows)==3
                cold.append(dict(family=family,cell=cell,method=method,
                    build_plus_query_seconds_mean=float(np.mean([r['seconds'] for r in rows])),
                    build_seconds_mean=float(np.mean([r['build_wall_seconds'] for r in rows])),
                    training_included=False))
    checks=sum(len(g['reference_checks']) for family in admission['geometry'].values() for g in family.values())
    full_calls=(len(admission['samples'])+len(admission['diagnostics'])+len(admission['build_admission'])+
        sum(len(r['samples'])+len(r['single_queries']) for r in runs))
    summary=dict(primary=table,mixtures=mixtures,mapping=mapping,build_seconds=builds,
        cold_queries=cold,geometry=admission['geometry'],selection=training['selection'],
        guide_seconds=training['guide_seconds'],retained_calls=len(samples),warmups=180,
        admission_calls=75,build_plus_query_calls=len(build_calls),full_gpu_calls=full_calls,
        reference_occupancy_checks=checks,max_device_used_mib=max(g['max_device_used_mib'] for g in guards),
        bootstrap=dict(draws=20000,seed=2026090817,unit='process then paired repetitions'),
        scope='Within-dataset heldout rows, shared transductive PCA; one training proxy and fixed bound axes; not FM.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    for name,rows in [('summary',table),('mixtures',mixtures),('mapping',mapping),('cold_queries',cold)]:
        with (out/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    keys=['process','repeat','cell','method','configuration','seconds','executed_tiles','output_count','output_sha256']
    with (out/'raw_times.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(samples)
    plot(summary,training)
    report(summary,training,admission)
    print(json.dumps(summary,indent=2))

def plot(s,training):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    x=np.arange(3)
    for offset,split,color in [(-.18,'test','#56b4e9'),(.18,'full','#0072b2')]:
        vals=[100*s['geometry'][f][split]['mean_pair_pruning'] for f in FAMILIES]
        bars=axes[0,0].bar(x+offset,vals,width=.34,label=split,color=color)
        axes[0,0].bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
    axes[0,0].set_xticks(x,FAMILIES);axes[0,0].set_ylabel('Mean safely pruned pairs (%)')
    axes[0,0].set_title('Geometry quality under the same bound');axes[0,0].legend()
    for family,color in [('linear','#e69f00'),('mlp','#009e73')]:
        histories=[r['history'] for r in training['models'] if r['family']==family]
        step=[r['step'] for r in histories[0]]
        vals=np.asarray([[r['loss'] for r in h] for h in histories])
        axes[0,1].plot(step,vals.mean(axis=0),label=family,color=color)
        axes[0,1].fill_between(step,vals.min(axis=0),vals.max(axis=0),alpha=.2,color=color)
    axes[0,1].set_title('Proxy training loss: all three seeds')
    axes[0,1].set_xlabel('Updates');axes[0,1].set_ylabel('Distance-stress objective');axes[0,1].legend()
    xx=np.arange(6)
    for j,method in enumerate(METHODS):
        rows=[r for r in s['primary'] if r['method']==method]
        yy=[r['mlp_speedup_vs_pca'] for r in rows]
        cc=np.asarray([r['mlp_speedup_ci95'] for r in rows])
        axes[1,0].vlines(xx+(j-.5)*.1,cc[:,0],cc[:,1],color=['#0072b2','#e69f00'][j])
        axes[1,0].plot(xx+(j-.5)*.1,yy,'o-',label=method,color=['#0072b2','#e69f00'][j])
    axes[1,0].axhline(1,color='black',lw=.8);axes[1,0].set_xticks(xx,CELLS)
    axes[1,0].set_title('Repeated complete queries: PCA / MLP')
    axes[1,0].set_ylabel('Speedup (>1 favors MLP)');axes[1,0].legend()
    mapping=s['mapping'];labels=[f"{r['family']}\n{r['device']}" for r in mapping]
    vals=[r['mean_process_median_ms'] for r in mapping]
    bars=axes[1,1].bar(np.arange(4),vals,color=['#999999','#56b4e9','#999999','#56b4e9'])
    axes[1,1].bar_label(bars,fmt='%.2f',padding=3)
    axes[1,1].set_xticks(np.arange(4),labels)
    axes[1,1].set_title('One-pass scores for 60,000 points')
    axes[1,1].set_ylabel('Host-to-host milliseconds (preloaded model)')
    for ax in axes.ravel():ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(HERE/'analysis/learned_layout.png',dpi=180)
    fig.savefig(HERE/'analysis/learned_layout.pdf')
    plt.close(fig)

def report(s,training,admission):
    lines=['# 一次前向小网络布局验证','',
        '远程目录：`gpu-host-8:@TENSORJOIN_ROOT@/tensorjoin_20260908_learned_layout`。',
        '所有执行比较都在本轮 GPU0 内完成。训练/模型选型、准入和三个确认进程分别经独占守护。','',
        '本轮验证上条建议中的“小网络直接输出排序坐标”，没有训练 flow matching，也没有 ODE 推理。'
        '测得的成本不能直接作为多步 FM、向量场网络或跨点注意力模型的成本。','',
        '## 结果判断','',
        '当前距离保持代理目标没有产生超过 PCA 的可用布局。网络本身规模很小，'
        '但学到的排序显著削弱了固定 PCA 坐标包围盒的判界能力。'
        '这支持停止本轮具体训练目标，不证明所有神经布局或 FM 都无效。','',
        '| 布局 | 参数数 | 验证行平均剪枝 | 测试行平均剪枝 | 完整数据平均剪枝 |',
        '|---|---:|---:|---:|---:|']
    for family in FAMILIES:
        val=training['baseline_validation'] if family=='pca' else training['selection']['models'][family]['validation']
        params='—' if family=='pca' else str(training['selection']['models'][family]['parameters'])
        lines.append(f"| {family} | {params} | {100*val['mean_pair_pruning']:.4f}% | "
            f"{100*s['geometry'][family]['test']['mean_pair_pruning']:.4f}% | {100*s['geometry'][family]['full']['mean_pair_pruning']:.4f}% |")
    lines += ['', '剪枝为六个既定阈值的等权平均、按无序点对数（含自对）加权的比例。'
        '完整数据是实际部署输入，包含训练和验证行；测试行未参与梯度或模型选型。'
        '共同 PCA 特征使用全部无标签坐标，因此这是同一数据集内的留出检查，不是独立数据分布泛化。','',
        '## 模型与训练成本','',
        '共同输入为已有 32 维 PCA 特征。MLP 为 32→64→64→1、ReLU、PCA 分数残差，6337 个参数；'
        '线性残差只有 33 个参数。输出只用于排序，可靠判界始终使用原有 PCA32 投影误差包络。','',
        f"为 8192 个训练锚点构造 top-8 训练邻居的 GPU GUIDE 计算耗时 {training['guide_seconds']:.3f} 秒，"
        f"共计算 {training['guide_distance_pairs']:,} 个训练引导距离。该图没有用于连接候选生成或正确性判定。",'',
        '| 模型 | 选定随机种子/更新数 | 该检查点训练秒数 | 各种子完整 1200 步训练秒数 |',
        '|---|---|---:|---|']
    for family in ['linear','mlp']:
        chosen=training['selection']['models'][family]
        times=[r['training_seconds'] for r in training['models'] if r['family']==family]
        lines.append(f"| {family} | {chosen['seed']} / {chosen['step']} | {chosen['training_seconds']:.3f} | "+
                     ' / '.join(f'{t:.3f}' for t in times)+' |')
    lines += ['', '训练时间在 CUDA 同步边界计时，剔除检查点保存与验证暂停；'
        '共同 PCA 时间保存在 artifacts/features.json；图准备、初始化、训练历史和验证时间保存在 results/training.json。'
        '每类模型尝试三个随机种子和三个检查点，共保留 18 个候选；始终按验证剪枝率选型，没有用测试/完整数据或 GPU 查询时间选模型。','',
        f"六个完整训练运行累计更新耗时 {sum(m['training_seconds'] for m in training['models']):.3f} 秒；"
        f"图准备、全部训练和验证所在计时段共 {training['ended']-training['started']:.3f} 秒。"
        '这不含共同 PCA 准备、守护等待、导入或后续查询验证。','',
        '## 一次映射与建造布局的成本','',
        '| 模型 | 执行设备 | 6 万行一次映射 ms | 三个进程中位数 ms |',
        '|---|---|---:|---|']
    for row in s['mapping']:
        lines.append(f"| {row['family']} | {row['device']} | {row['mean_process_median_ms']:.3f} | "+
            ' / '.join(f'{v:.3f}' for v in row['process_median_ms'])+' |')
    lines += ['', '映射时间包括从 FP64 公共特征转换为 FP32、GPU 路径的 H2D、模型前向与分数 D2H；'
        '模型权重已加载。CPU 路径使用四线程。表中不含 PCA 拟合、分组排序和距离界构造。'
        'MLP 稠密层的乘加约为 0.745 GFLOP/6 万行，仅是算术量估算；不是实际耗时或 FM 成本推算。','',
        '| 布局 | 本轮完整布局建造平均秒数（已训练模型） |',
        '|---|---:|']
    for family in FAMILIES:lines.append(f"| {family} | {s['build_seconds'][family]:.4f} |")
    lines += ['', '建造包含重新拟合 PCA、特征计算、模型映射、稳定排序、64 行分块、'
        '保守投影包围盒与块对下界矩阵，以及原始输入重排。'
        '所有布局均删除了上一轮在 Cifar 上无贡献的球界/原坐标界/枢轴界；'
        'PCA 的六个保留掩码与上一轮完全一致。这里只报告本轮成本，不用新旧查询时间相除。','',
        '## 完整查询性能：首先对比 PCA','',
        '| 阈值 | 路径 | PCA ms | 线性 ms | MLP ms | PCA/MLP [95% CI] |',
        '|---|---|---:|---:|---:|---:|']
    for r in s['primary']:
        lo,hi=r['mlp_speedup_ci95']
        lines.append(f"| {r['cell']} | {r['method']} | {r['pca_geo_ms']:.3f} | {r['linear_geo_ms']:.3f} | "
            f"{r['mlp_geo_ms']:.3f} | {r['mlp_speedup_vs_pca']:.3f} [{lo:.3f}, {hi:.3f}] |")
    lines += ['', '大于 1 才表示 MLP 更快。计时复用 CPU 布局与下界矩阵，包含阈值掩码、tile 列表、'
        '新 GPU 输入和 metadata、全部计算/精化、原始 ID 恢复、CUB 完整结果排序及 CPU 下载。'
        '训练和布局建造不包含在这张表中；原始全扫描与 MLP 无剪枝对照也完整保存于 summary.csv。','',
        '| 路径 | 学习布局 | 六阈值等查询权重耗时节省 vs PCA [95% CI] | 5% 门槛 |',
        '|---|---|---:|---|']
    for r in s['mixtures']:
        lo,hi=r['saving_ci95']
        lines.append(f"| {r['method']} | {r['family']} | {100*r['saving_fraction']:.3f}% [{100*lo:.3f}%, {100*hi:.3f}%] | "+
                     ('通过' if r['passes_5pct_gate'] else '未通过')+' |')
    lines += ['', '负的节省表示比 PCA 更慢。三个独立进程，各配置每进程四个保留重复；'
        '按进程和成对重复进行 20000 次分层 bootstrap。只有三个进程簇，置信区间限于本轮重复性。','',
        '## 单次重建后查询与摊销','',
        '| 布局 | 阈值 | 路径 | 实测建造＋完整查询秒数 |',
        '|---|---|---|---:|']
    for r in s['cold_queries']:
        lines.append(f"| {r['family']} | {r['cell']} | {r['method']} | {r['build_plus_query_seconds_mean']:.4f} |")
    lines += ['', '这些调用实际重新建造布局，但模型已训练；不把训练时间误称为已包含。'
        '相对 PCA 的摊销估算使用“图准备＋选定检查点训练＋额外建造时间”，除以每次查询节省；'
        '若学习布局本身的查询更慢，则复用更多次也不能靠摊销弥补。'
        '各格估算（没有正节省时为 null）保存在 summary.json；接近零的时间差不宜据此作部署决策。','',
        '## 为什么损失下降而剪枝变差','',
        '| 布局 | 完整数据平均 PC1 块宽度 | 平均 PCA32 包围盒对角线长度 |',
        '|---|---:|---:|']
    for family in FAMILIES:
        g=s['geometry'][family]['full']
        lines.append(f"| {family} | {g['mean_pc1_block_width']:.6f} | {g['mean_block_bbox_diagonal']:.6f} |")
    lines += ['', '这是事后诊断：PCA 排序使同一块在 PC1 方向非常窄；学习后的排序混合了多个方向，'
        '在固定 PCA 坐标轴上块区间变宽，更容易重叠。'
        '训练优化的是点对距离保持代理目标，并未直接优化可安全剪掉的块数。'
        '零残差网络控制恢复了 PCA 的验证剪枝率，排除了“评估器无法处理网络分数”的简单解释。','',
        '这一结果同时受训练代理目标、单一排序坐标和固定判界坐标轴约束。'
        '没有测试直接针对分块判界的目标、随线性排序方向旋转的包围盒、其他可证明界或 FM；'
        '不能把它扩大为所有学习布局的否定结论。','',
        '## 正确性、资源和复现','',
        f"- 测试行/完整数据的 {s['reference_occupancy_checks']} 个布局阈值组合均未剪掉参考命中。",
        f"- {s['admission_calls']} 次准入完整调用逐项匹配 FP64 参考；另有 {s['warmups']} 次预热、"
        f"{s['retained_calls']} 次保留计时、{s['build_plus_query_calls']} 次建造后查询；"
        f"合计 {s['full_gpu_calls']} 次完整输出数量和 SHA256 均匹配。",
        '- 算术核和共同输出实现保持冻结身份；新模型只改变排列，原始 FP64 判定合同不变。'
        '继承数值范围见 inherited/NUMERICAL_SCOPE.md，未主张无条件 exact-real。',
        f"- 五个守护进程均通过；最大设备占用 {s['max_device_used_mib']} MiB，未改时钟或终止其他任务。",
        '- `PROTOCOL.md`、`artifacts/training_freeze.json`、`model_selection.json`、`timing_freeze.json` 保存冻结依据。',
        '- `results/training.json` 保存全部 18 候选与训练曲线，`artifacts/*.pt` 保存模型权重。',
        '- `artifacts/admission.json`、`results/confirm_*.json`、`raw/` 保存几何、完整输出与原始计时。',
        '- `analysis/summary.json`、`summary.csv`、`raw_times.csv`、`mapping.csv`、`cold_queries.csv`。',
        '- `analysis/output_audit.json` 复核全部输出记录；`analysis/final_environment.json`、`analysis/final_gpu_state.json` 和 `archive_manifest.json` 保存环境与归档核验。',
        '- `python analyze.py` 重建本报告与 PNG/PDF 图；旧数据与数值核由原实验目录只读提供。',
        '- 从新目录复现：CPU `src/prepare.py` → 守护下 `src/train.py` → 守护下 `src/admit.py` → `run_campaign.py`；禁止覆盖已有结果。','',
        '![验证结果](analysis/learned_layout.png)','']
    (HERE/'REPORT.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
