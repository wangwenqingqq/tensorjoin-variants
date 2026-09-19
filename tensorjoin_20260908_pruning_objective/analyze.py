"""Reproducible statistics for the frozen direct block-pruning objective experiment."""
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
CONFIGS=['original_full','pca_geo','proxy_geo','direct_geo','direct_full']
FAMILIES=['pca','proxy','direct']

def read(rel):return json.loads((HERE/rel).read_text())
def ci(values):return np.quantile(values,[.025,.975]).tolist()

def main():
    out=HERE/'analysis';out.mkdir(exist_ok=True)
    training=read('results/training.json')
    historical=read('inherited/previous_training.json')
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
    rng=np.random.default_rng(2026090824)
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
            for family,j in [('proxy',2),('direct',3)]:
                ratio=point[cj,mj,1]/point[cj,mj,j]
                interval=ci(boot[:,cj,mj,1]/boot[:,cj,mj,j])
                process_ratios=(proc[:,cj,mj,1]/proc[:,cj,mj,j]).tolist()
                row[family+'_speedup_vs_pca']=float(ratio)
                row[family+'_speedup_ci95']=interval
                row[family+'_process_speedups']=process_ratios
                eligible=family!='direct' or (not training['selection']['models']['direct']['zero_initialization']
                    and not admission['geometry']['direct']['full']['permutation_equals_pca'])
                row[family+'_passes_5pct_gate']=bool(interval[0]>=1/.95 and min(process_ratios)>1 and eligible)
                saving=point[cj,mj,1]-point[cj,mj,j]
                selected=training['selection']['models'][family]
                guide=historical['guide_seconds'] if family=='proxy' else 0.
                extra=guide+selected['training_seconds']+selected['initialization_seconds']+builds[family]-builds['pca']
                row[family+'_modeled_selected_checkpoint_break_even_vs_pca']=max(0,math.ceil(extra/saving)) if saving>0 and eligible else None
                if family=='direct':
                    total=sum(m['total_seconds'] for m in training['models'])+builds[family]-builds['pca']
                    row['direct_modeled_full_search_break_even_vs_pca']=max(0,math.ceil(total/saving)) if saving>0 and eligible else None
            table.append(row)
    mixtures=[]
    for mj,method in enumerate(METHODS):
        times=point[:,mj,:].mean(axis=0)
        bb=boot[:,:,mj,:].mean(axis=1)
        for family,j in [('proxy',2),('direct',3)]:
            saving=1-times[j]/times[1]
            interval=ci(1-bb[:,j]/bb[:,1])
            process_savings=(1-proc[:,:,mj,j].mean(axis=1)/proc[:,:,mj,1].mean(axis=1)).tolist()
            eligible=family!='direct' or (not training['selection']['models']['direct']['zero_initialization']
                and not admission['geometry']['direct']['full']['permutation_equals_pca'])
            mixtures.append(dict(method=method,family=family,pca_ms=float(times[1]*1000),
                learned_ms=float(times[j]*1000),saving_fraction=float(saving),saving_ci95=interval,
                process_savings=process_savings,passes_5pct_gate=bool(interval[0]>=.05 and min(process_savings)>0 and eligible)))
        mixtures.append(dict(method=method,family='direct_vs_proxy',pca_ms=None,
            reference_ms=float(times[2]*1000),learned_ms=float(times[3]*1000),
            saving_fraction=float(1-times[3]/times[2]),saving_ci95=ci(1-bb[:,3]/bb[:,2]),
            process_savings=(1-proc[:,:,mj,3].mean(axis=1)/proc[:,:,mj,2].mean(axis=1)).tolist(),
            passes_5pct_gate=None))
    mapping=[]
    for family in ['proxy','direct']:
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
    reference=read('inherited/reference_manifest.json')['cells']
    pca_bound=np.load(HERE/'artifacts/pca_bounds.npz')['combined']
    direct_bound=np.load(HERE/'artifacts/direct_bounds.npz')['combined']
    mask_changes={cell:int(np.count_nonzero((pca_bound<=reference[cell]['threshold']['T']+2**-20)
        !=(direct_bound<=reference[cell]['threshold']['T']+2**-20))) for cell in CELLS}
    full_calls=(len(admission['samples'])+len(admission['diagnostics'])+len(admission['build_admission'])+
        sum(len(r['samples'])+len(r['single_queries']) for r in runs))
    summary=dict(primary=table,mixtures=mixtures,mapping=mapping,build_seconds=builds,
        cold_queries=cold,geometry=admission['geometry'],selection=training['selection'],
        guide_seconds=training['guide_seconds'],retained_calls=len(samples),warmups=180,
        admission_calls=75,build_plus_query_calls=len(build_calls),full_gpu_calls=full_calls,
        reference_occupancy_checks=checks,max_device_used_mib=max(g['max_device_used_mib'] for g in guards),
        training_search=dict(total_proposals=training['total_proposals'],
            selected_zero_initialization=training['selection']['models']['direct']['zero_initialization'],
            total_search_seconds=sum(m['training_seconds'] for m in training['models']),
            total_runs_seconds=sum(m['total_seconds'] for m in training['models']),
            selected_search_seconds=training['selection']['models']['direct']['training_seconds'],
            accepted_parameter_changes=sum(m['accepted'] for m in training['models']),
            proposals_improving_initial=sum(row['reward']>m['history'][0]['reward']
                for m in training['models'] for row in m['proposals']),
            best_proposal_pruning=max(row['mean_pair_pruning']
                for m in training['models'] for row in m['proposals']),
            parameters_total=6337,parameters_searched=64),
        direct_vs_pca_changed_tile_masks=mask_changes,
        bootstrap=dict(draws=20000,seed=2026090824,unit='process then paired repetitions'),
        scope='Adaptive within-dataset follow-up; reused test rows; fixed hidden features and direct readout search; not FM.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    for name,rows in [('summary',table),('mixtures',mixtures),('mapping',mapping),('cold_queries',cold)]:
        with (out/(name+'.csv')).open('w') as f:
            keys=list(dict.fromkeys(k for row in rows for k in row))
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    keys=['process','repeat','cell','method','configuration','seconds','executed_tiles','output_count','output_sha256']
    with (out/'raw_times.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(samples)
    proposals=[dict(seed=m['seed'],initial_pruning=m['history'][0]['train_pruning'],**row)
               for m in training['models'] for row in m['proposals']]
    with (out/'proposals.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(proposals[0]));w.writeheader();w.writerows(proposals)
    plot(summary,training)
    report(summary,training,admission)
    print(json.dumps(summary,indent=2))


def plot(s,training):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    xx=np.arange(3)
    for offset,split,color in [(-.18,'test','#56b4e9'),(.18,'full','#0072b2')]:
        values=[100*s['geometry'][f][split]['mean_pair_pruning'] for f in FAMILIES]
        bars=axes[0,0].bar(xx+offset,values,width=.34,label=split,color=color)
        axes[0,0].bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
    axes[0,0].set_xticks(xx,['PCA','old proxy','direct objective'])
    axes[0,0].set_title('Same certified bound: pruning quality')
    axes[0,0].set_ylabel('Mean safely pruned pairs (%)');axes[0,0].legend()
    for model,color in zip(training['models'],['#0072b2','#e69f00','#009e73']):
        initial=model['history'][0]['train_pruning']
        axes[0,1].scatter([r['proposal'] for r in model['proposals']],
            [100*(r['mean_pair_pruning']-initial) for r in model['proposals']],
            s=7,alpha=.4,label=str(model['seed']),color=color)
    axes[0,1].axhline(0,color='black',lw=.8)
    axes[0,1].set_title('Every proposed parameter change (all seeds)')
    axes[0,1].set_xlabel('Proposal number per seed')
    axes[0,1].set_ylabel('Training pruning change vs initial (pp)')
    axes[0,1].legend(fontsize=8)
    xx=np.arange(6)
    for j,method in enumerate(METHODS):
        rows=[r for r in s['primary'] if r['method']==method]
        values=[r['direct_speedup_vs_pca'] for r in rows]
        intervals=np.asarray([r['direct_speedup_ci95'] for r in rows])
        color=['#0072b2','#e69f00'][j]
        axes[1,0].vlines(xx+(j-.5)*.1,intervals[:,0],intervals[:,1],color=color)
        axes[1,0].plot(xx+(j-.5)*.1,values,'o-',label=method,color=color)
    axes[1,0].axhline(1,color='black',lw=.8)
    axes[1,0].set_xticks(xx,CELLS);axes[1,0].legend()
    axes[1,0].set_title('Repeated complete queries: PCA / direct')
    axes[1,0].set_ylabel('Speedup (>1 favors direct)')
    for j,method in enumerate(METHODS):
        rows=[r for r in s['primary'] if r['method']==method]
        values=[np.mean([r[f'{f}_geo_ms'] for r in rows]) for f in FAMILIES]
        bars=axes[1,1].bar(np.arange(3)+(j-.5)*.36,values,width=.34,
            color=['#0072b2','#e69f00'][j],label=method)
        axes[1,1].bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
    axes[1,1].set_xticks(np.arange(3),['PCA','old proxy','direct objective'])
    axes[1,1].set_title('Equal-six-query mixture, this round only')
    axes[1,1].set_ylabel('Complete query milliseconds');axes[1,1].legend()
    axes[1,1].set_ylim(0,max(r['learned_ms'] for r in s['mixtures'])*1.23)
    for ax in axes.ravel():ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(HERE/'analysis/pruning_objective.png',dpi=180)
    fig.savefig(HERE/'analysis/pruning_objective.pdf')
    plt.close(fig)


def report(s,training,admission):
    chosen=s['selection']['models']['direct']
    direct_mixtures=[r for r in s['mixtures'] if r['family']=='direct']
    passed=any(r['passes_5pct_gate'] for r in direct_mixtures)
    if chosen['zero_initialization']:
        verdict='选型最终保留了零残差的 PCA 初始化，没有学到超过 PCA 的排序。'
    elif passed:
        verdict='直接目标在本轮部分计时路径达到既定收益门槛；有效范围和摊销成本见下表。'
    else:
        verdict='直接目标选出了非零残差模型，但没有达到相对 PCA 的完整查询收益门槛。'
    lines=['# 直接面向块剪枝的训练目标验证','',verdict,'',
        '目录：`gpu-host-8:@TENSORJOIN_ROOT@/tensorjoin_20260908_pruning_objective`。'
        '本轮全部 GPU 对照在 GPU4 完成；原定 GPU0 在训练前被其他任务占用，切换依据和 UUID 在冻结协议中记录。','',
        '## 结论与范围','',
        '本轮直接用“真实排序、64 行分块后，保守界能剪掉的点对数”评价候选参数。'
        '保持 32→64→64→1 的 ReLU MLP 结构，固定随机隐藏层，只搜索最后 64 个输出权重。'
        '这是针对一种有限参数搜索的可行性实验，没有训练所有网络权重、FM 或 ODE。','',
        f"三个种子共评估 {s['training_search']['total_proposals']} 个非初始候选参数，"
        f"接受 {s['training_search']['accepted_parameter_changes']} 次参数变化；"
        f"有 {s['training_search']['proposals_improving_initial']} 个候选的训练剪枝率严格超过同种子的初始化。",
        '优化器变化与固定隐藏层也会影响结果，因此不能把相对上一轮的变化仅归因于目标函数。','',
        '| 布局 | 验证行剪枝率 | 复用测试行剪枝率 | 完整数据剪枝率 |',
        '|---|---:|---:|---:|']
    for family in FAMILIES:
        valid=training['baseline_validation'] if family=='pca' else s['selection']['models'][family]['validation']
        lines.append(f"| {family} | {100*valid['mean_pair_pruning']:.5f}% | "
            f"{100*s['geometry'][family]['test']['mean_pair_pruning']:.5f}% | "
            f"{100*s['geometry'][family]['full']['mean_pair_pruning']:.5f}% |")
    lines+=['','剪枝率为六个既定阈值等权平均，按各块内无序点对数量加权，分母含自对。'
        'proxy 是上一轮冻结的距离保持目标 MLP，direct 是本轮直接剪枝目标选出的模型。','',
        '数据仍为 Cifar60000×512。40k 训练行仅用于参数搜索，10k 验证行选检查点。'
        '另 10k 测试行在上一轮已经看过，本轮是复用测试集的适应性后续实验；共同 PCA 还使用全部无标签坐标。'
        '不能称为全新独立留出或跨数据集泛化验证。选定模型冻结后才评价本轮测试/完整布局和查询性能。','',
        '## 搜索与成本','',
        '每个种子从零输出层开始。按 0.03、0.01、0.003、0.001、0.0003 的残差分数标准差做五轮坐标搜索，'
        '每轮随机遍历 64 个权重并试两个方向。每个候选都在全部 4 万训练行上构造同一 FP64 保守判界，'
        '以六阈值安全剪掉的点对整数总数为主目标；相同计数时优先较小权重范数。'
        '固定隐藏层缓存只用于训练加速，每个检查点都验证缓存分数与普通完整前向逐元素相同。','',
        '| 种子 | 候选数 | 接受变化数 | 初始/最终训练剪枝率 | 搜索秒数 |',
        '|---|---:|---:|---|---:|']
    for model in training['models']:
        lines.append(f"| {model['seed']} | {len(model['proposals'])} | {model['accepted']} | "
            f"{100*model['history'][0]['train_pruning']:.5f}% / {100*model['history'][-1]['train_pruning']:.5f}% | "
            f"{model['training_seconds']:.3f} |")
    lines+=['',f"累计搜索 {s['training_search']['total_search_seconds']:.3f} 秒，"
        f"三次搜索连同初始化和检查点验证共 {s['training_search']['total_runs_seconds']:.3f} 秒。"
        '搜索时间包含 GPU 分数计算、CPU 稳定排序和保守界评价，不含验证/保存暂停；'
        '不含共同 PCA 准备、进程导入、30 秒守护等待或后续查询实验。','',
        f"选定 `direct_s{chosen['seed']}_e{chosen['step']}`，"
        f"输出权重平方范数 {chosen['output_weight_squared_norm']:.9g}，"
        f"SHA256 `{chosen['checkpoint_sha256']}`。"
        '初始模型与每轮结束模型均是选型候选，共保存 18 个检查点；先比较验证剪枝率，平局选更少搜索步和较小种子。','',
        '## 与 PCA 的布局关系','',
        '| 指标（完整数据） | direct |', '|---|---:|',
        f"| 排列与 FP64 PCA 排序逐元素相同 | {s['geometry']['direct']['full']['permutation_equals_pca']} |",
        f"| 点保持原 PCA 块编号的比例 | {100*s['geometry']['direct']['full']['same_block_assignment_fraction']:.6f}% |",
        f"| 平均绝对排名变化 | {s['geometry']['direct']['full']['mean_absolute_rank_change']:.6f} |",
        f"| 六阈值 tile 掩码变化数 | {list(s['direct_vs_pca_changed_tile_masks'].values())} |",'',
        '零残差网络仍经过 FP32 特征/分数计算，而 PCA 基线使用原有 FP64 排序分数，极近点的并列顺序可能不同。'
        '不应把零模型的计时波动解释为学习带来的收益。','',
        '## 完整查询性能','',
        '| 阈值 | 路径 | PCA ms | 旧 proxy ms | direct ms | PCA/direct [95% CI] |',
        '|---|---|---:|---:|---:|---|']
    for row in s['primary']:
        lo,hi=row['direct_speedup_ci95']
        lines.append(f"| {row['cell']} | {row['method']} | {row['pca_geo_ms']:.3f} | "
            f"{row['proxy_geo_ms']:.3f} | {row['direct_geo_ms']:.3f} | "
            f"{row['direct_speedup_vs_pca']:.4f} [{lo:.4f}, {hi:.4f}] |")
    lines+=['','上表复用 CPU 布局和下界，但包含阈值掩码、tile 列表、新 GPU 输入/metadata、'
        '所有扫描与精化、原始 ID 恢复、CUB 完整排序及结果 CPU 下载。映射/训练/建造不在复用查询时间内。'
        '原始全扫描和 direct 无剪枝对照保存在 summary.csv。','',
        '| 路径 | 比较 | 六阈值等权耗时节省 [95% CI] | 相对 PCA 的 5% 门槛 |',
        '|---|---|---|---|']
    for row in s['mixtures']:
        lo,hi=row['saving_ci95']
        gate='—' if row['passes_5pct_gate'] is None else ('通过' if row['passes_5pct_gate'] else '未通过')
        comparison='direct vs proxy' if row['family']=='direct_vs_proxy' else row['family']+' vs PCA'
        lines.append(f"| {row['method']} | {comparison} | {100*row['saving_fraction']:.3f}% "
            f"[{100*lo:.3f}%, {100*hi:.3f}%] | {gate} |")
    lines+=['','节省为负表示更慢。三个独立守护进程，每配置每进程四次保留重复，'
        '没有按耗时删除样本。按进程和成对重复做 20000 次分层 bootstrap；只有三个进程簇，区间仅描述本轮重复性。'
        '门槛还要求三个进程同方向；零初始化或与 PCA 相同的排列不能归为学得加速。','',
        '## 映射、建造与摊销','',
        '| 模型 | 设备 | 6 万行一次映射 ms |', '|---|---|---:|']
    for row in s['mapping']:
        lines.append(f"| {row['family']} | {row['device']} | {row['mean_process_median_ms']:.3f} |")
    lines+=['','模型已加载；映射计时包含 FP64 特征转 FP32、GPU 路径的 H2D、前向与 D2H。'
        '每个进程预热后保留七次，表中是三个进程中位数的平均。CPU 使用四线程。'
        '6337 个总参数的网络每次完整前向仍计算全部层，约 0.745 GFLOP/6 万行；只搜索 64 个权重不自动降低部署前向算量。','',
        '| 布局 | 完整建造平均秒数（已训练模型） |', '|---|---:|']
    for family in FAMILIES:lines.append(f"| {family} | {s['build_seconds'][family]:.4f} |")
    lines+=['','建造实际重新执行 PCA、特征、模型映射、稳定排序、分块、保守界和原始输入重排。'
        '另外实测 54 次建造加完整查询，按布局/阈值/精度聚合见 cold_queries.csv，训练未包括在这些调用中。'
        'summary.json 分别保留选定检查点成本和全部三种子搜索成本的摊销估算；'
        '没有正查询节省或选中零初始化时记为 null。零模型无需投入生产，可直接保留 PCA。','',
        '## 正确性、资源与复现','',
        f"- 36 个布局/测试范围/阈值组合未剪掉参考命中；{s['admission_calls']} 次准入逐项匹配完整 FP64 参考。",
        f"- {s['warmups']} 次预热、{s['retained_calls']} 次保留计时、{s['build_plus_query_calls']} 次建造后查询，"
        f"加准入合计 {s['full_gpu_calls']} 次完整输出数量与 SHA256 一致。",
        f"- 五个独占守护过程均通过，GPU4 峰值设备占用 {s['max_device_used_mib']} MiB；未改时钟或终止其他任务。",
        '- 原始算术核及共同输出冻结身份保持；数值合同见 inherited/NUMERICAL_SCOPE.md，未主张无条件 exact-real。',
        '- PROTOCOL.md、training_freeze.json、model_selection.json、timing_freeze.json 记录冻结与选型。',
        '- results/training.json 含全部 1920 个提案、18 个检查点评价和成本；artifacts/*.pt 含权重。',
        '- artifacts/admission.json、results/confirm_*.json、raw/ 含几何与原始调用记录。',
        '- python analyze.py 从归档结果重建本报告、CSV/JSON 和 PNG/PDF 图；后处理不进入计时。',
        '- analysis/output_audit.json、final_environment.json、final_gpu_state.json 与 archive_manifest.json 保存最终核验。',
        '- 新目录复现：CPU src/prepare.py → GPU4 守护下 src/train.py → GPU4 守护下 src/admit.py → run_campaign.py；不要覆盖归档。','',
        '本轮限制于固定 PCA 判界坐标轴、标量排序和随机隐藏特征的坐标搜索。'
        '结果不能证明所有网络、全参数训练、其他排序结构或 FM 都无效。','',
        '![结果图](analysis/pruning_objective.png)','']
    (HERE/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
