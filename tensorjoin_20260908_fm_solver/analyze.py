"""Post-freeze analysis of solution-state FM and its ODE-resolution follow-up."""
import csv
import json
import math
import platform
import sys
import importlib.metadata
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
CELLS=['k4','k16','k64','original','k256','k1024']
METHODS=['F8','F16']
CONFIGS=['pca','all','heuristic25','random25','direct25','fm05','fm10','fm25']
BUILD_CONFIGS=['pca','all','heuristic25','direct25','fm25']


def read(rel):return json.loads((HERE/rel).read_text())
def ci(a):return np.quantile(a,[.025,.975]).tolist()
def csv_write(name,rows):
    with (HERE/'analysis'/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def break_even(extra,saving):return max(0,math.ceil(extra/saving)) if saving>0 else None


def main():
    (HERE/'analysis').mkdir(exist_ok=True)
    t=read('results/training.json');ad=read('artifacts/admission.json');prep=read('artifacts/preparation.json')
    runs=[read(f'results/confirm_{i:02d}.json') for i in range(3)]
    resolution=read('resolution_check/results.json')
    guards=[read(f'results/{label}_guard.json') for label in ['preflight_a2','training_a0','admission_a0','confirm_00_a0','confirm_01_a0','confirm_02_a0','resolution_a0']]
    assert t['pass_'] and ad['pass'] and resolution['pass'] and all(r['pass'] for r in runs+guards)
    samples=[s for r in runs for s in r['samples'] if s['phase']=='retained'];assert len(samples)==1152
    a=np.full((3,4,6,2,8),np.nan)
    for s in samples:
        idx=(s['process'],s['repeat'],CELLS.index(s['cell']),METHODS.index(s['method']),CONFIGS.index(s['configuration']))
        assert np.isnan(a[idx]);a[idx]=s['seconds']
    assert np.isfinite(a).all();proc=np.median(a,axis=1);point=proc.mean(axis=0)
    rng=np.random.default_rng(2026090929);pp=rng.integers(0,3,(20000,3));rr=rng.integers(0,4,(20000,3,4))
    boot=np.median(a[pp[:,:,None],rr],axis=2).mean(axis=1)
    comparisons=[];mixtures=[];table=[]
    pairs=[(config,'pca') for config in CONFIGS[1:]]+[('fm25','heuristic25'),('fm25','direct25')]
    for cj,cell in enumerate(CELLS):
        for mj,method in enumerate(METHODS):
            row=dict(cell=cell,method=method)
            row.update({config+'_ms':float(point[cj,mj,j]*1000) for j,config in enumerate(CONFIGS)});table.append(row)
            for config,baseline in pairs:
                j=CONFIGS.index(config);bj=CONFIGS.index(baseline)
                saving=float(1-point[cj,mj,j]/point[cj,mj,bj]);interval=ci(1-boot[:,cj,mj,j]/boot[:,cj,mj,bj])
                ps=(1-proc[:,cj,mj,j]/proc[:,cj,mj,bj]).tolist()
                comparisons.append(dict(cell=cell,method=method,configuration=config,baseline=baseline,
                    saving_fraction=saving,saving_ci95=interval,process_savings=ps,
                    passes_5pct_gate=bool(interval[0]>=.05 and min(ps)>0),primary=config=='fm25'))
    for mj,method in enumerate(METHODS):
        means=point[:,mj,:].mean(0);bb=boot[:,:,mj,:].mean(1)
        for config,baseline in pairs:
            j=CONFIGS.index(config);bj=CONFIGS.index(baseline)
            interval=ci(1-bb[:,j]/bb[:,bj]);ps=(1-proc[:,:,mj,j].mean(1)/proc[:,:,mj,bj].mean(1)).tolist()
            mixtures.append(dict(method=method,configuration=config,baseline=baseline,
                configuration_ms=float(means[j]*1000),baseline_ms=float(means[bj]*1000),
                saving_fraction=float(1-means[j]/means[bj]),saving_ci95=interval,process_savings=ps,
                passes_5pct_gate=bool(interval[0]>=.05 and min(ps)>0),primary=config=='fm25'))
    components=[]
    for method in METHODS:
        for config in CONFIGS:
            rows=[s for s in samples if s['method']==method and s['configuration']==config]
            total=np.mean([s['seconds'] for s in rows]);routing=np.mean([s['routing']['routing_seconds'] for s in rows])
            pred=np.mean([s['routing']['prediction_seconds'] for s in rows]);selection=np.mean([s['routing']['selection_seconds'] for s in rows])
            certificate=np.mean([s['routing']['certificate_seconds'] for s in rows])
            components.append(dict(method=method,configuration=config,total_mean_ms=float(total*1000),prediction_ms=float(pred*1000),
                selection_ms=float(selection*1000),certificate_ms=float(certificate*1000),
                other_routing_ms=float((routing-pred-selection-certificate)*1000),remaining_solver_ms=float((total-routing)*1000),
                note='Arithmetic mean of all retained samples, additive decomposition; different point estimator from the primary median table.'))
    mapping=[]
    for family,steps in [('direct',0),('fm',2),('fm',4),('fm',8)]:
        rows=[next(m for m in r['mapping'] if m['family']==family and m['solver_steps']==steps) for r in runs]
        assert len({m['prediction_sha256'] for m in rows})==1
        mapping.append(dict(family=family,solver_steps=steps,nfe=rows[0]['nfe'],selected=rows[0]['selected'],
            parameters=rows[0]['parameters'],rows=rows[0]['rows'],mean_process_median_ms=float(np.mean([m['median_seconds'] for m in rows])*1000),
            process_median_ms=[m['median_seconds']*1000 for m in rows],dense_layer_flops=rows[0]['dense_layer_flops']))
    cold_calls=[s for r in runs for s in r['single_queries']];assert len(cold_calls)==90
    builds={c:float(np.mean([s['build_wall_seconds'] for s in cold_calls if s['configuration']==c])) for c in BUILD_CONFIGS}
    cold=[]
    for config in BUILD_CONFIGS:
        for cell in ['k4','original','k1024']:
            for method in METHODS:
                rows=[s for s in cold_calls if s['configuration']==config and s['cell']==cell and s['method']==method]
                assert len(rows)==3
                cold.append(dict(configuration=config,cell=cell,method=method,build_plus_query_seconds=float(np.mean([s['seconds'] for s in rows])),
                    build_seconds=float(np.mean([s['build_wall_seconds'] for s in rows])),training_included=False))
    costs={}
    for family in ['direct','fm']:
        sel=t['selection']['models'][family];mods=[m for m in t['models'] if m['family']==family]
        costs[family]=dict(selected_training_seconds=sel['training_seconds']+sel['initialization_seconds'],
            all_seed_training_seconds=sum(m['training_seconds']+m['initialization_seconds'] for m in mods),
            all_seed_evaluation_seconds=sum(m['evaluation_seconds'] for m in mods))
    oracle=read('inherited/reference_manifest.json');oracle_seconds=oracle['ended']-oracle['started']
    for row in mixtures:
        if row['configuration']=='fm25' and row['baseline']=='pca':
            saving=(row['baseline_ms']-row['configuration_ms'])/1000
            common=prep['labels_aggregation_seconds']+t['validation_certificate_seconds']+builds['fm25']-builds['pca']
            row['selected_training_break_even_queries']=break_even(common+costs['fm']['selected_training_seconds'],saving)
            row['all_seed_break_even_queries']=break_even(common+costs['fm']['all_seed_training_seconds']+costs['fm']['all_seed_evaluation_seconds'],saving)
            row['including_historical_oracle_break_even_queries']=break_even(common+oracle_seconds+costs['fm']['all_seed_training_seconds']+costs['fm']['all_seed_evaluation_seconds'],saving)
        else:
            row['selected_training_break_even_queries']=row['all_seed_break_even_queries']=row['including_historical_oracle_break_even_queries']=None
    res_table=[];res_mix=[]
    for method in METHODS:
        mix={}
        for cell in CELLS:
            row=dict(cell=cell,method=method)
            for config in ['pca','heuristic25','direct25','fm25']:
                values=[s['seconds'] for s in resolution['samples'] if s['phase']=='retained' and s['configuration']==config and s['cell']==cell and s['method']==method]
                assert len(values)==3;row[config+'_ms']=float(np.median(values)*1000)
            res_table.append(row)
        for config in ['pca','heuristic25','direct25','fm25']:
            mix[config+'_ms']=float(np.mean([r[config+'_ms'] for r in res_table if r['method']==method]))
        res_mix.append(dict(method=method,solver_steps=resolution['selected_steps'],**mix,
            fm_saving_vs_pca=float(1-mix['fm25_ms']/mix['pca_ms']),scope='One supplementary process; descriptive only, no primary confidence claim.'))
    total_calls=len(ad['samples'])+len(ad['diagnostics'])+len(ad['build_admission'])+sum(len(r['samples'])+len(r['single_queries']) for r in runs)
    assert total_calls==1643 and len(resolution['samples'])==192
    summary=dict(primary_times=table,comparisons=comparisons,mixtures=mixtures,components=components,mapping=mapping,
        build_seconds=builds,cold_queries=cold,training_costs=costs,joint_training_wall_seconds=t['ended']-t['started'],
        label_aggregation_seconds=prep['labels_aggregation_seconds'],historical_oracle_generation_seconds=oracle_seconds,
        validation_certificate_seconds=t['validation_certificate_seconds'],geometry=ad['geometry'],
        raw_prediction_metrics=ad['raw_prediction_metrics'],selection=t['selection'],flow_diagnostics=ad['flow_diagnostics'],
        resolution=dict(selected_steps=resolution['selected_steps'],convergence_gate_passed=resolution['convergence_gate_passed'],
            diagnostics=resolution['diagnostics'],raw_prediction_metrics=resolution['raw_prediction_metrics'],times=res_table,mixtures=res_mix),
        main_complete_calls=1643,supplementary_complete_calls=192,total_complete_calls=1835,
        main_admission_array_calls=113,supplementary_array_calls=48,main_retained_calls=1152,main_warmups=288,cold_calls=90,
        max_device_used_mib=max(g['max_device_used_mib'] for g in guards),
        bootstrap=dict(draws=20000,seed=2026090929,scope='Three-process paired hierarchical bootstrap for main experiment only.'),
        limitations=['Scalar block occupancy, not direct generation of all pair IDs.','Lossy input summaries.','Adaptive same-dataset labels and transductive PCA.',
            'Primary selected two-step solver is underresolved; fixed-weight supplementary convergence check reported separately.',
            'Exact output relies on conservative completion, not neural confidence.'])
    (HERE/'analysis/summary.json').write_text(json.dumps(summary,indent=2))
    for name,rows in [('times',table),('comparisons',comparisons),('mixtures',mixtures),('components',components),('mapping',mapping),('cold_queries',cold),
                      ('raw_prediction_metrics',ad['raw_prediction_metrics']),('resolution_times',res_table),('resolution_mixtures',res_mix)]:csv_write(name+'.csv',rows)
    keys=['process','repeat','cell','method','configuration','seconds','executed_tiles','output_count','output_sha256','tile_mask_sha256']
    csv_write('raw_times.csv',[{k:s[k] for k in keys} for s in samples])
    csv_write('candidate_evaluations.csv',[dict(family=c['family'],seed=c['seed'],updates=c['step'],solver_steps=c['solver_steps'],
        extra_pruning=c['validation']['mean_extra_pair_pruning'],endpoint_mse=c['validation']['mean_endpoint_mse'],
        selected=c==t['selection']['models'][c['family']],checkpoint_sha256=c['checkpoint_sha256']) for c in t['candidates']])
    (HERE/'analysis/statistics_environment.json').write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),
        packages={n:importlib.metadata.version(n) for n in ['numpy','matplotlib']}),indent=2))
    report(summary,t,prep);plot(summary,t)
    print(json.dumps({k:summary[k] for k in ['mixtures','mapping','build_seconds','training_costs','total_complete_calls']},indent=2))


def report(s,t,prep):
    primary=[r for r in s['mixtures'] if r['configuration']=='fm25' and r['baseline']=='pca']
    lead='主实验 FM25 '+('有路径达到' if any(r['passes_5pct_gate'] for r in primary) else '未达到')+'相对 PCA 的预定收益门槛。'
    lead+='六阈值平均耗时变化：'+ '，'.join(f"{r['method']} {-100*r['saving_fraction']:+.2f}%" for r in primary)+'。'
    lines=['# 从数据初始状态到连接解的条件 FM 验证','',lead,'',
        '目录：`gpu-host-8:@TENSORJOIN_ROOT@/tensorjoin_20260908_fm_solver`。所有性能比较使用本轮 GPU4；此前实验只读。','',
        '本轮实际学习“数据构造的粗糙状态→精确参考的块占用标签”，没有高斯目标或随机高斯起点。'
        'FM 只预测一个 64×64 点对块是否有命中；最终点对集合由可靠检查及原有算术核生成。'
        '这验证的是面向解的候选生成与补全，不是直接生成完整离散连接矩阵。','',
        '## 两个必须区分的结果','',
        '未经验证的 FM 判空仍可能漏掉真实输出，因此不能把模型终点直接称为精确解。'
        '所有安全执行只允许通过保守距离证书的块被丢弃；模型判空但证书不通过的块全部继续计算。','',
        '| 阈值 | 主实验 FM 两步直接判空会漏掉的输出条目 | 加密积分后会漏掉的输出条目 |',
        '|---|---:|---:|']
    for cell in CELLS:
        a=next(r for r in s['raw_prediction_metrics'] if r['split']=='full' and r['family']=='fm' and r['cell']==cell)
        b=next(r for r in s['resolution']['raw_prediction_metrics'] if r['cell']==cell)
        lines.append(f"| {cell} | {a['missed_reference_entries']:,} | {b['missed_reference_entries']:,} |")
    lines+=['','条目按原输出格式计数，同一个非自对通常包含两个方向。这里是假设直接删除模型判空块的诊断，'
        '正式执行从未这样剪枝。完整误报/漏报、候选召回率和验证/测试结果见 raw_prediction_metrics.csv。','',
        f"安全版本合计 {s['total_complete_calls']} 次完整调用均匹配参考输出数量与 SHA256；其中 161 次逐项比较完整数组。"
        '全量所有主配置/阈值的 48 个掩码，以及补充积分配置的掩码，均未剪掉参考命中。','',
        '## 实际模型、监督和选型','',
        '条件为 20 个数据几何统计加查询阈值。初始状态 a=tanh((T−粗距离估计)/(0.25T))，'
        '监督终点 y 为有命中 +1、无命中 −1。采样 t，构造 z_t=(1−t)a+ty，回归速度 v(c,z_t,t)≈y−a。'
        '推理实际积分 dz/dt=v(c,z,t)。这是条件监督插值；不宣称有限步 ODE 精确输运到离散分布。'
        '非高斯来源与配对条件路径的依据见 [CFM](https://arxiv.org/html/2302.00482v4)，具体状态与证书设计是本实验选择。','',
        'FM 为条件仿射层 21→64、状态/时间权重、SiLU、64→64、SiLU、标量输出，共 5761 参数。'
        '直接残差网络删除时间输入，共 5697 参数。隐藏层/状态初始权重匹配、输出层为零；全部参数参与训练。'
        '三个种子，每类每种子 3000 步、batch 1024，相同训练样本、Adam lr 0.001。'
        '只缓存每次推理中与状态和时间无关的首层条件仿射项，其他速度场求值全部实际执行。','',
        '| 模型 | 选定种子 / 更新数 | ODE 步数 / NFE | 验证集额外可靠剪枝 |',
        '|---|---|---|---:|']
    for f in ['direct','fm']:
        m=t['selection']['models'][f];lines.append(f"| {f} | {m['seed']} / {m['step']} | {m['solver_steps']} / {m['nfe']} | {100*m['validation']['mean_extra_pair_pruning']:.3f}% |")
    lines+=['','选择依据是：在相同 25% 额外检查预算下，六个验证阈值平均增加的可靠剪枝量。'
        'FM 尝试 2/4/8 步，共 27 候选；直接网络九候选。全部 18 权重与 36 次评估保留。'
        '没有用测试标签、全量证书结果或查询耗时选择主模型。两类回归预测目标不同，固定回归诊断的抽样也不同，'
        '不能横向比较其损失数值；实际选型使用同一完整验证块集合。','',
        '训练/验证/测试按整块分开，行数为 40000/9984/10016，只使用两个端点块均属于相应划分的点对。'
        'PCA 使用全体无标签数据，同一数据集已在先前报告中出现；这是适应性后续实验，不是新数据泛化证据。'
        '监督标签从已有精确参考中提取，因此不能把标签当作首次求解时无代价获得的答案。','',
        '## 新增可靠检查和各项对照','',
        '先应用原 PCA32 包围盒；模型或规则从剩余非对角块中选择固定比例进行更细检查。'
        '新增 GPU 核计算块中实际 4096 对的投影距离下界，每八维检查一次，可在 8/16/24 维提前结束，最多 32 维。'
        '所有投影、FP32 舍入和阈值误差都计入保守余量；当前全局绝对余量为 0.000253584。'
        '细节见 CERTIFICATE_SCOPE.md。未经此检查证明为空的块继续走原求解器。','',
        '| 策略 | 剩余块的额外检查预算 | 全量总可靠剪枝率 | 检查成功率 |',
        '|---|---:|---:|---:|']
    for config in CONFIGS:
        rows=[r for r in s['geometry'] if r['configuration']==config]
        success=np.mean([r['extra_certified_tiles']/max(r['certificate_tiles'],1) for r in rows])
        lines.append(f"| {config} | {100*rows[0]['budget']:.0f}% | {100*np.mean([r['total_pair_pruning'] for r in rows]):.3f}% | {100*success:.2f}% |")
    lines+=['','pca 为原下界；all 为全部新增检查；heuristic25 按数据粗估计排序；random25 随机选择；'
        'direct25 与 fm25 使用相同预算。fm05/fm10 检查 FM 在较低预算下能否抵消推理成本。'
        '总剪枝率按无序点对数量加权后对六阈值等权平均。all 体现当前证书可达到的范围，并不读取理想空块标签进行路由。','',
        '## 主实验完整查询时间','',
        '| 阈值 | 路径 | PCA ms | 全部检查 ms | 启发式25 ms | 直接网络25 ms | FM25 ms |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in s['primary_times']:lines.append(f"| {r['cell']} | {r['method']} | {r['pca_ms']:.3f} | {r['all_ms']:.3f} | {r['heuristic25_ms']:.3f} | {r['direct25_ms']:.3f} | {r['fm25_ms']:.3f} |")
    lines+=['','计时包含每次重新构造模型条件、真实 ODE、排序选择、实际证书检查、GPU 输入/metadata、'
        '原始算术与精化、ID 恢复、CUB 完整结果排序和 CPU 下载。复用的是查询无关的 CPU 索引与摘要，'
        '没有复用模型预测或证书结果。模型权重已加载；训练与索引建造另计。','',
        '| 路径 | FM25 相对基线 | 六阈值平均耗时节省 [95% CI] | 5% 门槛 |',
        '|---|---|---:|---|']
    for r in s['mixtures']:
        if r['configuration']!='fm25':continue
        lo,hi=r['saving_ci95'];lines.append(f"| {r['method']} | {r['baseline']} | {100*r['saving_fraction']:.2f}% [{100*lo:.2f}%, {100*hi:.2f}%] | "+('通过' if r['passes_5pct_gate'] else '未通过')+' |')
    lines+=['','负节省表示更慢。主点估计为三个进程各自四次重复中位数的均值；20000 次成对分层 bootstrap。'
        '只有三个进程簇，置信区间限于本轮重复性。主门槛为节省下界至少 5%，且三个进程方向均正。'
        '其余预算、随机对照和每格比较完整保存在 times.csv、comparisons.csv、mixtures.csv；未删除慢样本。','',
        '## ODE 精度补充验证','',
        f"主实验选中了两步 Heun，但其 512 个诊断终点相对 16 步差异达 {100*s['flow_diagnostics']['solvers'][0]['relative_rms_vs_16']:.2f}%。"
        '因此，主实验的两步推理不能称为已收敛的连续流求解。保留主实验冻结结果，另开固定权重的积分加密检查。','',
        '| Heun 步数 | 验证样本终点相对 RMS 差（对 512 步） | 符号判断分歧 |',
        '|---|---:|---:|']
    for r in s['resolution']['diagnostics']:lines.append(f"| {r['steps']} | {100*r['relative_rms_vs_512']:.4f}% | {100*r['sign_disagreement']:.4f}% |")
    lines+=['',f"补充检查在 512 个验证块输入上按预先写定的 1% RMS/1% 判断分歧门槛选定 {s['resolution']['selected_steps']} 步；"
        f"数值门槛{'通过' if s['resolution']['convergence_gate_passed'] else '未通过，使用最细参考步长并保留限制'}。"
        '512 步仍只是数值参考。权重不变，不看测试标签或查询耗时选步长；此补充本身是发现误差后的适应性实验。','',
        '| 路径 | 同进程 PCA ms | 启发式25 ms | 直接网络25 ms | 加密积分 FM25 ms |',
        '|---|---:|---:|---:|---:|']
    for r in s['resolution']['mixtures']:lines.append(f"| {r['method']} | {r['pca_ms']:.3f} | {r['heuristic25_ms']:.3f} | {r['direct25_ms']:.3f} | {r['fm25_ms']:.3f} |")
    lines+=['','补充验证是一个新进程、每格一次预热和三次保留重复，共 192 次完整调用。'
        '这些同进程比值只作描述，不沿用主实验三进程置信区间，也不把不同阶段耗时直接相除。','',
        '## 推理、训练和建造成本','',
        '| 模型 | 步数 / NFE | 原阈值剩余块数量 | 推理平均进程中位数 ms |',
        '|---|---|---:|---:|']
    for r in s['mapping']:lines.append(f"| {r['family']} | {r['solver_steps']} / {r['nfe']} | {r['rows']:,} | {r['mean_process_median_ms']:.3f} |")
    lines+=['','独立推理计时从已准备的主机摘要/初始状态到主机终点，含归一化、H2D、实际各步和 D2H；'
        '正式查询还计入摘要选择、初态构造等工作。每进程预热后七次保留重复，完整分项见 components.csv。','',
        '| 模型 | 选定检查点训练＋初始化秒数 | 三种子训练＋初始化秒数 | 检查点评估秒数 |',
        '|---|---:|---:|---:|']
    for family,c in s['training_costs'].items():lines.append(f"| {family} | {c['selected_training_seconds']:.3f} | {c['all_seed_training_seconds']:.3f} | {c['all_seed_evaluation_seconds']:.3f} |")
    lines+=['',f"本轮联合训练阶段墙钟 {s['joint_training_wall_seconds']:.3f} 秒，标签提取 {s['label_aggregation_seconds']:.3f} 秒，"
        f"验证证书准备 {s['validation_certificate_seconds']:.3f} 秒。已有全量 FP64 参考生成记录为 {s['historical_oracle_generation_seconds']:.3f} 秒，"
        '该值来自历史记录，不是本轮 GPU4 重测，也未包含在本轮标签提取时间中。'
        '选定检查点训练账单是已知选择后的重训下限；独立模型负担完整共享批准备成本。','',
        '| 策略 | 新建所需索引平均秒数 |',
        '|---|---:|']
    for config,value in s['build_seconds'].items():lines.append(f"| {config} | {value:.4f} |")
    lines+=['','索引构造包含 PCA、原界、重排，以及相应策略需要的数据摘要。主实验另有 90 次实际建造后查询，'
        '完整数据见 cold_queries.csv。训练成本、标签成本与额外建造成本除以正查询节省得到的摊销估计保存在 mixtures.csv；'
        '查询本身无正节省时记为 null。相同查询已有完整答案时还可直接缓存答案，不能把此实验的监督标签视为首次求解的免费资源。','',
        '## 结论范围和复核','',
        '本轮把用户提出的思路实现为有监督的块候选状态修正。原始预测、可靠补全以及 ODE 数值精度分别检查，'
        '没有把高候选召回率或低训练损失当作精确解证明。它仍受摘要信息量、标量块状态、网络与训练预算限制；'
        '没有测试完整点对状态、更强条件编码、新数据泛化或其他更便宜的精确补全方式。','',
        '- 训练、选型、证书和计时依赖均保存冻结哈希；原算术/输出实现身份保持一致。',
        '- 两个训练前开发失败及对应源码保留：提前结束分支的 Triton break 语法不支持；预检未列出有效的 24 维提前结束。修正后完整预检通过，再冻结训练。',
        '- 初始裸模块导入检查的 admit/runner 名字解析到了继承模块；实际工作进程始终按新文件绝对路径执行。analysis/import_paths.json 另核验了实际入口及新 engine/train 函数身份。',
        f"- 七个成功守护阶段最高 GPU4 占用 {s['max_device_used_mib']} MiB；两个开发失败另行保留，未终止其他任务或修改时钟。",
        '- analysis/output_audit.json、最终环境/GPU 状态、archive_manifest.json 记录归档核验；manifest 仅排除缓存目录和自身。',
        '- 主复现入口：CPU prepare.py → GPU 守护下 preflight.py → train.py → admit.py → run_campaign.py；随后按 resolution_check/PROTOCOL.md 执行积分补充。禁止覆盖旧结果。',
        '- python analyze.py 重建统计、报告与独立 PNG/PDF。正式数值范围见 CERTIFICATE_SCOPE.md 和 inherited/NUMERICAL_SCOPE.md，不主张无条件 exact-real。','',
        '![条件 FM 验证](analysis/fm_solver.png)','']
    (HERE/'REPORT.md').write_text('\n'.join(lines))


def plot(s,t):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
    for family,color in [('direct','#e69f00'),('fm','#0072b2')]:
        histories=[m['history'] for m in t['models'] if m['family']==family]
        x=[r['step'] for r in histories[0]];y=np.array([[r['loss'] for r in h] for h in histories])
        axes[0,0].plot(x,y.mean(0),label=family,color=color);axes[0,0].fill_between(x,y.min(0),y.max(0),color=color,alpha=.2)
    axes[0,0].set(title='Training losses (different regression targets)',xlabel='Updates',ylabel='MSE');axes[0,0].legend()
    x=np.arange(6)
    for offset,label,rows,color in [(-.18,'FM 2 steps',[r for r in s['raw_prediction_metrics'] if r['split']=='full' and r['family']=='fm'],'#56b4e9'),
                                   (.18,f"FM {s['resolution']['selected_steps']} steps",s['resolution']['raw_prediction_metrics'],'#e69f00')]:
        yy=[next(r for r in rows if r['cell']==c)['missed_reference_entries'] for c in CELLS]
        axes[0,1].bar(x+offset,yy,.34,label=label,color=color)
    axes[0,1].set_xticks(x,CELLS);axes[0,1].set_yscale('symlog',linthresh=1)
    axes[0,1].set(title='Raw FM rejection would lose true outputs',ylabel='Missed ordered reference entries');axes[0,1].legend()
    configs=['pca','heuristic25','direct25','fm25','all']
    vals=[100*np.mean([r['total_pair_pruning'] for r in s['geometry'] if r['configuration']==c]) for c in configs]
    bars=axes[0,2].bar(configs,vals,color=['#777777','#009e73','#e69f00','#0072b2','#999999'])
    axes[0,2].bar_label(bars,fmt='%.2f',padding=3);axes[0,2].set(title='Certified pruning, equal six-threshold average',ylabel='Pruned unordered pairs (%)')
    for j,method in enumerate(METHODS):
        rows=[r for r in s['comparisons'] if r['configuration']=='fm25' and r['baseline']=='pca' and r['method']==method]
        xx=x+(j-.5)*.1;yy=100*np.array([r['saving_fraction'] for r in rows]);cc=100*np.array([r['saving_ci95'] for r in rows]);color=['#0072b2','#e69f00'][j]
        axes[1,0].vlines(xx,cc[:,0],cc[:,1],color=color);axes[1,0].plot(xx,yy,'o-',color=color,label=method)
    axes[1,0].axhline(0,color='black',lw=.8);axes[1,0].set_xticks(x,CELLS)
    axes[1,0].set(title='Main complete queries: FM25 versus PCA',ylabel='Time saved (%)');axes[1,0].legend()
    rows=[next(r for r in s['components'] if r['configuration']==c and r['method']=='F8') for c in configs]
    bottom=np.zeros(len(rows))
    for key,label,color in [('remaining_solver_ms','Remaining solver','#999999'),('other_routing_ms','Routing setup','#cc79a7'),
        ('selection_ms','Selection','#009e73'),('certificate_ms','Certificate','#e69f00'),('prediction_ms','Model / ODE','#0072b2')]:
        v=np.asarray([r[key] for r in rows]);axes[1,1].bar(configs,v,bottom=bottom,label=label,color=color);bottom+=v
    axes[1,1].set(title='F8 cost components (all-sample means)',ylabel='Milliseconds');axes[1,1].legend(fontsize=8)
    rows=s['resolution']['diagnostics'];xx=[r['steps'] for r in rows]
    axes[1,2].semilogx(xx,[100*r['relative_rms_vs_512'] for r in rows],'o-',label='Endpoint relative RMS')
    axes[1,2].semilogx(xx,[100*r['sign_disagreement'] for r in rows],'s-',label='Sign disagreement')
    axes[1,2].axhline(1,color='black',ls=':',label='1% numerical gate');axes[1,2].set_xticks([2,8,32,128,512],['2','8','32','128','512'])
    axes[1,2].set(title='Fixed weights: ODE resolution on validation',xlabel='Heun steps',ylabel='Difference versus 512 steps (%)');axes[1,2].legend(fontsize=8)
    for ax in axes.ravel():ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(HERE/'analysis/fm_solver.png',dpi=180);fig.savefig(HERE/'analysis/fm_solver.pdf');plt.close(fig)

if __name__=='__main__':main()
