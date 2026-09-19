"""Reproduce all statistics and standalone plots from archived JSON."""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CELLS = ['k4','k16','k64','original','k256','k1024']
METHODS = ['F8','F16']
MODES = ['original_full','layout_full','geometric']
LAYOUTS = ['original','pca_sort','balanced_axis','balanced_2means']

def read(p):
    return json.loads((HERE/p).read_text())

def main():
    out = HERE/'analysis'
    out.mkdir(exist_ok=True)
    screen = read('results/screen_all.json')
    admission = read('artifacts/admission.json')
    environment = read('artifacts/environment.json')
    assert screen['pass_'] and admission['pass'] and environment['pass_']
    screen_freeze = read('artifacts/screen_freeze.json')['files']
    for rel,expected in screen_freeze.items():
        assert hashlib.sha256((HERE/rel).read_bytes()).hexdigest()==expected,rel
    runs = [read(f'results/confirm_{i:02d}.json') for i in range(3)]
    guards = [read('results/admission_a0_guard.json')]+[
        read(f'results/confirm_{i:02d}_a0_guard.json') for i in range(3)]
    assert all(r['pass'] for r in runs+guards)
    samples = [s for r in runs for s in r['samples'] if s['phase']=='retained']
    assert len(samples) == 432
    a = np.full((3,4,6,2,3),np.nan)
    for s in samples:
        ix = (s['process'],s['repeat'],CELLS.index(s['cell']),METHODS.index(s['method']),MODES.index(s['mode']))
        assert np.isnan(a[ix])
        a[ix] = s['seconds']
    assert np.isfinite(a).all()
    # Resample process clusters, then paired repetition blocks within each
    # selected process; keep the same resampling across methods/modes/cells.
    rng = np.random.default_rng(2026090811)
    pi = rng.integers(0,3,(20000,3))
    ri = rng.integers(0,4,(20000,3,4))
    boot = np.median(a[pi[:,:,None],ri],axis=2).mean(axis=1)
    process = np.median(a,axis=1)
    point = process.mean(axis=0)
    primary = [r for r in screen['records'] if r['dataset']=='cifar60000']
    selected = screen['selected']['selected']
    chosen = next(r for r in primary if r['layout']==selected)
    geom = {c['cell']:c for c in chosen['cells'] if c['bound']=='combined'}
    cold = [s for r in runs for s in r['single_query']]
    ideals = [s for r in runs for s in r['ideal']]
    btime = float(np.mean([s['build_wall_seconds'] for s in cold]))
    table = []
    def interval(values):
        return np.quantile(values,[.025,.975]).tolist()
    for ci,cell in enumerate(CELLS):
        for mi,method in enumerate(METHODS):
            base,layout,pruned = point[ci,mi]
            ratio = base/pruned
            cis = interval(boot[:,ci,mi,0]/boot[:,ci,mi,2])
            per_process = (process[:,ci,mi,0]/process[:,ci,mi,2]).tolist()
            rows = [s for s in cold if s['cell']==cell and s['method']==method]
            ideal_rows = [s for s in ideals if s['cell']==cell and s['method']==method]
            saving = base-pruned
            row = dict(cell=cell,method=method,original_ms=base*1000,layout_full_ms=layout*1000,
                geometric_ms=pruned*1000,speedup=ratio,speedup_ci95=cis,
                saving_fraction=1-pruned/base,process_speedups=per_process,
                passes_cell_5pct_gate=bool(cis[0] >= 1/.95 and min(per_process)>1),
                prune_vs_layout_speedup=layout/pruned,
                prune_vs_layout_ci95=interval(boot[:,ci,mi,1]/boot[:,ci,mi,2]),
                pruned_tile_fraction=geom[cell]['pruned_tile_fraction'],
                pruned_pair_fraction=geom[cell]['pruned_pair_fraction'],
                ideal_empty_tile_fraction=geom[cell]['ideal_empty_tile_fraction'],
                build_seconds_mean=btime,
                modeled_break_even_queries=math.ceil(btime/saving) if saving>0 else None,
                single_query_seconds_mean=float(np.mean([s['seconds'] for s in rows])) if rows else None,
                ideal_diagnostic_ms=float(np.mean([s['seconds'] for s in ideal_rows]))*1000 if ideal_rows else None)
            table.append(row)
    mixtures = []
    for mi,method in enumerate(METHODS):
        base,layout,pruned = point[:,mi,:].mean(axis=0)
        bb = boot[:,:,mi,:].mean(axis=1)
        ci = interval(1-bb[:,2]/bb[:,0])
        proc_saving = 1-process[:,:,mi,2].mean(axis=1)/process[:,:,mi,0].mean(axis=1)
        saving = base-pruned
        mixtures.append(dict(method=method,original_ms=base*1000,layout_full_ms=layout*1000,
            geometric_ms=pruned*1000,saving_fraction=1-pruned/base,saving_ci95=ci,
            process_saving_fraction=proc_saving.tolist(),
            passes_mixture_5pct_gate=bool(ci[0]>=.05 and np.min(proc_saving)>0),
            modeled_break_even_queries=math.ceil(btime/saving) if saving>0 else None))
    results = dict(selected=selected,primary=table,equal_six_query_mixtures=mixtures,
        build_seconds_mean=btime,build_seconds_range=[min(s['build_wall_seconds'] for s in cold),
          max(s['build_wall_seconds'] for s in cold)],retained_measurements=len(samples),
        warmups=sum(s['phase']=='warmup' for r in runs for s in r['samples']),
        single_queries=len(cold),ideal_diagnostic_queries=len(ideals),
        admission_calls=len(admission['samples'])+len(admission['diagnostics'])+1,
        gpu_complete_calls=len(admission['samples'])+len(admission['diagnostics'])+1+
            sum(len(r['samples'])+len(r['single_query'])+len(r['ideal']) for r in runs),
        geometry_cells=sum(len(r['cells']) for r in screen['records']),
        max_device_used_mib=max(g['max_device_used_mib'] for g in guards),
        bootstrap=dict(draws=20000,seed=2026090811,unit='process + paired repetition blocks'),
        scope='Three process clusters; uncertainty is limited to this host/GPU/input/run.')
    (out/'summary.json').write_text(json.dumps(results,indent=2))
    with (out/'summary.csv').open('w') as f:
        writer = csv.DictWriter(f,fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    with (out/'raw_times.csv').open('w') as f:
        keys = ['process','repeat','cell','method','mode','seconds','executed_tiles','batches','output_sha256']
        writer = csv.DictWriter(f,fieldnames=keys,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(samples)
    geometry_rows = [c for r in screen['records'] for c in r['cells']]
    with (out/'geometry.csv').open('w') as f:
        writer = csv.DictWriter(f,fieldnames=list(geometry_rows[0]))
        writer.writeheader()
        writer.writerows(geometry_rows)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    xx = np.arange(6)
    colors = ['#666666','#1769aa','#e69f00','#009e73']
    for layout,color in zip(LAYOUTS,colors):
        rec = next(r for r in primary if r['layout']==layout)
        cells = {c['cell']:c for c in rec['cells'] if c['bound']=='combined'}
        axes[0,0].plot(xx,[100*cells[c]['pruned_tile_fraction'] for c in CELLS],'o-',label=layout,color=color)
        axes[0,1].plot(xx,[100*cells[c]['ideal_empty_tile_fraction'] for c in CELLS],'o-',label=layout,color=color)
    axes[0,0].set_title('Certified geometric tile pruning: Cifar60K')
    axes[0,1].set_title('Ideal empty tiles: diagnostic only')
    for ax in axes[0]:
        ax.set_xticks(xx,CELLS)
        ax.set_ylabel('Tiles (%)')
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=9)
    for mi,method in enumerate(METHODS):
        rows = [r for r in table if r['method']==method]
        y = np.asarray([r['speedup'] for r in rows])
        cis = np.asarray([r['speedup_ci95'] for r in rows])
        # Bootstrap quantiles need not bracket the plug-in estimate exactly.
        axes[1,0].vlines(xx+(mi-.5)*.12,cis[:,0],cis[:,1],color=colors[mi+1])
        axes[1,0].plot(xx+(mi-.5)*.12,y,'o-',label=method,color=colors[mi+1])
    axes[1,0].axhline(1,color='black',lw=.8)
    axes[1,0].axhline(1/.95,color='grey',ls='--',lw=.8)
    axes[1,0].set_xticks(xx,CELLS)
    axes[1,0].set_title('Complete repeated-query speedup (build excluded)')
    axes[1,0].set_ylabel('Original full / geometric')
    axes[1,0].legend()
    axes[1,0].grid(alpha=.2)
    q = np.geomspace(1,1e5,300)
    for mi,method in enumerate(METHODS):
        for ci,cell in [(0,'k4'),(3,'original'),(5,'k1024')]:
            base,_,pruned = point[ci,mi]
            axes[1,1].plot(q,base/(pruned+btime/q),label=f'{method} {cell}')
    axes[1,1].axhline(1,color='black',lw=.8)
    axes[1,1].set_xscale('log')
    axes[1,1].set_ylim(0,1.4)
    axes[1,1].set_title('Modeled build amortization: B/Q + query')
    axes[1,1].set_xlabel('Queries reusing layout and bound metadata')
    axes[1,1].set_ylabel('Speedup over original full')
    axes[1,1].legend(fontsize=8,ncol=2)
    axes[1,1].grid(alpha=.2)
    fig.savefig(out/'block_feasibility.png',dpi=180)
    fig.savefig(out/'block_feasibility.pdf')
    plt.close(fig)
    write_report(results,screen,admission,runs)
    print(json.dumps(results,indent=2))

def write_report(s,screen,admission,runs):
    rows = s['primary']
    lines = ['# 分块剪枝可行性实验', '',
      '主机：`gpu-host-8`；GPU0 RTX PRO 6000 Blackwell Server Edition。',
      '远程目录：`@TENSORJOIN_ROOT@/tensorjoin_20260908_block_feasibility`。', '',
      '本轮实现了简单布局和保守几何剪枝，没有训练 flow matching 模型。'
      '所有性能比较均在本轮 GPU0 内部完成；没有与旧 GPU2 时间相除。', '',
      '## 主要结果', '',
      '冻结规则选择了 `pca_sort`：按第一主成分排序，并以投影包围盒等四类界的最大值剪枝。'
      '在 Cifar60K 上全部实际剪枝来自 32 维 PCA 投影界；球界、原坐标包围盒、16 个枢轴界没有额外贡献。', '',
      '| 阈值 | 实际剪掉 tile | 该布局实际空 tile（理想诊断） | F8 查询加速 [95% CI] | F16 查询加速 [95% CI] |',
      '|---|---:|---:|---:|---:|']
    for cell in CELLS:
        r8,r16 = [next(r for r in rows if r['cell']==cell and r['method']==m) for m in METHODS]
        def ratio(r): return f"{r['speedup']:.3f} [{r['speedup_ci95'][0]:.3f}, {r['speedup_ci95'][1]:.3f}]"
        lines.append(f"| {cell} | {r8['pruned_tile_fraction']*100:.2f}% | {r8['ideal_empty_tile_fraction']*100:.2f}% | {ratio(r8)} | {ratio(r16)} |")
    lines += ['', '查询加速包括阈值掩码、tile 列表、输入上传、GPU metadata、扫描与精化、'
      '原始 ID 恢复、CUB 完整排序及 CPU 结果下载；复用 CPU 布局和几何 metadata，未计入建造成本。'
      '表中加速为原始全扫描 / 几何剪枝；仅重排对照保存在下表与 CSV。', '',
      '六个阈值等查询权重时：', '']
    for m in s['equal_six_query_mixtures']:
        lines.append(f"- {m['method']}：平均 {m['original_ms']:.3f} → {m['geometric_ms']:.3f} ms，"
          f"节省 {100*m['saving_fraction']:.3f}% [95% CI {100*m['saving_ci95'][0]:.3f}%, {100*m['saving_ci95'][1]:.3f}%]；"
          f"5% 混合门槛：{'通过' if m['passes_mixture_5pct_gate'] else '未通过'}。")
    lines += ['', '## 预处理和单次查询', '',
      f"18 次包含重新建造的完整单次查询中，建造平均 {s['build_seconds_mean']:.3f} 秒"
      f"（范围 {s['build_seconds_range'][0]:.3f}–{s['build_seconds_range'][1]:.3f} 秒）。"
      '本实现为可审计的 CPU 通用筛选器，构造了全部四类界及块对下界矩阵；没有删除本轮无贡献的界。', '',
      '| 工作点 | 路径 | 原始全扫描 ms | 仅重排 ms | 剪枝查询 ms | 实测完整单次查询 s | 建造摊销盈亏平衡查询数* |',
      '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        single = '—' if r['single_query_seconds_mean'] is None else f"{r['single_query_seconds_mean']:.3f}"
        q = str(r['modeled_break_even_queries']) if r['modeled_break_even_queries'] is not None else '无'
        lines.append(f"| {r['cell']} | {r['method']} | {r['original_ms']:.3f} | {r['layout_full_ms']:.3f} | "
                     f"{r['geometric_ms']:.3f} | {single} | {q} |")
    lines += ['', '*盈亏平衡由 `B / (T原始 − T剪枝)` 估算、向上取整；B 为全部完整单次调用的平均建造时间。'
      '这是静态数据复用布局/metadata 的模型，不是这些查询次数的实际批量测量。'
      '不缓存结果；不含初次 JIT、磁盘读输入或核验哈希。差值很小时估算不稳定。', '',
      '## 布局机会与判界能力', '',
      '| 布局 | k4 实际剪枝 | k4 理想空 tile | 原阈值实际剪枝 | 原阈值理想空 tile |',
      '|---|---:|---:|---:|---:|']
    for layout in LAYOUTS:
        rr = next(r for r in screen['records'] if r['dataset']=='cifar60000' and r['layout']==layout)
        cs = {c['cell']:c for c in rr['cells'] if c['bound']=='combined'}
        lines.append(f"| {layout} | {100*cs['k4']['pruned_tile_fraction']:.2f}% | {100*cs['k4']['ideal_empty_tile_fraction']:.2f}% | "
          f"{100*cs['original']['pruned_tile_fraction']:.2f}% | {100*cs['original']['ideal_empty_tile_fraction']:.2f}% |")
    lines += ['', '平衡二均值分组产生了更多空块，但本轮几何界识别率更低。'
      '因此“分组更像聚类”不足以推断剪枝更有效；需要同时考虑界的紧度与 GPU 执行成本。', '',
      '理想占用掩码由完整参考结果反推。以下只是在事先免费知道该掩码时的乐观执行诊断，'
      '未包含获取答案的成本，不是可部署算法，也不是对其他布局/机制的严格上界。', '',
      '| 工作点 | 路径 | 原始全扫描 ms | 几何剪枝 ms | 理想掩码执行 ms |',
      '|---|---|---:|---:|---:|']
    for r in rows:
        if r['ideal_diagnostic_ms'] is not None:
            lines.append(f"| {r['cell']} | {r['method']} | {r['original_ms']:.3f} | {r['geometric_ms']:.3f} | {r['ideal_diagnostic_ms']:.3f} |")
    lines += ['', '## 其他分布和正对照', '',
      'SIFT128 与 Fashion784 使用此前冻结的 4096 行样本及 k1/k16/k64 阈值。'
      '它们仅做 CPU 几何与完整参考占用核验，没有测 GPU 端到端加速，不能与主实验的速度混合。', '',
      '| 数据 | 布局 | k1 剪枝 | k16 剪枝 | k64 剪枝 |',
      '|---|---|---:|---:|---:|']
    for name in ['sift128','fashion784']:
        for layout in LAYOUTS:
            rr = next(r for r in screen['records'] if r['dataset']==name and r['layout']==layout)
            cs = {c['cell']:c for c in rr['cells'] if c['bound']=='combined'}
            lines.append(f"| {name} | {layout} | "+' | '.join(f"{100*cs[c]['pruned_tile_fraction']:.2f}%" for c in ['k1','k16','k64'])+' |')
    pos = next(r for r in screen['records'] if r['dataset']=='positive_control' and r['layout']=='balanced_2means')
    pc = next(c for c in pos['cells'] if c['bound']=='combined')
    lines += ['', f"打乱顺序的明显分离簇正对照，经平衡二均值分组剪掉 {100*pc['pruned_tile_fraction']:.2f}% tile，"
      '且与全部参考占用一致；此结果只验证机制能在适当分布上工作。', '',
      '## 正确性与统计边界', '',
      f"- {s['geometry_cells']} 个布局/数据/阈值/界配置全部通过完整参考占用检查；被剪掉的 tile 没有参考命中。",
      f"- GPU 准入 {s['admission_calls']} 次完整调用（其中 36 个普通配置），另有 {s['warmups']} 次预热、"
      f"{s['retained_measurements']} 次保留计时、{s['single_queries']} 次完整单次查询、"
      f"{s['ideal_diagnostic_queries']} 次理想诊断；每次输出数量和完整 SHA256 匹配冻结 FP64 参考。准入还逐项比较数组。",
      '- 新增 GPU 核只做整数 ID 映射，已检查空输入、边界、非整块长度、超过有符号 32 位的 ID 和 canary。'
      'F8、F16 算术核保持原有二进制身份。',
      '- 独立扩展精度抽查确认了投影矩阵范数裕量、256 行投影误差包络及 8192 个原始点对的块界。'
      '它补充源级论证与完整输出核验，不替代普遍性证明；环境和检查结果见 `artifacts/environment.json`。',
      f"- 四个 GPU 独占守护进程全部通过，峰值设备占用 {s['max_device_used_mib']} MiB，未改变时钟或终止其他任务。",
      '- 置信区间以进程和配对重复为单位进行 20000 次分层 bootstrap。只有三个独立进程簇，'
      '区间仅描述本轮重复性；没有独立工作负载确认集，不支持广泛性能主张。',
      '- 数值合同仍是匹配指定 FP64 参考。保守几何界的源级误差论证见 NUMERICAL_SCOPE.md；'
      '保留原 F8/F16 算子的数值前提，不宣称整个系统无条件 exact-real。', '',
      '## 对 flow matching 的判断', '',
      '本实验提供了简单投影排序与块剪枝的实际证据，但没有提供 FM 优于简单方法的证据。'
      '若继续，先收缩到有贡献的投影界、降低预处理成本，并针对“实际空块很多但无法判出”的差距研究更紧界。'
      'FM 必须在相同可靠判界与完整成本条件下超过 PCA 等简单方法，才能成为必要组件。', '',
      '![结果图](analysis/block_feasibility.png)', '',
      '## 文件与复现', '',
      '- `PROTOCOL.md`、`NUMERICAL_SCOPE.md`：冻结设计与误差范围。',
      '- `artifacts/screen_freeze.json`、`artifacts/timing_freeze.json`：筛选及计时依赖哈希。',
      '- `artifacts/selection.json`：基于真实几何剪枝率选择布局，未使用 GPU 耗时或理想占用。',
      '- `results/screen_*.json`、`results/confirm_*.json`、`raw/`：全部筛选、原始时间和守护记录。',
      '- `analysis/summary.json`、`summary.csv`、`raw_times.csv`、`geometry.csv`、`block_feasibility.pdf`。',
      '- `python analyze.py` 重建统计、报告与图。GPU 工作依赖原实验目录中的冻结数据、参考数组和算术核。',
      '- 新目录运行顺序：CPU `src/screen.py` → 守护下 `src/admit.py` → `run_campaign.py`；已有结果禁止覆盖。', '']
    (HERE/'REPORT.md').write_text('\n'.join(lines))

if __name__ == '__main__':
    main()
