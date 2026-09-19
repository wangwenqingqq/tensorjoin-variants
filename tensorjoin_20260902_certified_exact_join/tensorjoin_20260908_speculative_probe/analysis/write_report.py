import json
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
s=json.loads((HERE/'analysis/summary.json').read_text())
tracepath=HERE/'analysis/trace_summary.json'
trace=json.loads(tracepath.read_text()) if tracepath.exists() else None
names=dict(original_full='原顺序 FP16',project64='投影＋整块',packed16='行列压紧',
    sequential='串行验证后执行',pipeline='普通流水线',draft_serial='串行提案',speculative='投机执行')
def f(v,n=2):return f'{v:.{n}f}'
mix=s['mixtures'];cmp=s['comparisons'];best=s['strongest_fixed_control']
lines=['# TensorJoin 可认证投机执行小实验','',
'实验日期：2026-09-08。主机：8p / gpu-host-8。GPU2：RTX PRO 6000 Blackwell Server Edition。','',
'**结论：本轮投机执行未通过预设的 5% 净收益门槛，停止这套固定原型的继续调参。**' if not s['pass_gate'] else '**结论：本轮投机执行通过了预设收益门槛，仍需独立数据与外部基线验证。**','',
]
# Keep the result statement generated from final retained data only.
lines += [
f"六阈值等查询权重下，普通流水线为 **{f(mix['pipeline']['mean_ms'])} ms**，投机执行为 **{f(mix['speculative']['mean_ms'])} ms**。投机相对流水线的时间节省为 **{f(cmp['pipeline']['saving_percent'])}%**（负数表示更慢）。"
f"最强固定旧对照为{names[best]}，平均 **{f(mix[best]['mean_ms'])} ms**；投机相对它的时间节省为 **{f(cmp[best]['saving_percent'])}%**。",'',
'## 1. 实际实现与对照','',
'使用完整 CIFAR 60,000×512 FP32 输入和六个冻结阈值。复用输入构造的 PCA 排列与 64 维投影；每次查询重新上传 GPU 并构造完整维度 FP16 元数据。最终返回完整、升序的有向 CPU pair 数组，含自身连接，匹配冻结 FP64 判定合同。','',
'提案规则固定为每个 64 点块取偏移 0、4、…、60 的 16 个代表点，计算其 16×16 个 64 维投影距离。完整验证仍检查每个块的全部 64×64 投影 pair。设提案集合为 D，可靠未决集合为 U，实际执行 D 与 U\\D 两个互不相交的队列。任何提案遗漏均被完整验证发现并补全。','',
'| 方式 | 实际流程 |','|---|---|',
'| 串行验证后执行 | 全量生成可靠队列，再计算 |',
'| 普通流水线 | 下一批可靠判界与当前批计算重叠，当前批等待验证完成 |',
'| 串行提案 | 生成提案和补全队列，再依次计算二者 |',
'| 投机执行 | 同样的验证预取距离，草稿计算可以先于当前验证完成，随后计算补全队列 |','',
'四组使用相同的固定 4096 输入 tile 分批、GPU 队列和距离核。设备计数控制实际计算，队列未使用位置直接退出。草稿与补全共享每批的一次 FP32 精化／FP64 终端，避免把两次精化同步额外强加给投机组。普通流水线和投机组最多提交当前及下一批验证，不无限提前排队。','',
'额外保留原顺序 FP16、原有全局压缩的 project64 和 packed16 行列压紧三个强对照。它们在本轮同一 GPU、同一进程组内重测；没有使用旧实验耗时计算加速。新四组的固定输入分批不同于旧对照的全局工作表压缩，这一影响不能算作投机收益。','',
'## 2. 完整查询时间','',
'单位 ms。先取每个进程四次保留测量的中位数，再对三个进程等权平均。全部包含上传、元数据、提案／验证、任务队列、事件等待、距离计算、精化、ID 恢复、CUB 排序和 CPU 下载。CPU 索引构建单独报告。','',
'| 阈值 | 原 FP16 | 投影＋整块 | 行列压紧 | 串行验证 | 流水线 | 串行提案 | 投机 |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for row in s['rows']:
    lines.append('| '+row['cell']+' | '+' | '.join(f(row[m+'_ms']) for m in s['methods'])+' |')
lines+=['| 六阈值等权 | '+' | '.join(f(mix[m]['mean_ms']) for m in s['methods'])+' |','',
'| 比较：投机相对基线 | 整体时间节省 | 三进程各自节省 |',
'|---|---:|---|']
for base in ['sequential','pipeline','draft_serial',best]:
    v=cmp[base];lines.append('| '+names[base]+' | '+f(v['saving_percent'])+'% | '+', '.join(f(x)+'%' for x in v['process_saving_percent'])+' |')
lines+=['',
'门槛要求：投机相对普通流水线及最强固定旧对照，在三个确认进程中的六阈值混合耗时均至少节省 5%。逐点波动及进程比值范围完整保留在 summary.csv；这些范围不是置信区间。三个进程仅描述本机重复性，不支持跨数据泛化。','',
'![结果图](analysis/speculative_probe.png)','',
'## 3. 提案覆盖与补全','',
'以下统计来自准入的实际在线 GPU 标志；没有读取参考答案生成任务。','',
'| 阈值 | 可靠未决块 | 提案覆盖的未决块 | 覆盖率 | 补全块 | 额外无效草稿块 |',
'|---|---:|---:|---:|---:|---:|']
for v in s['coverage']:
    lines.append(f"| {v['cell']} | {v['verified_tiles']:,} | {v['drafted_verified_tiles']:,} | {f(100*v['drafted_verified_tiles']/v['verified_tiles'])}% | {v['repair_tiles']:,} | {v['unnecessary_draft_tiles']:,} |")
lines+=['',
'正常提案在本输入上没有增加完整距离执行的 tile 总数；其额外成本仍包括代表点距离、第二份工作队列、每批第二次距离核发射，以及并发管理。覆盖率高不等于能隐藏更多延迟：验证器与后续计算之间是否仍有等待才是决定因素。','',
'## 4. 实际重叠诊断','']
if trace:
    lines+=['CUDA trace 是正式计时结束后的独立诊断，不计入速度统计。下面的重叠由 Nsight Systems 实际 kernel 时间戳计算，排除了仅有流事件区间交叠的情况。','',
        '| 方式 | 验证 kernel 总时长 ms | 与距离 kernel 重叠 ms | 本批计算开始时验证已完成 |',
        '|---|---:|---:|---:|']
    for m,v in trace['methods'].items():
        lines.append(f"| {names[m]} | {f(v['verify_kernel_ms'],3)} | {f(v['verify_scan_overlap_ms'],3)} | {v['current_verification_done_before_scan']}/{v['groups']} |")
    lines+=['',
        '两个流并未在本原型中形成充分的 kernel 重叠：流水线这次追踪没有观测到验证核与距离核交叠；投机仅约 0.021 ms，提案核本身约 0.404 ms。投机的 108 批中，107 批在草稿距离核启动前，本批验证已完成；只有一批实际越过了尚未完成的当前验证。',
        '',
        '这说明这条队列／主机发射路径很少存在可由投机消除的当前验证等待，同时仍支付提案、额外发射和并发管理成本。不能把普通流水线的提前完成解释为已经实现大量 GPU kernel 重叠，也不能仅由流事件交叠宣称并发加速。',
        '',
        'Nsight 可能扰动主机发射和调度，每种方式只追踪原阈值的一次完整查询；上述数值是机制诊断，不与无 profiler 的正式端到端时间直接相减。未加 profiler 的准入事件同样观察到投机 107/108 批已完成当前验证。结论限于本次固定批次和同一 GPU 实现，不能推断其他投机调度无效。']
else:lines+=['实际 kernel 时间线尚未汇总；事件区间仅作辅助诊断，不用其直接声称 kernel 并发。']
lines+=['','## 5. 构建成本与正确性','',
'| 原始阈值，实际从 CPU 原始向量重新构建 | 构建 ms | 查询 ms | 合计 ms |',
'|---|---:|---:|---:|']
for v in s['cold']:
    lines.append(f"| {names[v['method']]} | {f(v['build_seconds']*1000)} | {f(v['query_seconds']*1000)} | {f(v['seconds']*1000)} |")
lines+=['',
'冷启动各做一次诊断，统一复用前轮的公共构建函数，包括 PCA、排序、投影元数据及包围盒；其中包围盒未用于本轮四种调度。该函数不是每种方法分别最小化后的构建下限，单次值也不作速度确认。没有训练模型，没有读取 oracle 掩码。','',
f"- {s['retained_measurements']} 次保留测量全部通过完整输出个数和 SHA256 核验。",
f"- 全部已完成阶段共 {s['full_calls']} 次完整调用通过，其中 {s['exact_array_calls']} 次逐项比较完整 FP64 参考数组，其余使用完整数组哈希。",
'- 49 次准入包括 42 个方法／阈值配置、三个强制草稿（全空、全满、交替）及四次实际冷启动。主配置的接受、不确定、终端和最终输出计数在七种方法间一致。',
'- 每阈值、每个新调度的完整可靠标志和 pair 计数与继承过滤器完全一致。逐队列检查唯一性、分组边界和 D 与 U\\D 的精确覆盖；覆盖尾块、自身和对角线。',
'- 独立 CPU 直接差分检查每阈值 105 个代表 tile、25,548 个采样 pair，共 153,288 次 pair-阈值检查，零错误；没有样本落入预设的近阈值不确定带。',
'- 输出、第一阶段不确定列表和 FP64 列表的前后哨兵均通过。新编译核无寄存器溢出到本地内存。',
'- 数值保护沿用前轮约 0.0136634 的投影余量，范围与证明见 [NUMERICAL_SCOPE.md](NUMERICAL_SCOPE.md)。不宣称普适 exact-real 语义。','',
'GPU2 UUID 为 GPU-16f27f5a-dfcd-48e0-bb39-bebbe4009245。全部 GPU 守护通过；采样峰值设备用量 '
+str(max(v['max_device_used_mib'] for v in s['guards'] if v['max_device_used_mib'] is not None))
+' MiB，低于 4096 MiB 限额。最终设备与锁检查记录在 results/final_resource_check.json。','',
'## 6. 对 TensorJoin 的判断','',
'本轮补上了“提案能否提前执行并带来净收益”的直接检验。结果不支持将本原型加入主算法，也不支持以投机解码作为新的论文核心。保留强 FP16／投影／压紧对照及正确性结果，停止为这套提案和流水线继续调参。',
'',
'该结论仅针对固定采样提案、64×64 回退、4096 tile 分批、两个流和本数据集。它不排除其他硬件、跨设备传输或存在更长串行依赖时的投机机会。本轮没有比较外部系统，也没有把概率采样分布保证误当成完整连接集合保证。','',
'## 文件与复现','',
'- [冻结协议](PROTOCOL.md)、[数值与覆盖不变量](NUMERICAL_SCOPE.md)。',
'- [执行引擎](src/spec_engine.py)、[GPU 核](src/spec_kernels.py)、[准入与计时驱动](src/run.py)。',
'- [全部逐次计时](analysis/raw_times.csv)、[汇总](analysis/summary.json)、[独立统计核验](analysis/statistical_audit.json)。',
'- results/ 保存各进程与守护记录，raw/ 保存完整日志和 CUDA trace，artifacts/ 保存代码／输入／依赖哈希与编译产物。',
'- 原始数据及参考结果只读引用先前冻结目录；实际依赖源码和二进制归档在 inherited/，大数组引用及哈希见 artifacts/dependency_archive.json。',
'- 远端目录：`@TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join/tensorjoin_20260908_speculative_probe`。',
'- 使用 `@TENSORJOIN_ROOT@/isaacsim6/env/bin/python`，新目录按 guard 下 admission → run_campaign.py → run_profiles.py 顺序复现；不得覆盖现有标签。后处理依次运行 analysis/analyze.py、analysis/statistical_audit.py、analysis/plot.py、analysis/write_report.py。',
'- 所有正式测量代码在准入后冻结，没有按计时修改参数或删除慢样本。开发预检源码单独保存在 development/。',
]
(HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
