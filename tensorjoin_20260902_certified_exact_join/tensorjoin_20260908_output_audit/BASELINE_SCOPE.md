# 本轮基线审计的范围

本轮执行了同机、同输入、同六个阈值、同完整输出合同的三个固定路径。
主张范围是保留算子的比较，不是已经击败所有相似连接系统。

| 对照 | 本轮执行与核验 | 仍有限制 |
|---|---|---|
| F8 | 冻结 INT8 Tensor Core 初筛，FP32 精化，冻结 FP64 终端；新输出只处理整数 ID | 没有新精度机制；数值与数据范围继承既有合同 |
| F16 | 相同的既有融合 FP16 初筛；每个阈值的 cubin/PTX/IR 与上轮字节比对 | 固定数据的完整参考核验，不等于消除 Tensor Core 数值前提 |
| F32 | 已有 pedantic cuBLAS FP32 GEMM，冻结 norm/classifier/terminal，保持库哈希与数学模式 | 面板布局与 TC 路径不同；不是所有 FP32 距离实现的最优性证明 |
| FaSTED | 核对论文中的 TC 数据复用、流水、全量距离计算与准确性说明 | 本轮未运行；原文报告相对 FP64 的集合差异，不能直接计入同合同基线 |
| RT-HiSS | 核对索引与精化、两遍分批、结果掩码与传输消融 | 本轮未运行；exact 术语尚未审计为本项目冻结 FP64 过程，原输出表示也不同 |

FaSTED 的 FP16/FP32 距离执行值得作为性能方法参考，但为它补充返回结果
的 FP64 重算不能修复候选生成阶段的漏报。要进入本项目的同答案合同
比较，接受和拒绝均需要参考判定覆盖论证，并统一完整输出成本。

RT-HiSS 已把输出压缩、分批和传输作为系统优化并做消融。因此本轮用
成熟 CUB 替换慢 CPU 输出流程，只应记为公共基线强化，不能据此宣称
首次优化高维连接结果物化。若以后复现 RT-HiSS，需要把其压缩结果
还原为同样的完整排序 CPU ID，或另立双方一致且有实际用途的输出合同。

已核查的一手来源（2026-09-08）：

- FaSTED，ICPP 2025，§3、§4.6：
  https://arxiv.org/html/2508.21230v1
- RT-HiSS，2026-09-02 预印本，§IV-D–F、§V-G–H：
  https://arxiv.org/html/2609.01975v1
- NVIDIA CUB DeviceRadixSort 官方实现；实际构建使用本机 CUDA 13.1
  自带 CUB，具体版本、源码依赖和产物哈希见 artifacts/build.json：
  https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/device_radix_sort.cuh
