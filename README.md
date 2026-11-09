# TRM Sudoku-Extreme reproduction

## 已完成的复现结果

TRM 已完成 **65,100 次更新**和 **422,786 道官方测试题**。

| 指标 | 验证集（1,000 题） | 官方测试集（422,786 题） |
| --- | ---: | ---: |
| 整盘完全正确率 | 87.40% | **86.16%** |
| 未知格准确率 | 93.20% | 92.46% |
| 全部单格准确率 | 95.32% | 94.80% |

官方测试整盘正确 **364,254 / 422,786**，95% Wilson 区间 **86.05%–86.26%**。相对论文参考 87.4% 低 **1.24 个百分点**，位于本次预期约 ±2 个百分点范围内；采用新固定 seed-0 训练子集，无法保证与作者原始 1K 完全一致。验证集 87.4% 与官方测试 86.16% 为两个不同结果。

详细结果和可追溯记录见 [reproduction/final/](reproduction/final/README.md)，历次验证见 [validation_history.csv](reproduction/final/validation_history.csv)。数据、checkpoint、逐题预测和完整日志保留本机；`trm_ode/` 仍只有占位结构。

新增独立项目；原有 `/workspace/_trm_revision_probe`、`/workspace/sudoku`、数据和 checkpoint 均未改动。

```
project/
  trm/        model.py, train.py, evaluate.py, analysis.ipynb
              runtime.py, data.py, vendor/, vendor_optimizer/
  trm_ode/    仅四个占位文件，暂不实现
  data/       共享、固定、带哈希及原始索引的训练/验证/测试和增强池
  runs/trm/   性能实测、校验与正式训练 sudoku_mlp_seed0/
  runs/trm_ode/
```

模型代码直接取自官方 TRM revision `c01103738605ba39d1430519b1ee0c62f4c707f8`；模型定义未改动。
来源：[官方代码](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)、[论文](https://arxiv.org/abs/2510.04871)。TRM-MLP 的论文参考值 87.4%，官方 README 预期约 87% ±2%，本次结果以本机完整测试为准。

训练协议：512 hidden、2 层共享 MLP、H=3、L=6、ACT 上限16、BF16、StableMax、Adam-ATan2、LR=1e-4、warmup=2000、WD=1、EMA=.999、有效 batch=768。README 的 50K epochs 对应实际官方循环 10×6510=65100 次优化器更新；论文正文提及 60K epochs，本项目明确执行 README 配方。首次运行会从随机初始化训练，未加载网上预训练权重。

固定数据版本 `sapientinc/sudoku-extreme@58942f96baeb572ca3127e2a9e9c70f330783d6b`。随机种子0抽1000道训练原题和不重叠的1000道验证题，官方测试422786题。每题保留原题并生成1000个官方变换；记录索引、哈希与划分。原作者未公布原题索引且其构建脚本未固定 NumPy seed，因此本项目是相同协议的新固定划分，不能保证与论文同一1K。

为避免测试集挑模型：每6510步在独立验证集评估EMA；训练结束后对最终EMA checkpoint跑完整官方测试。不会把子集结果或测试未完成的结果报告为完整复现。原始预测无求解器、无约束后处理；验证和测试生成不读取标签。

## 使用

默认 Python：`/venv/main/bin/python`。依赖在项目 `.deps-cuda/` 中，未替换系统 PyTorch。Adam-ATan2 0.0.3 源码在 `trm/vendor_optimizer/`，仅将编译目标改为 `sm_120`，C++17改为C++20，以适配此GPU/PyTorch；保留原许可证。

```bash
cd /workspace/project
/venv/main/bin/python trm/data.py
# micro-batch 和 compile 参数须使用 runs/trm/ 的实测选择；正式启动由 supervisor 管理。
/venv/main/bin/python trm/train.py --micro-batch 256 --compile full --resume
# 独立完整测试，路径必须指向该实验输出的EMA checkpoint：
/venv/main/bin/python trm/evaluate.py --checkpoint runs/trm/sudoku_mlp_seed0/ema_step_065100.pt --split test --output runs/trm/sudoku_mlp_seed0/official_test_final --batch-size 256
```

正式任务服务名 `trm_reproduction`（安装并启动后可用 `supervisorctl status trm_reproduction`）。训练日志 `/var/log/portal/trm_reproduction.log`，结构化进度 `runs/trm/sudoku_mlp_seed0/status.json`，训练曲线 `train.jsonl`。`latest.pt` 存完整优化器、EMA、递归状态、采样位置和随机数；定期导出独立EMA快照，自动重启时使用 `--resume`。实验目录有进程锁，阻止同时写入。不要在服务运行期间另启同目录训练。

`status.json` 的 `complete=true` 仅在训练和官方422786题测试全部完成后写入。结果不保证达到参考值；不足时需分析训练曲线和复现差异。

本实例 `/workspace` 不是持久卷；停止/启动保留文件，recycle/destroy会丢失。训练完成前不要销毁实例，需要长期保留时将产物另存。

## 本机验证

检查 `runs/trm/verification.json`、`benchmark_*.json` 和 `smoke_*/`。测试涵盖模型与固定官方源码一致、增强合法性、采样续接、优化器状态续接和小批训练/评估闭环。吞吐以实测为准，显存占满不是优化目标。

## 性能选择

RTX 5070 Ti 16GB / PyTorch 2.14.1+cu130 实测：256 micro-batch + full compile 约1.22秒/update、峰值10.10 GiB；block compile约1.24秒；不编译约2.00秒、峰值13.63 GiB。128 micro-batch + block约1.26秒。384 micro-batch显存不足。选择256/full，有效batch仍为768。估算纯训练约22小时，额外需要验证、保存和完整测试时间。

完整数据集的1,001,000份增强均经过合法解和clue一致性校验。最终结果另写 `runs/trm/sudoku_mlp_seed0/RESULTS.md`，并明确 `replication_target_met`，避免将跑完训练等同达到论文效果。

## CIFAR-10 生成实现

| 实现 | 目录 | 输出头与训练目标 |
| --- | --- | --- |
| 原始TRM Mixer | [cifar10_looped_flows](cifar10_looped_flows/README.md) | 线性图像输出，普通MSE、均匀顺序时间 |
| 残差卷积版 | [cifar10_looped_flows_conv_archive](cifar10_looped_flows_conv_archive/README.md) | 零初始化3×3残差卷积，支持旧checkpoint迁移及从头训练 |
| v2 / JiT | [cifar10_looped_flows_v2_jit](cifar10_looped_flows_v2_jit/README.md) | JiT调制输出头、加权MSE、logit-normal顺序时间、分母下限 |
| v3 / attention TRM + RoPE / JiT | [cifar10_looped_flows_v3_jit](cifar10_looped_flows_v3_jit/README.md) | 独立attention TRM模块、RoPE、单时间JiT、Nsup=4 deep supervision、输出仅gamma缩放 |

原 Looped Flows 方法方案见[cifar10_looped_flows_trm.md](cifar10_looped_flows_trm.md)，v3 改用单时间 JiT。四个实现均使用原TRM Adam-ATan2，CIFAR-10固定45000/5000划分，不继承上方Sudoku指标。

旧实验均已停止，旧服务模板关闭autostart。v3新增独立训练服务，运行状态见其run目录。v2已完成100000步，但自由生成存在严重对比度膨胀，质量未达目标，未计算FID；检查见[v2采样评估](cifar10_looped_flows_v2_jit/results/REVIEW.md)。代码、运行脚本和说明保留在仓库；数据、完整日志、样图和模型权重仅留本机，不上传Git。
