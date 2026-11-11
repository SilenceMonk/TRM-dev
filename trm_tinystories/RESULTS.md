# TRM TinyStories D256：raw / EMA 最终结果

2026-11-11 从随机初始化训练，16:13 UTC 达到预设验证早停条件；共 10 轮、81,410 optimizer updates、35,336.38 秒（9 小时 49 分钟）。seed=0，RTX A4000 16GB，PyTorch 2.7.1+cu128。训练结束后进程正常退出。

## 完整验证集结果

| 记录 | 权重 | 轮次 | Step | Loss | PPL |
| --- | --- | ---: | ---: | ---: | ---: |
| 最佳 raw | raw | 7 | 49,000 | 1.915671 | 6.791492 |
| **最佳 EMA** | **EMA** | **7** | **51,000** | **1.880033** | **6.553722** |
| 第三轮结束 | raw | 3 | 24,423 | 1.966510 | 7.145696 |
| 第三轮结束 | EMA | 3 | 24,423 | 1.913571 | 6.777249 |
| 最后一轮结束 | raw | 10 | 81,410 | 1.943150 | 6.980706 |
| 最后一轮结束 | EMA | 10 | 81,410 | 1.904554 | 6.716412 |

所有行均为独立官方 validation split 的 4,691,712 个计分 tokens（18,327 个完整窗口），并非训练集或固定验证子集指标。PPL=exp(token 平均自然对数交叉熵)。窗口长 256，各窗口重置递归状态，EOS 计分；首个无预测目标的 token 及末尾不足窗口的部分不计。

8192 byte-level BPE 仅在前 100,000 篇训练故事上拟合。PPL 只适合在相同 tokenizer、上下文及评估协议下比较。当前没有独立 test split 指标。

## 训练与停止规则

宽度 256，4 heads，2 层共享因果 attention/SwiGLU，RoPE，H=3/L=6，Nsup=1，输入输出 embedding 共享，3,801,088 参数。BF16 运算、FP32 参数，batch/micro-batch=224。固定 TinyStories revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`，训练 2,119,719 篇、466,811,075 tokens。

AdamATan2，betas=(.9,.95)，WD=1，梯度裁剪 1。LR=3e-4，warmup=200，轮末没有实质改善时减半，下限 1e-6。EMA=.999，从随机初始参数开始，每次优化器更新后平均一次。

每 100 步评估固定子集，每 1000 步及轮末评估完整验证集。分别保留所有完整验证点中 raw 和 EMA 的最低 loss；早停依据轮末二者较低的 loss，连续 3 轮未改善至少 .001 且至少完成 3 轮后停止。第 8–10 轮未达到改善阈值；这表示达到已定义的停止规则，不表示证明了全局最优。

## 本机产物与复现

run 目录：`runs/trm_tinystories/d256_seed0_ema/`。推荐推理用 `best_ema.pt`，其余主要文件为 `best_raw.pt`、`latest.pt`（raw + EMA + 优化器 + 采样/RNG + 早停状态）、`RESULTS.json`、`epochs.jsonl`、`validation.jsonl`。数据和权重不进入 Git。

入口：`service/trm_tinystories_ema.sh` / `trm_tinystories/train_ema.py`。完整协议及评估命令见 [README](README.md)；EMA 和断点恢复检查见 `verify_ema.py`。此前 raw 单轮实验及被停止的续训是独立实验，未用作此次随机初始化的权重来源。
