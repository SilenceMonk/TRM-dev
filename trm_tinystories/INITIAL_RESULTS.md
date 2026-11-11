# D256 TinyStories：首次完整验证

记录时间：2026-11-11 05:01 UTC。训练仍在后台进行，当前结果不代表收敛，也不代表一轮训练已完成。

模型：3,801,088 参数，宽度 256，4 heads，2 个共享 causal attention/SwiGLU block，H=3/L=6，Nsup=1。上下文 256，训练集拟合的 8192 byte-level BPE，输入输出 embedding 共享，标准 next-token softmax CE。完整配置及复现入口见 [README](README.md)。

数据固定为官方 TinyStories revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`。训练 2,119,719 篇、466,811,075 tokens；独立验证 21,990 篇、4,691,751 tokens。默认预算为训练集一轮，8,141 optimizer updates，batch/micro-batch 均为 224。

| Update | 验证范围 | Tokens | Loss | PPL |
| --- | --- | ---: | ---: | ---: |
| 0 | 固定验证子集 | 262,144 | 9.496113 | 13,307.894 |
| 100 | 固定验证子集 | 262,144 | 6.288571 | 538.383 |
| 200 | 固定验证子集 | 262,144 | 4.695264 | 109.428 |
| 500 | 固定验证子集 | 262,144 | 3.339706 | 28.211 |
| 1000 | 固定验证子集 | 262,144 | 2.843988 | 17.184 |
| **1000** | **全部完整验证窗口** | **4,691,712** | **2.858084** | **17.428** |

完整验证耗时 20.63 秒。按 token 加权，`PPL=exp(loss)`，每个窗口重置状态；EOS 计分，开头无目标的 token 和末尾不足完整窗口的部分不计。PPL 只可在相同 tokenizer 与评估协议下比较。

RTX A4000 16GB / PyTorch 2.7.1+cu128 实测：稳定约 0.37 秒/update、15.5–15.9 万 tokens/s；峰值分配 12.66 GiB，reserved 13.69 GiB。30 秒监测 GPU 利用率平均 99.33%（95–100%），显存 14,235 MiB，平均功耗 137.51W，峰值 139.85W / 140W。

已通过未来 token 扰动不影响前缀预测、标签右移、有限梯度、优化器恢复逐位一致检查，以及训练/保存/恢复/周期完整验证/独立评估闭环检查。检查记录在本机 `runs/trm_tinystories/`。

实时结果以 `runs/trm_tinystories/d256_seed0/` 为准：`status.json`、`validation.jsonl`、`validation_full_latest.json`、`service.log`、`latest.pt`、`best.pt`。本文件仅记录首次完整验证的固定快照。一小时左右的估计仅指跑完一轮及验证，不表示模型已经完全收敛。
