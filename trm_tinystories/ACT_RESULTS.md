# TinyStories D256 token ACT16：手动停止记录

2026-11-13 20:42 UTC 按用户要求停止实验。Supervisor 状态为 `STOPPED`，GPU 已释放；不是自动早停，也未宣称已达到最低可实现验证 loss。

## 停止点

- 完成输入窗口批次：29,212；局部优化器 / EMA 更新：467,392。
- 完成 3 个整轮，正在第 4 轮；每轮 9,498 批。
- 累计运行 183,693.69 秒，约 51 小时 2 分钟。
- 当前 LR=1.5e-4；第 3 轮没有达到改善阈值后已减半。
- 早停等待计数为 1，尚未达到连续 3 轮的阈值。
- `latest.pt` 已保存 raw、EMA、优化器、采样排列/位置、RNG、学习率及早停状态，支持在完整输入窗口边界继续。

## 已观测到的最佳完整验证结果

| 权重 | 输入批次 | 局部优化次数 | Loss | PPL | 平均 token 循环数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw | 29,000 | 464,000 | 2.336479 | 10.344745 | 8.7247 |
| **EMA** | **29,000** | **464,000** | **2.183540** | **8.877675** | **8.3502** |

评估使用官方独立 validation split 的 4,691,712 个计分 tokens，窗口长 256，同一 8192 BPE；按 token 加权 CE，PPL=exp(loss)。这里是确定性逐 token ACT 指标，最大循环 16，标签不参与停止决策。该 EMA 点约 47.51% 的 token 到达第 16 轮。

此前单周期 Nsup=1 raw/EMA 实验的最佳 EMA PPL=6.553722；本次 ACT 在停止前未超过它。两次实验的 batch 大小、数据复用频率和停止计算协议不同，不构成严格控制变量的 ACT 消融。降低学习率后 ACT 指标确有改善，因此这次手动停止不能写为“已完全收敛”。

## 代码与本机产物

实现及训练协议见 [ACT.md](ACT.md)。入口 `service/trm_tinystories_act16.sh`，模型、训练、评估及检查分别为 `model_act.py`、`train_act.py`、`evaluate_act.py`、`verify_act.py`。

本机 run：`runs/trm_tinystories/d256_seed0_act16_ema/`。重要文件为 `latest.pt`、`best_raw.pt`、`best_ema.pt`、`best_overall.json`、`status.json`、`validation.jsonl`、`epochs.jsonl` 和 `source/`。数据、tokenizer、完整日志和权重保留本机，不进入 Git。

停止后保持服务 `autostart=false`；仅在明确要求恢复时重新启动。检查服务状态：

```bash
.venv/bin/supervisorctl -c runs/trm_tinystories/supervisord.conf status
```
