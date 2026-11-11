# TRM TinyStories / D256

## 当前实验：从头训练 raw + EMA，多轮验证早停

本次实验已于 2026-11-11 16:13 UTC 达到早停条件，完成 **10 轮 / 81,410 步**，耗时 **9 小时 49 分钟**。最佳完整验证 EMA loss **1.880033**、PPL **6.553722**；最佳 raw loss **1.915671**、PPL **6.791492**。最终实测及各轮结果见 [RESULTS.md](RESULTS.md)。代码和结果摘要同步 Git；数据、日志、tokenizer 和权重保留本机。

入口 `train_ema.py` / `service/trm_tinystories_ema.sh`，输出 `runs/trm_tinystories/d256_seed0_ema/`。此前 raw 第二轮已按用户要求在第 8899 步停止并保存；新实验不加载旧模型或优化器，从 seed=0 随机初始化开始。缓存数据及 tokenizer 保持相同，便于比较。

EMA 从第 0 步复制 raw 参数，每次优化器更新后按 `ema = .999 * ema + .001 * raw` 在 FP32 中更新；不参与反向传播。每 100 步分别评估两套权重在同一固定验证子集上的 loss/PPL；每 1000 步及每轮结束分别评估全部完整验证窗口。

初始 LR=3e-4，前 200 步 warmup，之后保持不变；轮末验证无实质改善时减半（下限 1e-6）。每轮以 raw/EMA 完整验证 loss 的较小者判断是否至少改善 0.001；连续 3 轮未达到该幅度且至少完成 3 轮后早停。没有固定轮数上限。相对每轮验证锚点累计改善达到阈值会重置耐心；并非声称找到了数学上的全局最低点。周期完整验证用于挑选最佳权重，早停只在轮末判断。

`best_raw.pt`、`best_ema.pt` 分别保存所有完整验证点中各自 loss 最低的模型；`best_overall.json` 指向二者中的最低记录。`latest.pt` 保存 raw、EMA、优化器、EMA 更新次数、采样位置/RNG、学习率及早停状态，支持精确恢复。`validation_full_latest.json` 同时包含 `raw` / `ema` 两组指标；`epochs.jsonl` 记录轮末决策。`status.complete=true` 仅代表达到已定义的验证早停条件。

```bash
.venv/bin/supervisorctl -c runs/trm_tinystories/supervisord.conf status
tail -f runs/trm_tinystories/d256_seed0_ema/service.log
cat runs/trm_tinystories/d256_seed0_ema/validation_full_latest.json
cat runs/trm_tinystories/d256_seed0_ema/best_overall.json
# 独立评估导出的最佳 EMA（自动识别权重类型）
.venv/bin/python trm_tinystories/evaluate.py \
  --checkpoint runs/trm_tinystories/d256_seed0_ema/best_ema.pt --compile
# 从完整 latest 中选择 EMA 或 raw
.venv/bin/python trm_tinystories/evaluate.py \
  --checkpoint runs/trm_tinystories/d256_seed0_ema/latest.pt --weights ema --compile
```

EMA 更新、raw/EMA/优化器精确恢复和早停检查见 `verify_ema.py`；验证记录在本机 `runs/trm_tinystories/verification_ema*.json`。每轮预计约一小时，总轮数和结束时间取决于验证曲线。

## 原始单轮实验（保留）

第一轮最终完整验证：第 8141 步 loss **2.1801**、PPL **8.8468**，覆盖 4,691,712 tokens，耗时 55.25 分钟。第二轮曾从第一轮最终 checkpoint 接续，随后在第 8899 步按要求停止，改为上方从头训练的 EMA 实验。[首次运行记录](INITIAL_RESULTS.md) 保留第 1000 步的历史快照；实时指标以各轮本机 run 目录为准，完成一轮不代表完全收敛。

独立语言模型实现，文件布局沿用 `trm/`：`model.py`、`train.py`、`evaluate.py`、`data.py`、`runtime.py`、`verify.py`、`analysis.ipynb`。原 Sudoku 和 CIFAR 实现不变。

模型固定宽度 256，4 heads，RoPE，两层共享 attention + SwiGLU，post RMSNorm，H=3/L=6；每次前向共享模块调用 21 次，等效 42 个 block。前两个 H round 无梯度，最后一个 H round 反传。每个窗口重置双状态；输入输出 embedding 共享。标准 causal attention，自回归 next-token CE，Nsup=1；不使用 Sudoku 的 puzzle embedding、StableMax 或 ACT/Q head。采用原 TRM 的 AdamATan2、betas=(.9,.95)、WD=1，增加梯度裁剪；学习率 3e-4，warmup 200 updates 后 cosine 到 10%。BF16 运算，FP32 参数，编译共享递归模块。

数据为 [官方 TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)，固定 revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`，四个 train parquet 和独立 validation parquet。8192 词表 byte-level BPE 仅用前 100000 篇训练故事拟合，然后编码全部训练/验证故事，故事间添加 EOS。数据、分词器和源文件哈希写入 `data/tinystories/manifest.json`。该 PPL 仅可与相同分词器、上下文和评估协议比较，不能直接和 GPT-2 词表的 PPL 比较。

训练窗口长 256，标签严格右移一位，无 padding；窗口可跨越故事的 EOS。随机打乱不重叠训练窗口，无放回遍历，默认一整轮全部训练窗口（`--steps 0`）。每 100 步验证固定、均匀分布于整个验证集的 262144 tokens，保存到 `validation.jsonl`；每 1000 步及训练结束评估全部完整验证窗口，尾部不足 256 tokens 的部分不计，单独保存 `validation_full_latest.json`。loss 是 token 平均自然对数交叉熵，`PPL=exp(loss)`。验证窗口内部使用可变长度左侧上下文，每窗口状态重置，EOS 参与评分。

```bash
.venv/bin/python trm_tinystories/data.py
.venv/bin/python trm_tinystories/verify.py
.venv/bin/python trm_tinystories/train.py --benchmark 5 --micro-batch 224
# 正式 batch 以本机 benchmark 结果与 service/trm_tinystories.sh 为准
bash service/trm_tinystories.sh
# 独立完整验证（可能与训练争用 GPU，先停止训练再运行）
.venv/bin/python trm_tinystories/evaluate.py \
  --checkpoint runs/trm_tinystories/d256_seed0/latest.pt \
  --max-tokens 0 --compile --output runs/trm_tinystories/d256_seed0/full_validation.json
```

依赖使用仓库 `.venv` 和已有 AdamATan2 CUDA 扩展，额外依赖见 `requirements.txt`。`latest.pt` 保存模型、优化器、采样排列/位置和 RNG；`best.pt` 按固定验证子集 loss 选择。SIGTERM/SIGINT 在当前 update 完成后保存并退出；`--resume` 检查配置、源代码哈希和数据 manifest。目录锁防止同目录双写。训练和模型权重只保留本机 `runs/trm_tinystories/`，不进入 Git。

检查结果见 `runs/trm_tinystories/verification.json`；benchmark 记录显存和吞吐。实时进度见 `runs/trm_tinystories/d256_seed0/status.json`，验证结果见 `validation_latest.json`。`status.complete` 仅在预算训练和末次完整验证结束后为 true。

## 第二轮续训

第一轮已完成 8141 步，完整验证 loss=2.180051、PPL=8.846756，耗时 55.25 分钟；这表示跑完一轮，不表示完全收敛。

`service/trm_tinystories_epoch2.sh` 从第一轮 `latest.pt` 恢复模型、优化器、采样排列/位置及 RNG，在独立 `runs/trm_tinystories/d256_seed0_epoch2/` 再跑一轮，总步数到 16282。第二轮起始学习率 3e-5，cosine 衰减到 3e-6，warmup=1；数据在第一轮边界自动重新打乱。配置记录父 checkpoint 的路径、SHA256 和起始步数。`phase_step` 表示本轮进度，`step` 表示累计进度；`elapsed_seconds` 包括父阶段的训练时间。每 100/1000 个累计步数做子集/完整验证，末次再做完整验证。

第一轮目录和结果保留；原始训练代码快照在其 `source_epoch1/` 中。继续训练用 `--init-checkpoint` 指定父权重并使用新目录；`--resume` 用于恢复这个续训目录本身。第二轮服务名为 `trm_tinystories_epoch2`：

```bash
.venv/bin/supervisorctl -c runs/trm_tinystories/supervisord.conf status
tail -f runs/trm_tinystories/d256_seed0_epoch2/service.log
```

本机由独立 supervisor 管理：

```bash
.venv/bin/supervisorctl -c runs/trm_tinystories/supervisord.conf status
.venv/bin/supervisorctl -c runs/trm_tinystories/supervisord.conf stop trm_tinystories
.venv/bin/supervisorctl -c runs/trm_tinystories/supervisord.conf start trm_tinystories
tail -f runs/trm_tinystories/d256_seed0/service.log
```
