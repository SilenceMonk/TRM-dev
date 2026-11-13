# 逐 token ACT / 最大 16 个监督循环

本次实验已于 **2026-11-13 20:42 UTC 按用户要求手动停止**，完成 3 个整轮及部分第 4 轮，共 29,212 批输入、467,392 次局部优化更新。最佳完整验证 EMA loss **2.183540**、PPL **8.877675**；尚未触发预设早停条件。停止点和实测结果见 [ACT_RESULTS.md](ACT_RESULTS.md)。

独立入口：`model_act.py`、`train_act.py`、`evaluate_act.py`、`verify_act.py`。此前固定单周期 raw/EMA 实验代码、结果及权重保留。新实验从随机初始化训练，继续使用相同 TinyStories splits、8192 BPE、D=256、上下文 256。

## 与原 TRM 的对应及语言模型适配

- 两层共享 causal attention/SwiGLU，H=3、L=6。每个外层监督循环仍有 21 次共享模块调用；前两个 H round 无梯度，仅最后一个 H round 反传。
- 外层 `halt_max_steps=16`，每个 token 有独立的 `steps` / `halted` 和两个递归状态。达到上限强制停止；更早可由自己的 `q_halt > 0` 决定停止。q head 零权重、bias=-5，训练探索概率 .1；使用原 TRM `no_ACT_continue=True` 的单 halt-head 目标。
- 原 TRM 按整题/样本 halt；这里改为逐 token。因果 attention、halt head、冻结状态均只依赖当前位置及左侧输入。标签不进入 forward 或停止决策，只用于训练损失。
- 每个监督循环对尚未停止的 token 计算平均 next-token CE，加 `.5 * BCE(q_halt, argmax(logits)==target)`，完成一次优化器和 EMA 更新；携带的 h/l 在循环边界 detach。停止的 token 不再承担后续监督损失，且在每个共享 block 内保持状态不变，供后续位置读取上下文。
- 同一输入窗口保留到所有 token 停止或达到第 16 轮，然后换下一批窗口；不会把不同故事的 token 随意拼接到旧递归状态。每次从初始状态开始独立评估窗口。

这是稠密 GPU 实现：已停止 token 的状态逻辑冻结，但 attention 仍以稠密 kernel 计算它们作为上下文的 K/V。因此平均 token 循环数下降不意味着 FLOPs 同比例下降。只有所有 token 都停止，才提前结束整批循环。

原 TRM 在验证时固定用满上限；本实验主指标使用确定性的逐 token ACT（无训练探索），从最终停止状态输出归一化 softmax 分布。`--force-max` 可另行评估固定 16 轮，作为诊断；两种协议有独立的 `inference` 标记，不能混为同一条验证曲线。

## 深监督、计数及恢复

日志 `step` 表示已完成的输入窗口批次数；`optimizer_updates` / `ema_updates` 表示实际局部优化次数，每批最多 16 次。warmup=200 按局部优化次数计算。EMA=.999，每次局部更新后平均一次。

`train_loss` 是局部循环中活跃 token CE 的平均，`halt_loss` 单独报告；训练附带停止步数直方图。验证 loss 为所有计分 token 的平均 CE，不包括 halt 辅助损失，PPL=exp(loss)，报告平均循环数、达到上限的比例和直方图。

checkpoint 只在完整输入窗口批次结束后保存，包含 raw/EMA、优化器、采样/RNG 及早停状态；SIGTERM 在当前输入批次完成后保存。断点恢复不需要重建跨循环 autograd 图，意外中断最多从上一个完整保存点重放。

## 运行与验证

```bash
.venv/bin/python trm_tinystories/verify_act.py
bash service/trm_tinystories_act16.sh
.venv/bin/supervisorctl -c runs/trm_tinystories/supervisord.conf status
tail -f runs/trm_tinystories/d256_seed0_act16_ema/service.log
```

正式 batch/micro-batch=192。每 20 批在固定 65,536 个 validation tokens 上分别评估 raw/EMA；每 1000 批及每轮结束评估所有 4,691,712 个完整验证窗口 tokens。`best_raw.pt`、`best_ema.pt` 各自按完整 ACT 验证 loss 保存，`best_overall.json` 指向二者最低记录。至少 3 轮且连续 3 轮改善不足 .001 时早停；未改善则将 LR 减半至最低 1e-6，没有固定轮数上限。

```bash
.venv/bin/python trm_tinystories/evaluate_act.py \
  --checkpoint runs/trm_tinystories/d256_seed0_act16_ema/best_ema.pt --compile
# 完整训练 checkpoint 中指定 EMA；固定 16 轮诊断：
.venv/bin/python trm_tinystories/evaluate_act.py \
  --checkpoint runs/trm_tinystories/d256_seed0_act16_ema/latest.pt \
  --weights ema --force-max --max-tokens 65536 --compile
```

RTX A4000 实测 batch=192 时峰值分配 11.74 GiB、reserved 13.22 GiB，GPU 利用率 100%、功耗约 139/140W。跑满 16 轮约 5.09 秒/批（约 .318 秒/局部优化），独立输入约 9,650 tokens/s。每轮数据需 9,498 批；纯训练约 13.4 小时，另加 raw/EMA 验证和保存。实际时长取决于训练与验证的停止步数；原单周期实验的一小时/轮估计不适用于此版本。
