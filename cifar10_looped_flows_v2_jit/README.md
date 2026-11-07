# CIFAR-10 Looped Flows v2 / JiT

从最初无卷积的 TRM Mixer 开始，加入 JiT 式目标、时间分布、分母下限及输出头。全新 seed=0 训练，不加载任何旧 checkpoint。独立项目不是完整 JiT Transformer 复现。

## 当前结果

本地训练已完成100000次局部更新，服务停止并关闭autostart。采样存在对比度膨胀和裁剪饱和，生成质量未达到目标；未计算FID。详细检查见[results/REVIEW.md](results/REVIEW.md)。完成训练不代表方法有效。

## 本次改动

- 主干保持 p=4、64 tokens、D=512、两层共享 post-norm Mixer、R=3、m=6。前两周期 no_grad，最后周期完整反传；无卷积头。
- 目标为 `mean(((prediction-Y) / clamp_min(1-t, .05))**2)`。它等价于官方把预测和目标都换算为速度后的 MSE；分母被截断时不要将目标简单替换为未截断的 `Y-noise`。
- 先独立采 K+1=5 个 `sigmoid(N(-.8,.8²))` 时间，再排序并使用前K=4个。继续保留原 Looped Flows 的顺序、共享噪声、真实插值输入、跨时间携带并 detach 隐状态、每个局部时间点单独更新参数。排序与丢弃最大点会改变实际调用的时间边际分布；不是未经修改的 JiT 单时刻采样。
- 采样仍用 stateful Euler，速度为 `(prediction-y)/max(1-t,.05)`。每一步包括末步均执行 Euler；N=32/64 时末步不能直接返回 prediction。积分和时间系数 FP32，不在积分中裁剪。
- 输出采用 JiT 官方 FinalLayer：带可学习缩放的 RMSNorm（eps=1e-6）→时间条件 shift/scale 调制→带偏置的512→48线性投影。modulation 和输出投影权重、偏置均零初始化，RMSNorm缩放为1。主干使用原TRM初始化。首步零输出使主干首步梯度为零属于预期，第二步开始获得非零梯度。
- 无条件任务，head condition 使用现有 TRM 的 `1→512→512` 时间 MLP 输出；未引入 JiT 正弦时间嵌入、类别嵌入或CFG。patch 内像素排序保持原TRM实验的 C×4×4，不切换为官方JiT的4×4×C；编码与解码一致。Head 用FP32计算，主干BF16。可训练参数 **5,655,600**。

输出头与默认 P_mean/P_std/t_eps 依据官方固定revision：
https://github.com/LTH14/JiT/tree/cbc743a2ada5e9762697da2c83f8c4f8379e8c17

`reference/` 保留对应代码与MIT许可证；`jit_head.py` 复用官方 FinalLayer/RMSNorm，只移除了自动编译装饰器。

## 首轮运行配置

数据复用官方train50k固定45000/5000划分，RGB归一化[-1,1]、无增强、无类别条件。原 TRM Adam-ATan2 0.0.3，LR=1e-4、betas=(.9,.95)、WD=1，warmup=2000后恒定LR，EMA=.999。总100000次局部更新 = 25000条batch序列。

本轮 batch=512。batch=640虽完成8步真实数据smoke，训练期间出现显存分配失败后重试；512的40步smoke稳定，峰值约11.73GiB，稳态约.404秒/局部更新。新输出头有额外显存开销，因此与旧640版本的训练图像处理量不同。100000次局部更新为51.2M次图像局部监督、12.8M次独立序列图像抽样；不要混淆与原batch640实验的预算。

正式服务 `cifar10_looped_flows_v2_jit`，run目录 `runs/cifar10_looped_flows_v2_jit/seed0/`，日志 `/var/log/portal/cifar10_looped_flows_v2_jit.log`。

```bash
cd /workspace/TRM-dev
/venv/main/bin/python cifar10_looped_flows_v2_jit/verify.py
# 手动运行时不要和supervisor写同一目录。
/venv/main/bin/python cifar10_looped_flows_v2_jit/train.py --compile --resume
supervisorctl status cifar10_looped_flows_v2_jit
```

新目录没有checkpoint时，`--resume`从随机初始化开始；之后只恢复本目录checkpoint，不查找旧实验。`startup_step_000000.json`记录未加载checkpoint、optimizer状态为空；初始随机模型/EMA/空优化器及RNG保存为第0步latest.pt。定期每1000步保存，记录数据采样所需CPU/CUDA RNG，仅在完整4步序列边界checkpoint。

每5000步对全部5000张验证集计算固定t=.1/.3/.5/.7/.9的 carried/reset 原始MSE和加权MSE；用相同64张初始噪声生成N=8/16/32/64样图，并记录各N输出的均值、标准差、越界比例、最小值、最大值（sample_stats_*.json）。这些样图是自由生成，不是验证集重建。训练新loss不能直接与旧普通MSE数值比较，日志同时记录raw_mse。

FID保持独立评估，用torch-fidelity==0.4.0、官方train50000真实统计；5000张趋势筛查和50000张最终指标分开注明。新目标不预设保证解决过强对比度或提升FID。

```bash
/venv/main/bin/python cifar10_looped_flows_v2_jit/evaluate.py \
  --checkpoint runs/cifar10_looped_flows_v2_jit/seed0/ema_100000.pt \
  --output runs/cifar10_looped_flows_v2_jit/seed0/fid5k_N32 \
  --num-images 5000 --flow-steps 32 --compile --fid
```

## 验证与旧实验

`runs/cifar10_looped_flows_v2_jit/verification.json`验证：分母下限激活时仍与官方损失等价、时间采样/顺序准确、head与独立执行的官方代码逐位一致、零初始化与次步主干梯度、14次no_grad/7次grad、Adam-ATan2下一步精确重放、包含末步的clamped Euler。

`smoke512/`保留40步真实数据训练、checkpoint、验证、EMA、采样全链路；`smoke640/`保留显存探测记录。均非正式实验结果。

旧服务 `cifar10_looped_flows` 已停止并禁用autostart。原 `cifar10_looped_flows/model.py` 恢复最初代码，SHA256与初始source_manifest.json一致。卷积版代码在 `cifar10_looped_flows_conv_archive/`，旧runs全部保留。本机/workspace非持久卷，recycle/destroy会丢失产物。
