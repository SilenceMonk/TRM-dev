# CIFAR-10 v3 / attention TRM + RoPE / JiT

当前Nsup=4主实验已在98288步暂停，新开Nsup=1对照；配置、服务和比较口径见 [NSUP_COMPARISON.md](NSUP_COMPARISON.md)。以下四周期说明描述原Nsup=4基线，代码也支持 `--nsup 1`。

将 attention TRM 作为独立去噪模块 `prediction = TRMJiT()(x, t)`：输入带噪 RGB，输出预测的干净 RGB，形状均为 B×3×32×32。无类别条件、无 CFG，从随机初始化训练。原 v2 与 Sudoku 项目保持不变。当前输出头已按 TRM 输出已做 RMSNorm 的设计，去掉 JiT 末端重复归一化，仅保留逐通道可学习 gamma（512维，初始化为1）。

## 递归与 deep supervision

- patch=4，64 tokens，D=512，8 heads，head_dim=64；两层共享原 TRM post-norm attention + SwiGLU，expansion=4。每次 attention 的 Q/K 使用原 TRM **RoPE**（一维 patch 顺序，theta=10000，64 positions）。原 attention、RoPE、SwiGLU、RMSNorm 原语直接复用 `trm/vendor/models/layers.py`。
- 固定 **Nsup=4**。每个监督周期含 **H_cycles=3** 轮，每轮执行 L_cycles=6 次 `l=core(l+h+embedding)`，再一次 `h=core(h+l)`。前两轮 no_grad，第三轮反传。因此一个周期 core 调用14次无梯度、7次有梯度。
- 四个周期使用**同一 batch、同一带噪输入 x、同一 t 和同一干净目标**。每周期计算 loss、backward、optimizer.step、更新 EMA。h/l 仅在这四个周期内携带，周期边界 detach；更新参数后重新计算输入/时间 embedding。不使用 ACT 提前停止或 Q-head。
- 对外 `forward(x,t)` 总是重新初始化 h/l，完整运行四个周期后返回最终 prediction。训练通过 `forward_cycle` 实现逐周期局部更新。ODE 每个时间点都重新调用完整独立模块，**没有跨时间携带状态**。

## 直接 JiT 训练

每张图独立采一个 `t=sigmoid(N(-.8,.8²))`、标准正态噪声 e，构造 `x=t*image+(1-t)*e`。不排序时间，不丢弃最大点，不使用 Looped Flows 时间序列。

```python
loss = mean(((prediction - image) / clamp_min(1-t, .05))**2)
```

此损失等价于 JiT 将预测和目标都换算为速度后的 MSE，包括分母下限激活的情况。输出改为 JiT FinalLayer 的 TRM 适配版本：FP32 `h * gamma`→时间 shift/scale→带偏置图像投影，**不再计算 RMS 或除以 RMS**。gamma 为可训练512维向量，初始化为1；modulation 与最终投影均零初始化。TRM 主干内部的 RMSNorm 保持不变。时间条件保留 v2 的 1→512→512 MLP，图像投影乘 sqrt(512)。共 **7,654,448** 个可训练参数。这里复用 JiT 的训练目标、时间分布和输出头；主干、时间编码、优化器采用 TRM/v2 设计，并非完整 JiT Transformer 配方复现。

采样为 FP32 Euler，速度 `(prediction-x)/max(1-t,.05)`，包含最后一步，积分过程中不裁剪。保存 PNG 时才裁剪到[-1,1]。无条件采样不读取类别或验证图像。

## 训练配置与启动

CIFAR-10 官方 train50k，MD5 校验，固定 seed0 的45000/5000划分；RGB[-1,1]，默认无增强。Adam-ATan2 0.0.3（原 vendored CUDA kernel）、LR=1e-4、betas=(.9,.95)、WD=1、warmup2000后恒定LR、EMA=.999。BF16 主干，FP32 head/loss/ODE。

本机 RTX A4000 16GB / torch2.7.1+cu128：batch=768 + core compile，gamma-only真实数据40次局部更新短测，后20次约 **.712秒/update、1079张局部监督/秒、峰值13.47 GiB**。编译训练时观测 GPU 利用率100%、功耗约139/140W。显存留有运行余量。短测不代表生成质量。

总100000次**局部优化器更新** = 25000个 batch / 100000个监督周期；每 batch 有4个监督周期。batch768对应76.8M次局部图像监督、19.2M次独立图像抽样。默认运行目录 `runs/cifar10_looped_flows_v3_jit/seed0_gamma_only/`，仅恢复本目录 checkpoint。

```bash
# 首次准备依赖（uv、CUDA toolkit 必须可用）
bash cifar10_looped_flows_v3_jit/setup_env.sh
.venv/bin/python cifar10_looped_flows_v3_jit/verify.py
.venv/bin/python cifar10_looped_flows_v3_jit/benchmark.py --batch-size 768 --compile
# 不要与服务同时写 seed0_gamma_only
bash service/cifar10_looped_flows_v3_jit.sh
```

`setup_env.sh` 在 `.deps/` 复制优化器并针对当前 GPU 编译；不修改原 vendor。服务模板在 `service/cifar10_looped_flows_v3_jit.conf`。本机使用项目独立 supervisor，其配置在 `runs/cifar10_looped_flows_v3_jit/supervisord.conf`：

```bash
.venv/bin/supervisorctl -c runs/cifar10_looped_flows_v3_jit/supervisord.conf status
# 安全停止会先完成当前四周期并保存 checkpoint
.venv/bin/supervisorctl -c runs/cifar10_looped_flows_v3_jit/supervisord.conf stop cifar10_looped_flows_v3_jit
```

原带输出 RMSNorm 的 `seed0/` 已在第1512次更新安全停止，checkpoint和日志保留，代码对应 Git revision `a3c59c2`。当前 gamma-only 版本在 `seed0_gamma_only/` 从随机初始化重新训练，不加载原模型；训练恢复和独立评估均检查输出头版本，防止混用。

每1000次更新保存模型、EMA、优化器和CPU/CUDA RNG，仅在完整四周期边界保存；进程锁避免目录并发写入。每5000次更新对5000张验证图在 t=.1/.3/.5/.7/.9 分别记录四个周期的原始/加权MSE，导出固定初始噪声的N=8/16/32/64自由样图与未裁剪统计。训练日志 `seed0_gamma_only/train.jsonl`，状态 `seed0_gamma_only/status.json`，启动信息和源文件SHA256均有记录。

## 验证与评估

`verify.py` 额外检查非单位RMS输入的输出头严格等于 `linear((h*gamma)*(1+scale)+shift)`，及gamma可学习梯度；同时检查与原 attention TRM block 逐位一致、patch往返、JiT loss等价、单时间采样、56次no_grad/28次grad、周期detach、零头初始化后的主干梯度、无状态四周期输出、优化器精确重放和全部Euler步。记录在 `runs/cifar10_looped_flows_v3_jit/gamma_only_checks/verification.json`。`gamma_only_smoke768/` 用真实数据覆盖训练、checkpoint、EMA验证和采样链路。

```bash
.venv/bin/python cifar10_looped_flows_v3_jit/evaluate.py \
  --checkpoint runs/cifar10_looped_flows_v3_jit/seed0_gamma_only/ema_100000.pt \
  --output runs/cifar10_looped_flows_v3_jit/seed0_gamma_only/fid5k_N32 \
  --num-images 5000 --flow-steps 32 --batch-size 64 --compile --fid
```

FID 使用 torch-fidelity0.4.0，对官方train50000；5k趋势筛查与50k最终指标应分别注明。训练完成或loss下降不代表生成质量达标；本次尚无最终FID。

JiT head从以下固定版本适配，并移除了重复的输出 RMSNorm： `cbc743a2ada5e9762697da2c83f8c4f8379e8c17`，MIT许可证保留在 `reference/LICENSE`；原始参考源码仍在 `../cifar10_looped_flows_v2_jit/reference/`。
