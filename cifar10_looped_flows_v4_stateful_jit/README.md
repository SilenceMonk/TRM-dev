# CIFAR-10 v4 / Stateful TRM-JiT

独立实现 `cifar10_looped_flows_v4_stateful_jit`，将 v3 的 attention TRM + RoPE + gamma-only JiT head 改为沿 flow 时间传递状态。对应训练协议标识 `v4_stateful_jit_sorted_logitnormal_all_times`。原 v3 的两个实验保持暂停，产物保留。

## 训练与采样

每条序列采一批目标图像 Y、一次标准高斯噪声 ε，以及每图独立的 K=4 个 `sigmoid(N(-.8,.8²))` 时间并升序排列。四个时间全部使用，因此将四个位置混合后的时间边际仍为原 JiT logit-normal 分布。排序位置各自的边际不同，日志中的最后一个时间点不能与 v3 随机单时间的训练loss直接比较。

```python
state = model.initial_state(batch)
noise = randn_like(target)                    # 整条序列只采一次
for t in sorted_logitnormal_times:            # K=4，每图各自排序
    x = t * target + (1-t) * noise            # 真实插值，不使用前一预测构造训练输入
    prediction, state = model(x, t, state)    # 每个时间点恰好一个TRM监督周期
    loss = mean(((prediction-target)/max(1-t,.05))**2)
    loss.backward()
    optimizer.step()
    update_ema()
    # model返回的h/l已detach，供下一时间点使用
```

每个时间点保留 H=3、L=6：前两轮no_grad、最后一轮反传；core调用14次无梯度、7次有梯度。每时间点单独loss、backward和参数更新，保留局部deep supervision。不同序列重新初始化h/l，不跨batch携带状态。

采样从噪声初始化RGB和h/l状态，每个ODE时间点运行一个监督周期，同时更新RGB流状态并将h/l传至下一时间点。FP32 Euler、速度分母 `max(1-t,.05)`、末步也执行完整Euler；积分中不裁剪，导出PNG时才裁剪。支持 `--reset-state` 作为独立采样消融，默认跨时间携带。

这种局部监督、共享噪声、时间递增及状态传递参照 [Thinking with Looped Flows §3 / Algorithms 1–2](https://arxiv.org/html/2609.11801v1#S3)。本实验保留JiT的RGB目标与时间分布：论文使用均匀时间、采K+1点并调用前K点；本实现采K个logit-normal时间并全部调用，以遵循“其余保持不变”。不声称完整复现论文配方，不引入ACT、Q-head、SDE或类别条件。

## 保持一致的配置

- 两层共享post-norm attention TRM、D512、8 heads、head_dim64、SwiGLU expansion4；64个patch4 tokens，Q/K上使用原TRM一维RoPE，theta10000。
- 输出头 `h*gamma → 时间shift/scale → RGB投影`，gamma512维、初值1；无额外输出RMSNorm。时间MLP、sqrt(512)输入缩放、零初始化modulation/输出投影、BF16主干/FP32 head保持一致。可训练参数7,654,448。
- CIFAR-10 train50000固定45000/5000划分，RGB[-1,1]，seed0，默认无增强；训练不读取类别。Adam-ATan2、LR1e-4、betas(.9,.95)、WD1、warmup2000后恒定LR、EMA.999。
- 总100000次局部时间点更新 = 25000条batch序列。batch768时76.8M次局部图像监督、19.2M次新图像抽样。每1000步checkpoint，每5000步验证与固定初始噪声的N=8/16/32/64采样。

与v3 Nsup=4的同更新数对比，训练core调用数及新图像抽样数相同，但训练时间分布和状态传递方向改变；每个ODE时间点由4个监督周期变为1个。与v3 Nsup=1相比，每个ODE时间点core调用数相同，但v3每次更新都采新目标/噪声，v4每四次更新共享一次抽样。报告结果应注明这些差异。

## 文件与运行

`model.py`为独立带状态去噪模块；`objective.py`定义排序时间、插值与JiT损失；`train.py`处理状态序列与保存；`evaluate.py`支持普通/EMA权重、stateful/reset采样和FID。独立数据加载、JiT头和MIT许可均保留在本目录。

```bash
bash cifar10_looped_flows_v4_stateful_jit/setup_env.sh
.venv/bin/python cifar10_looped_flows_v4_stateful_jit/verify.py
.venv/bin/python cifar10_looped_flows_v4_stateful_jit/benchmark.py --batch-size 768 --compile
# 不要与服务写同一目录
bash service/cifar10_looped_flows_v4_stateful_jit.sh
.venv/bin/supervisorctl -c runs/cifar10_looped_flows_v3_jit/supervisord.conf status
```

默认run：`runs/cifar10_looped_flows_v4_stateful_jit/seed0/`；只从本目录恢复，从随机初始化开始，不加载v3权重。checkpoint在完整4点序列边界保存，包含模型、EMA、优化器与CPU/CUDA RNG。锁阻止同目录并发训练，恢复会核对协议标识。服务名 `cifar10_looped_flows_v4_stateful_jit`，日志在本项目run根目录的 `service.log`。沿用现有独立supervisor守护进程，配置文件位于旧v3 run根目录。

验证固定t=.1/.3/.5/.7/.9，同一目标/噪声下分别记录carried/reset的原始与加权MSE；自由样图独立于验证图像，不是重建。生成质量不能只由训练loss或低噪声MSE判断。

```bash
.venv/bin/python cifar10_looped_flows_v4_stateful_jit/evaluate.py \
  --checkpoint runs/cifar10_looped_flows_v4_stateful_jit/seed0/latest.pt \
  --weights ema --output runs/cifar10_looped_flows_v4_stateful_jit/seed0/fid5k_ema_N32 \
  --num-images 5000 --flow-steps 32 --batch-size 512 --compile --fid
```

FID复用torch-fidelity0.4.0与官方train50000真实统计。FID-5k筛查和FID-50k最终结果应分别注明。

`verification.json`验证与v3单周期逐位一致、排序且全部使用的时间、共享噪声插值、无跨时间梯度、局部更新/优化器重放、状态影响预测，以及stateful/reset采样与末步Euler。`smoke768/`覆盖40步真实数据训练、checkpoint、EMA验证和全部采样网格。性能以该目录实测记录为准。

本机RTX A4000 16GB、batch768、core compile真实数据40步短测：后20步约0.708秒/更新，峰值13.47 GiB。短测与正式实验分目录保存；不代表生成质量。
