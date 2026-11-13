# Nsup=1 与 Nsup=4 对照

当前对照仅将监督周期数从4改为1。保留 attention + RoPE、gamma-only输出头、H_cycles=3、L_cycles=6、两层共享模块、D512、patch4；保留周期内前两轮no_grad、第三轮反传。Nsup=1没有跨监督周期的deep supervision，但仍有周期内递归。因此本实验衡量多个监督周期及其训练方式的收益，不能单独回答周期内H/L递归是否有用。

| 配置 | Nsup=4 基线 | Nsup=1 对照 |
| --- | --- | --- |
| run | `seed0_gamma_only` | `seed0_gamma_only_nsup1` |
| 当前安排 | 第98288次更新暂停、checkpoint保留 | 从头训练100000次更新 |
| 随机初始化 | seed0 | seed0；参数和buffer逐位一致已检查 |
| batch | 768 | 768 |
| 每次新输入对应优化器更新 | 4次，周期间detach状态 | 1次 |
| 每个监督周期core调用 | 14次no_grad + 7次grad | 相同 |
| 每个ODE时间点的core调用 | 84次 | 21次 |
| 验证/采样 | 每5000次更新；5000张验证图；固定采样噪声 | 相同 |

Adam-ATan2、LR=1e-4、betas=(.9,.95)、WD=1、warmup2000后恒定LR、EMA=.999、时间分布、噪声强度、数据划分、增强和编译设置均相同。训练/验证/独立评估都使用checkpoint中记录的Nsup，恢复时检查Nsup一致。

比较口径：

- **同优化器更新数/近似同训练计算量**：对比共同的5000至95000步checkpoint。两组每次更新均为21次core调用，其中7次有梯度。Nsup=1在此口径下有4倍的新图像/噪声/时间抽样，结果同时包含输入复用策略的影响。
- **同新输入抽样数**：例如Nsup=1的20000步对Nsup=4的80000步，两者均抽样15.36M张图像；基线的优化器更新量和训练计算量为4倍。也应注明每次独立去噪调用的推理计算量不同。
- 逐周期原始/加权MSE与固定初始噪声的N=8/16/32/64自由采样共同判断。只比较训练loss不足以判断生成质量；未计算FID时不宣称FID改善。

```bash
.venv/bin/python cifar10_looped_flows_v3_jit/verify.py --nsup 1
.venv/bin/python cifar10_looped_flows_v3_jit/verify.py --nsup 4
# 手动运行时不要与服务写同一目录
bash service/cifar10_looped_flows_v3_jit_nsup1.sh
.venv/bin/supervisorctl -c runs/cifar10_looped_flows_v3_jit/supervisord.conf status
```

新服务为 `cifar10_looped_flows_v3_jit_nsup1`，原服务关闭autostart并保持停止。新日志 `runs/cifar10_looped_flows_v3_jit/service_nsup1.log`；状态为 `runs/cifar10_looped_flows_v3_jit/seed0_gamma_only_nsup1/status.json`。首次5000步预计约一小时，具体纽约时间以启动后的实测进度为准。

`nsup1_checks/verification.json`记录单周期调用数、梯度、采样和优化器精确重放；`initialization_comparison.json`记录seed0逐位一致初始化。`nsup1_smoke768/`为40次真实数据更新、验证和采样闭环。原Nsup=4的验证/样图与checkpoint保留。
