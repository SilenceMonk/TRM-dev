# CIFAR-10 Looped Flows：原始 TRM Mixer

无卷积基线，模型代码已恢复为首次实验版本（5,129,728参数）。实现依据根目录 [方法说明](../cifar10_looped_flows_trm.md)。本目录只对应基线；卷积版在 [cifar10_looped_flows_conv_archive](../cifar10_looped_flows_conv_archive/README.md)，JiT式版本在 [cifar10_looped_flows_v2_jit](../cifar10_looped_flows_v2_jit/README.md)。所有实验当前均未运行。

RGB32×32图像归一化到[-1,1]，切成64个不重叠4×4 patch；48→512投影乘sqrt(512)，加入1→512→512时间条件。共享两层post-norm Mixer，token/channel SwiGLU宽度256/1536。R=3、m=6，前两周期no_grad，最后周期完整反传。固定随机h/l buffers，无ACT/Q/类别条件。线性512→48输出并无损拼回RGB，无输出激活。

每条序列共享真实图与高斯噪声，逐样本采5个均匀时间并排序，使用前4个。各时间输入由同一真实图直接插值构造，只携带并detach隐状态；每局部时间点独立普通像素MSE、backward、optimizer.step、EMA update。采样为携带状态的Euler，速度(prediction-y)/(1-t)，末步直接返回prediction；仅保存图片时裁剪。

优化器为仓库原TRM vendored fused Adam-ATan2 0.0.3，LR=1e-4、betas=(.9,.95)、WD=1、warmup=2000后恒定LR、EMA=.999。初始配置seed0、batch640、预算100000次局部更新；本地原始基线仅训练至29000步保存后转入卷积实验，不声称基线已完成预算。

## 安装与运行

在仓库根目录执行。需支持当前GPU的PyTorch/torchvision CUDA wheel；本机使用/venv/main。优化器现有编译配置为sm120/C++20，其他架构应先适配其setup.py。

```bash
uv pip install --python /venv/main/bin/python -r cifar10_looped_flows/requirements.txt
uv pip install --python /venv/main/bin/python setuptools wheel setuptools-scm ninja
MAX_JOBS=3 uv pip install --python /venv/main/bin/python --no-build-isolation ./trm/vendor_optimizer
/venv/main/bin/python cifar10_looped_flows/verify.py
# 新实验请用新的run目录，已有checkpoint只在--resume时恢复。
/venv/main/bin/python cifar10_looped_flows/train.py --run runs/cifar10_looped_flows/new_baseline --compile
```

官方CIFAR-10训练集按固定NumPy default_rng(0)划分45000/5000，无类别条件、默认无增强；官方测试集不参与训练。下载MD5、RGB与划分SHA256记录在data/cifar10/manifest.json。数据可重建，不随代码上传。

默认基线目录runs/cifar10_looped_flows/seed0/。每1000步原子保存模型、EMA、完整优化器、CPU/CUDA RNG，只在K4序列边界checkpoint；每5000步验证并输出同噪声N8/16/32/64样图。目录有进程锁。verify.py检查patch顺序、梯度截断、参数梯度、优化器重放、Euler终点；benchmark.py测试稳态性能。

```bash
/venv/main/bin/python cifar10_looped_flows/evaluate.py \
  --checkpoint runs/cifar10_looped_flows/seed0/latest.pt \
  --output runs/cifar10_looped_flows/seed0/fid5k_N32 \
  --num-images 5000 --flow-steps 32 --compile --fid
```

FID使用torch-fidelity==0.4.0和官方训练集50000张真实参考。5000张与50000张生成结果需分别标注；耗时包含PNG I/O及可选编译。未计算的FID不报告为结果。完整数据、checkpoint、日志和样图留在本机runs/，不上传Git；/workspace非持久卷，recycle/destroy前需另行备份。

历史卷积辅助入口migrate_refine.py、verify_refine.py、diagnose_refine.py转发到卷积版目录，不使用本目录基线模型执行卷积实验。
