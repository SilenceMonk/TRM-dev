# CIFAR-10 Looped Flows：零初始化残差卷积版

这是独立可运行的卷积版实现，保留续训与从头训练两个实验路径。名称中的archive表示保留版本，不表示占位代码。当前实验均已停止；权重、数据与日志不上传Git。

在原TRM Mixer的RGB输出后增加：

```python
self.refine = nn.Conv2d(3, 3, kernel_size=3, padding=1)
nn.init.zeros_(self.refine.weight)
nn.init.zeros_(self.refine.bias)
# forward
image = unpatchify(self.output(h)).float()
prediction = image + self.refine(image)
```

新增84个参数，总计5,129,812；卷积FP32。其余主干、普通像素MSE、均匀时间排序、K4、R3/m6、状态传递、原版Adam-ATan2和EMA均沿用[基线](../cifar10_looped_flows/README.md)。零初始化只保证刚接入时输出一致，不限制训练后的残差增益。

## 从头训练

依赖与GPU架构要求见基线安装说明。在仓库根目录执行：

```bash
/venv/main/bin/python cifar10_looped_flows_conv_archive/verify.py \
  --output runs/cifar10_looped_flows/verification_conv.json
/venv/main/bin/python cifar10_looped_flows_conv_archive/train.py \
  --run runs/cifar10_looped_flows/conv_new_seed0 --compile
```

主干随机初始化，卷积零初始化，优化器从空状态开始，warmup重新计数。本机历史从头训练目录为runs/cifar10_looped_flows/seed0_conv_scratch/，默认训练参数也指向它；重建实验应使用新的--run路径。

## 从旧基线迁移

```bash
/venv/main/bin/python cifar10_looped_flows_conv_archive/migrate_refine.py \
  --source runs/cifar10_looped_flows/seed0/latest.pt \
  --run runs/cifar10_looped_flows/conv_migrated_new
/venv/main/bin/python cifar10_looped_flows_conv_archive/train.py \
  --run runs/cifar10_looped_flows/conv_migrated_new --compile --resume
```

迁移只接收原基线结构，补齐模型与EMA的零卷积参数，保留原参数的Adam-ATan2 moments/step和RNG，新参数优化器状态首次更新时创建；记录父checkpoint SHA256与切换步数。训练全局步数与warmup继续，不从0重新开始。本机历史迁移从29000步开始，目录为seed0_refine_from29000/，原基线产物保留。

verify_refine.py需要本机基线checkpoint，检查初始输出逐位一致、原参数首步更新逐位一致、卷积梯度及新参数优化器第1步。diagnose_refine.py是历史checkpoint诊断脚本，依赖本机指定的29000/45000步权重；普通训练不需要这些历史文件。

evaluate.py支持本版EMA和旧基线EMA（缺少卷积参数时补零以保留输出）。生成/FID命令与基线相同，只将脚本路径换成本目录，checkpoint指向卷积实验。

历史续训出现较强对比度与越界；仅局部去噪MSE下降不能说明生成改善。后续不带卷积的JiT式版本同样出现这一问题，不能据此把全部原因归到卷积头。各实验不报告未计算的FID。

service/cifar10_looped_flows_conv.sh/.conf提供停用autostart的独立服务模板；本次GitHub同步不启动任何实验。
