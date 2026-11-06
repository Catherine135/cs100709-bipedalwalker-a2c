# CS100709 BipedalWalker A2C

按课程文件实现的单文件优势演员—评论家算法（Advantage Actor-Critic，A2C）。源码、英文注释、安装说明和自检均在 `train.py` 中。

主要依据 `CS100709_Reinforcement_Learning_Project.pdf`：第 1–6 页的任务和提交接口、第 20–22 页的算法推导、第 31–40 页的连续动作示例。原文件鸣谢 Hongtu Zhou、Di Zhang、Hai Zhang（TiEV lab）。课程 PDF 不包含在此仓库中。

## Windows 安装

在新的 Miniconda 环境中运行：

```powershell
conda create -n cs100709 -c conda-forge python=3.11 box2d-py=2.3.8 pip -y
conda activate cs100709
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install numpy==1.26.4 gymnasium==0.29.1 pygame==2.6.1 tensorboard==2.20.0 matplotlib ipython
```

NumPy 使用 pip 安装，避免与 Conda 的 BLAS/OpenMP 运行库混用。Box2D 已由 Conda 安装，不需要再次安装 `gymnasium[box2d]`。CUDA 版 PyTorch 也可以在 CPU 上运行；只需 CPU 时，将安装地址末尾的 `cu126` 换成 `cpu`。

## 推荐运行顺序

默认配置为 CPU、8 个并行环境、1 个 PyTorch 计算线程、每批 64 个样本，适用于用户的 i9-14900HX / RTX 4080 Laptop 电脑。运行时保持插电，训练期间关闭渲染；GPU 是否更快由相同配置的实际测速决定。

```powershell
python train.py --mode check
python train.py --mode benchmark --device cpu --num-envs 8
python train.py --mode benchmark --device cuda --num-envs 8
python train.py --mode train --device cpu --num-envs 8 --steps 1000000
```

只有 `train` 模式执行优化器更新。`check` 和 `benchmark` 使用未训练网络，检查权重保持不变。测速包含环境采样和前向／反向计算，不包括优化器更新、评测和文件保存，因此不能直接预测完整训练时间。

`--steps` 和 `--batch-size` 必须能被 `--num-envs` 整除。训练步数是所有环境累计的交互次数。100 万步来自文档的起始设置，不保证 Hardcore 的最终成绩。

## 评测、恢复与输出

```powershell
python train.py --mode eval --model-path a2c.model --eval-seeds 1 3 5 7 9 --render
python train.py --mode train --resume runs/RUN/checkpoint.pt --steps 2000000
tensorboard --logdir runs
```

将 `RUN` 替换为实际运行目录。恢复时 `--steps` 表示累计目标步数；网络、优化器、随机数状态和训练计数会恢复，环境回合重新开始。按 Ctrl+C 后，程序完成当前更新、最终评测和保存。

每次运行创建独立目录，保存配置、TensorBoard 日志、逐种子评测记录、恢复检查点和该次运行最佳的 `a2c.model`。`--model-path` 指定的位置也会更新为最佳验证模型；如需保留多次运行的外部导出文件，可分别指定路径。

训练使用独立验证种子 `[101, 103, 105, 107, 109]`。示例评测种子 `[1, 3, 5, 7, 9]` 可以替换为教师现场指定的种子。评测统计所有完整回合的原始奖励总和及算术平均值。

## 课程接口与实现细节

教师代码可直接使用：

```python
import torch
from train import *

agent = torch.load("a2c.model")
action = agent.select_action(state, eval=True)
```

保留文件名 `train.py`，与模型放在教师评测代码可访问的位置。导出的模型含 CPU 张量，载入不需要训练机 GPU；导入模块时仅为本文件的 `Agent` 注册 PyTorch 受限反序列化类型。

- 环境精确使用 `BipedalWalker-v3, hardcore=True`，保留 1600 步时限。
- 两个独立的双层 128 单元网络，Mish 激活、Adam 优化器、可学习标准差的高斯策略。
- 一步 TD 目标和 Actor 的优势值停止梯度传播；四个动作分量的对数概率相加。
- 执行动作裁剪到 `[-1, 1]`，概率计算使用原始采样动作。
- 真正终止时不自举，时间截断时使用回合末观测继续自举。
- 训练和评测使用不同环境；不改动原始奖励，也不加入熵奖励或观测归一化。

## 验证范围

内置检查覆盖手算 TD/损失、梯度隔离、动作接口、自动重置和剩余批次、真实 Box2D 渲染、spawn 并行、独立进程的教师载入方式、五个完整评测回合、TensorBoard、检查点恢复和中断保存。

训练管理器的控制流程使用模拟更新测试；优化器的 `step()` 被禁止调用，并用合成的 Adam 状态验证恢复。因此这些检查不训练模型，也不证明收敛或奖励水平。真实 RTX 4080 CUDA 性能需要在目标机器上验证。

GitHub Actions 在 Windows CPU 上执行同一套非训练自检。运行状态见仓库的 Actions 页面。

兼容性参考：[Gymnasium 注册源码](https://github.com/Farama-Foundation/Gymnasium/blob/v0.29.1/gymnasium/envs/__init__.py)、[时间限制处理](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)、[PyTorch 模型载入](https://docs.pytorch.org/docs/2.8/generated/torch.load.html)、[A2C 的 CPU 建议](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html)。
