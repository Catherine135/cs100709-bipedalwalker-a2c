# CS100709：两组 PPO 方案的自动训练与 300 分验收

**交给 Windows 电脑上的 agent：读取本 README，复用已有 Python 环境，运行 `python train.py --mode campaign --campaign-dir runs/campaign-issue2-v3`，持续跟进到生成提交包和结果报告。** 程序自动完成两组试训、筛选、换种子复核、后续训练、有限回退、最终评测及打包；不需要原聊天记录。实际启动前按下文检查环境和已有进程。

公开仓库：[Catherine135/cs100709-bipedalwalker-a2c](https://github.com/Catherine135/cs100709-bipedalwalker-a2c)。最终源码仍只有 `train.py`，教师提交物为 `train.py` 和 `a2c.model`。**本机 Mac 禁止训练，CLI 会拒绝；Mac 只运行自检、只读评测或 `--dry-run`。** 300 是目标，当前新方案尚无实际训练成绩。完成预算不等于达到目标。

## 1. 依据和上一轮的真实结果

课程 PDF 第 1 页允许引用开源算法，没有指定必须复现某个 GitHub 仓库，也没有要求算法完全原创。第 1–3 页要求控制 Hardcore 机器人前进、越障、提高平均奖励并展示训练结果；第 4–6 页规定教师接口、模型保存方式和单源码提交；第 31–40 页为 A2C 教学示例。本项目保留默认 A2C 对照，主训练采用明确引用的 PPO 扩展。PDF 的“300+”是奖励描述，不是明确的统一及格线。

[Issue #2](https://github.com/Catherine135/cs100709-bipedalwalker-a2c/issues/2)与 [Release `ppo-20261107-results`](https://github.com/Catherine135/cs100709-bipedalwalker-a2c/releases/tag/ppo-20261107-results)是本轮的起点：

| 已核对证据 | 结果 |
| --- | --- |
| Windows 20,021,248 步 PPO 后的示例集均分 | +14.38，0/5 到终点 |
| 最佳验证均分及位置 | +71.31，4,063,232 步时 |
| 验证总数 | 154 次 × 5 回合，0 次到终点 |
| PPO 更新计数 | 611 次迭代；789 是 epoch 计数，Adam 实际更新 153,361 次 |
| KL 早停 | 603 次，其中 560 次在第一个 epoch 内 |
| 只读轨迹复测 | 存在长时间停滞并持续消耗力矩的行为；跨平台回报不完全一致 |

因此从该 Release 的**最佳导出模型**热启动，而不是直接续训表现退步的最后状态。程序自动下载约 0.6 MB 的 `a2c.model`，先校验 SHA-256：

```text
3fd5afb261316ef0b79212d96ea982b940bc4ef07987e573a914a2002ac36769
```

只下载模型，不下载或重装 PyTorch。可用 `--campaign-baseline C:\actual\path\a2c.model` 指定已有的同一模型；哈希必须相符。它没有最佳时刻的优化器状态，所以第一次热启动重建 Adam。后续阶段使用新版本保存的完整最佳检查点，保留相应优化器状态。

## 2. 自动执行的实验协议

两组均使用 `gym.make('BipedalWalker-v3', hardcore=True)`，Gymnasium 0.29.1，原始 **1600 步**时限、24 维观测、4 维动作、原始奖励。不能换成 2000 步的 `BipedalWalkerHardcore-v3`，不能把普通地形成绩算进来。无训练奖励塑形、奖励归一化或额外力矩惩罚；策略内部的观测归一化统计量随模型导出。诊断坐标不进入策略观测。

| 设置 | A：稳定更新 | B：长期收益 |
| --- | --- | --- |
| 初始策略 | 同一个 Issue #2 最佳策略 | 同一个 Issue #2 最佳策略 |
| 折扣因子 gamma | 0.99 | 0.999 |
| 价值网络 | 保留旧价值权重 | 重新初始化；先仅更新价值网络 8 个 rollout |
| 价值校准期间 | 无 | 策略权重及观测统计量冻结，之后恢复正常 PPO |
| Actor / Critic | 各两层 256、Tanh | 相同 |
| CPU / 环境 / PyTorch 线程 | CPU / 16 / 1 | 相同 |
| rollout / minibatch | 每环境 2048 步 / 256 样本 | 相同 |
| 学习率 | 1e-4 线性降至 2.5e-5 | 相同 |
| PPO 裁剪范围 | 0.2 线性降至 0.1 | 相同 |
| 最大 epoch / KL / 梯度范数 | 10 / 0.02 / 0.5 | 相同 |
| GAE lambda / 熵系数 | 0.95 / 0.001 | 相同 |
| 初始训练种子 | 543 | 543 |

这两组是待验证假设；大 gamma 不保证更好。公开成功实现也有较小 gamma，不能将视野短当作唯一病因。8 个价值校准 rollout 计入 B 的交互预算。

自动流程：

1. 在目标机复测旧模型，作为同平台筛选基准。
2. A、B 各进行约 500 万步试训。每约 25 万步在 20 个验证种子上评测，按原始平均回报保存最佳完整检查点。
3. 用另一组 20 个开发种子评测两组最佳模型。进入后续训练需较旧模型均分至少提高 10，并且通关率提高，或平均前进比例提高至少 0.05。这是预先定义的工程筛选阈值，不是统计显著性结论。
4. 若两组均未通过筛选，各允许一次约 500 万步的最佳检查点恢复尝试，降低学习率；仍无改善则如实停止，保留最佳可用模型。
5. 对合格配置按开发均分排序，用**训练种子 1543**从相同 Release 策略重新试训约 500 万步；验证改进能否在另一个训练种子重现。首选配置未通过则尝试另一个合格配置，每个 gamma 最多一次复核。
6. 选择通过复核的配置，继续其较好检查点，每段约 500 万步。每段后比较开发集，保留较好策略；学习率不因新阶段而回升。连续 3 段没有明显进展后，最多再做 2 段降低学习率的恢复尝试，然后停止。明显进展定义为均分增加至少 5，或均分更高且通关率提高。
7. 开发集达到平均 300 且全部通关、预算耗尽、或触发有限停止规则时，**先锁定模型，再打开最终确认集和教师示例集**。最终确认失败不会触发重新选模型或继续调参。
8. 自动保存原始评测、逐阶段曲线、依赖清单、`RESULTS.md`、`submission.zip` 与 `diagnostics.zip`，结束任务。

同时通过验证集和开发筛选集目标的模型优先；其余以原始平均回报为第一排序项，其次为通关率、前进中位数、停滞比例。阶段内也优先保留达到均分目标且全部通关的候选。停滞诊断定义为每 100 步窗口内最大前进距离小于 0.1，仅做诊断，不改变奖励。

默认总预算 **100,000,000 次新训练环境交互**，包括试训、复核、价值校准和所有回退，不是每一组各一亿。为避免超支，向下取整到完整 rollout（16×2048=32768）：实际硬上限 **99,975,168**，每个约 500 万步阶段为 **4,980,736**。评测交互单独进行，不属于训练预算。可在首次运行前调整 `--campaign-budget`、`--pilot-steps`、`--chunk-steps`，中途恢复必须保持协议一致。

## 3. Windows agent：复用环境并开始任务

目标机已验证硬件：i9-14900HX、RTX 4080 Laptop 12 GB、约 32 GB 内存。上一轮实测选择 CPU + 16 环境 + 1 线程；新批量大小可能改变耗时，应报告本次实际时间，不承诺 24 核占满或固定完成时间。

**先检查进程与文件，再更新代码。** 用 `Get-CimInstance Win32_Process` 查看 `python.exe` 的 `CommandLine` 和 `CreationDate`，检查是否已有本仓库的 campaign/训练进程；若存在，跟进该进程，不重复启动。已有仓库在无活跃训练、无未保存改动时执行 `git pull --ff-only`；不得用 `reset --hard` 覆盖工作。没有仓库时：

```powershell
git clone https://github.com/Catherine135/cs100709-bipedalwalker-a2c.git
Set-Location cs100709-bipedalwalker-a2c
```

没有 Git 可下载 [main.zip](https://github.com/Catherine135/cs100709-bipedalwalker-a2c/archive/refs/heads/main.zip) 解压。公开源码和 Release 均无需 GitHub 登录。

以下在仓库根目录的 Windows PowerShell 执行，后续命令共用这组变量：

```powershell
$ErrorActionPreference = 'Stop'
if ($env:OS -ne 'Windows_NT') { throw 'Training must run on the target Windows computer.' }
$RlRepo = (Get-Location).Path
if (-not (Test-Path '.\train.py')) { throw 'Change into the repository directory first.' }
$RlConda = $env:CONDA_EXE
if (-not $RlConda) { $RlConda = (Get-Command conda.exe -ErrorAction Stop).Source }
$RlPrefix = Join-Path $RlRepo '.conda'
function Invoke-RlConda {
    & $script:RlConda @args
    if ($LASTEXITCODE -ne 0) { throw "Conda command failed: $LASTEXITCODE" }
}
function Invoke-RlPython {
    Invoke-RlConda run --no-capture-output -p $script:RlPrefix python @args
}
```

GWTower 已有 `.conda` 环境。若目录不同，用 `conda env list --json` 和旧交接记录定位，修改 `$RlPrefix`；Conda 不在 PATH 则检查已有 Miniconda/Miniforge 的 `Scripts\conda.exe`。**路径不同不构成重新安装理由。**

先检查依赖实际可导入及版本：

```powershell
Invoke-RlPython -c "import sys,torch,numpy,gymnasium,Box2D,pygame,stable_baselines3,tensorboard,matplotlib; print(sys.executable); print(torch.__version__,numpy.__version__,gymnasium.__version__,Box2D.__version__,stable_baselines3.__version__)"
Invoke-RlPython -m pip check
```

基线：Python 3.11、PyTorch 2.8.0（CPU 或已安装 cu126）、NumPy 1.26.4、Gymnasium 0.29.1、box2d-py 2.3.8、pygame 2.6.1、TensorBoard 2.20.0、SB3 2.4.1、matplotlib。上一轮已经安装 SB3，依赖通过就跳过安装；只缺 SB3 时才执行 `python -m pip install stable-baselines3==2.4.1`。不得常规使用 `--upgrade` / `--force-reinstall`，无需 TensorFlow、torchvision 或另装 CUDA Toolkit。

仅确认没有任何可复用环境时，才执行首次安装：

```powershell
# First installation only; skip on GWTower's configured machine.
Invoke-RlConda create -p $RlPrefix --override-channels -c conda-forge python=3.11 box2d-py=2.3.8 pip -y
Invoke-RlPython -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
Invoke-RlPython -m pip install numpy==1.26.4 gymnasium==0.29.1 pygame==2.6.1 tensorboard==2.20.0 matplotlib ipython stable-baselines3==2.4.1
```

Conda 本身不存在时使用 [Miniforge 官方 Windows 安装说明](https://github.com/conda-forge/miniforge#windows)，核对安装器 SHA-256，用户级安装。Box2D 用 Conda 预编译包；不要再安装 `gymnasium[box2d]` 引入另一套 Box2D 编译依赖。

运行零更新自检和协议预览：

```powershell
Invoke-RlPython -B train.py --mode check --algorithm ppo --device cpu --num-envs 8
Invoke-RlPython -B train.py --mode campaign --dry-run
```

自检必须退出 0，并显示 PPO/campaign 检查通过及 `Optimizer steps: 0`。预览不下载模型、不创建训练目录、不训练。

**正式启动，只执行这一条训练入口：**

```powershell
$RlTaskDir = Join-Path $RlRepo 'runs\campaign-issue2-v3'
New-Item -ItemType Directory -Force $RlTaskDir | Out-Null
Invoke-RlPython -u train.py --mode campaign --campaign-dir $RlTaskDir 2>&1 | Tee-Object -FilePath "$RlTaskDir\campaign-console.log" -Append
```

无需手动分别启动 A/B 或修改 gamma。使用 agent 可持续终端会话，不让工具短超时终止任务；记录 PID、创建时间、会话 ID、源码提交到交接记录。每 30–60 秒查看新增日志和进程，持续跟进至交付物完成，不能以“已经启动”结束任务。程序会在 Windows 任务运行期间申请系统保持唤醒并在退出时释放，不修改永久电源设置。

## 4. 暂停、恢复和预算记录

- Ctrl+C 请求在当前 rollout 更新、评测及保存后停止；也可创建 `$RlTaskDir\STOP` 文件。不要强制杀死子进程。
- 恢复时先确认旧父进程及训练子进程已经退出，然后**重跑同一条 campaign 命令，使用同一个目录**。程序清除此前 STOP 请求并恢复未完成阶段，跳过已完成阶段。
- `campaign.json` 固定源码的 LF 哈希、基线哈希、所有阶段及协议。源码/参数变化会被拒绝；恢复时使用原版本，或保留旧任务并明确新建实验目录。不要训练过程中 `git pull`。
- 父任务和每个训练阶段都有操作系统文件锁。父进程异常退出但子进程仍活着时，恢复会拒绝重复启动；等待子进程结束或用 STOP 请求其安全保存。不要凭残留 `.lock` 文件是否存在判断进程存活。
- 正常中断恢复网络、优化器、归一化统计量和 RNG；环境回合重新开始，不保证逐位连续重放。
- 异常断电/强杀可能丢弃检查点后正在进行的一个 rollout。恢复时最多额外预留一个 rollout 的预算，并在报告中单列 `lost_steps_upper_bound`；这不是声称精确测得的交互数。已确认完成的训练步数另列，预算上限仍受约束。
- 文件损坏、依赖或数值错误会停止并保留现场，agent 阅读阶段 `console.log` 后处理。最多两次有依据的环境/进程恢复尝试；不无限重试、不静默换奖励、换环境或增加预算。

每阶段 `checkpoint.pt` 为最新完整更新，`best-checkpoint.pt` 为最佳验证时刻的**完整**状态。`a2c.model` 是 CPU 导出对象，不能冒充含优化器的检查点。不要按全盘最大步数从其他实验挑检查点，campaign 已管理来源。

## 5. 种子隔离和验收

| 用途 | 种子 | 何时使用 |
| --- | --- | --- |
| 阶段验证 | 101,103,…,139（20 个） | 阶段内选择最佳检查点 |
| 开发筛选 | 401,403,…,439（20 个） | 基线、A/B、复核、阶段比较；明确参与调参 |
| 最终确认 | 201,203,…,239（20 个） | 锁定最终模型之后 |
| 教师示例 | 1,3,5,7,9 | 锁定最终模型之后 |

只有选定模型的验证、开发筛选、最终确认和示例评测均满足 **原始平均奖励 ≥300 且全部到终点**，`final-evaluation.json` 才写 `goal_reached: true`。这比只看一次高分严格。最终测试失败仍保存并报告实际成绩，不用结果继续筛模型；教师未知种子的表现仍需现场验证。

`time_limit` 不是通关。检查诊断的 `reward_reconstruction_error` 绝对值小于 `1e-4`。第三方成绩、测试中的模拟成绩、普通模式成绩或归一化奖励不能当成本项目的实测成绩。

## 6. 自动产物与 agent 收尾

| 路径（相对于任务目录） | 内容 |
| --- | --- |
| `campaign.json` | 固定协议、状态、阶段选择、预算上限、环境版本 |
| `stages/*/config.json` | 实际生效的 gamma、学习率/裁剪调度、校准步数 |
| `stages/*/console.log`、TensorBoard、`episodes.jsonl` | 真实训练过程与原始回合奖励 |
| `stages/*/progress.json` | 完整更新步数、真实 Adam 更新次数、epoch 计数、PID |
| `stages/*/evaluations.jsonl`、`selection.json` | 验证和开发筛选的逐种子结果 |
| `stages/*/checkpoint.pt`、`best-checkpoint.pt` | 最新及最佳完整状态 |
| `stages/*/reward-curves.png` | 各阶段训练回报及验证曲线 |
| `final-choice.json` | 查看最终测试前锁定的模型及 SHA-256 |
| `confirmation-evaluation.json`、`example-evaluation.json` | 最终完整回合结果 |
| `final-evaluation.json`、`RESULTS.md` | 目标是否达到、停止原因、步数和完整成绩 |
| `dependencies.txt` | 目标机实际依赖 |
| `submission.zip` | 仅 `train.py` 和 `a2c.model` |
| `diagnostics.zip` | JSON、日志、曲线和依赖，不含大模型检查点 |

程序执行独立 Python 进程中的教师导入/载入检查，验证 CPU 导出、动作一致性，并打包。agent 收尾时仍需核对 `campaign.json` 状态为 `completed`、所有产物存在、ZIP 内容正确；汇报实际阶段选择、原始均分、通关率、预算及是否达到目标。还应补记目标机实际墙钟时间和中断情况。若存在 `failed` / `interrupted`，不得称训练已完成。

仅同步源码/README 不授权公开新训练模型和日志。此次任务自动生成本地产物；用户后续要求上传时再发布 Release。

教师原接口无需 SB3、GPU 或训练目录：

```python
import torch
from train import *
agent = torch.load('a2c.model')
action = agent.select_action(state, eval=True)
```

可独立复核（在课程要求的机器上以其结果为准）：

```powershell
Invoke-RlPython -B train.py --mode eval --device cpu --model-path "$RlTaskDir\a2c.model" --eval-seeds 1 3 5 7 9
# Optional visualization after training:
Invoke-RlPython -B train.py --mode eval --device cpu --model-path "$RlTaskDir\a2c.model" --eval-seeds 1 3 5 7 9 --render
```

## 7. 公开代码的参考范围与限制

此次进一步查阅了原作者代码及结果，不把“用了同一算法”当作复现成功：

| 项目 | 作者公开的结果/做法 | 本次取舍 |
| --- | --- | --- |
| [SB3 / RL Zoo](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml) | Hardcore 配置使用一亿步、学习率和 clip 衰减；[公开 PPO 模型卡](https://huggingface.co/sb3/ppo-BipedalWalkerHardcore-v3)为 48.61 ±120.68 | 借鉴调度和预算管理；明确该模型不是已验证的 300 分方案 |
| [LLNL PPO-and-Friends](https://github.com/llnl/ppo_and_friends/blob/6e5dcbacb741d91c82ba72d440d8cbb9017f6301/baselines/README.md) | 报告 Hardcore 100 回合均分约 313；[源码](https://github.com/llnl/ppo_and_friends/blob/6e5dcbacb741d91c82ba72d440d8cbb9017f6301/baselines/gymnasium/bipedal_walker_hardcore.py)讨论分阶段训练、降低后期学习率，使用较大 minibatch | 借鉴阶段化、稳定更新及充分测试；它使用奖励归一化/裁剪和另一注册时限，不能直接声称复现该分数 |
| [dgriff777 A3C](https://github.com/dgriff777/a3c_continuous/tree/ef12d863a982019ace0608f2798947567230d715) | 作者报告 100 回合平均 300+；LSTM 记忆、独立随机地形、多回合确认 | 借鉴复核思路；源码会把训练奖励乘 1.5 并裁剪，且需维护/重置循环状态，不直接套入现有无状态教师接口 |
| [RL Zoo TQC benchmark](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/benchmark.md) | 某个 Hardcore 200 万步训练模型的评测均分约 235，标准差约 111；作者说明不是多次训练的统计基准 | 作为后续算法研究线索；本轮不据此断言换 SAC/TD3/TQC 就能解决任务 |

本次没有复制这些外部仓库的模型权重或宣称其训练结果。依赖的实际 PPO 实现为 [SB3 2.4.1](https://github.com/DLR-RM/stable-baselines3/blob/v2.4.1/stable_baselines3/ppo/ppo.py)，算法引用 [PPO](https://arxiv.org/abs/1707.06347) 与 [GAE](https://arxiv.org/abs/1506.02438)。奖励和时限参考 [Gymnasium 0.29.1 环境](https://github.com/Farama-Foundation/Gymnasium/blob/v0.29.1/gymnasium/envs/box2d/bipedal_walker.py)、[注册定义](https://github.com/Farama-Foundation/Gymnasium/blob/v0.29.1/gymnasium/envs/__init__.py)。

## 8. 保留的单次训练与 A2C 对照

常规交接使用 campaign；以下仅为明确需要独立实验时的入口：

```powershell
Invoke-RlPython train.py --mode train --algorithm ppo --num-envs 16 --ppo-batch-size 256 --steps 5000000 --warm-start C:\actual\a2c.model --gamma 0.99 --log-dir runs/manual-ppo --model-path runs/manual-ppo/a2c.model
Invoke-RlPython train.py --mode train --algorithm a2c --steps 1000000 --log-dir runs/a2c-baseline --model-path runs/a2c-baseline/a2c.model
Invoke-RlPython train.py --mode train --algorithm a2c --return-steps 8 --max-grad-norm 0.5 --steps 1000000 --log-dir runs/a2c-nstep --model-path runs/a2c-nstep/a2c.model
```

单次 PPO 的 `--resume` 恢复原模型及其已保存的调度；不会因为命令行写了新 gamma 就偷偷改变旧模型。新阶段使用 `--init-checkpoint` 保留优化器，并显式设置新阶段调度；从导出 Agent 开始用 `--warm-start`。campaign 自动处理这些差异。单次训练仍可能向上完成最后一个 rollout，严格总预算仅由 campaign 管理。
