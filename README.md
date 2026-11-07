# CS100709：提高地形通关率与原始奖励的自动训练

**交给 Windows 电脑上的 agent：读取本 README，复用已有环境，执行 `python train.py --mode campaign --campaign-dir runs/campaign-issue3-v4`，持续跟进到评测、打包完成。** 程序自动完成普通 PPO / gSDE PPO 两组训练、独立种子复核、奖励与通关候选筛选、后续训练和最终评测。启动前执行第 3 节的进程检查、自检和协议预览；无需原聊天记录。

公开仓库：[Catherine135/cs100709-bipedalwalker-a2c](https://github.com/Catherine135/cs100709-bipedalwalker-a2c)。教师提交物仍为单个 `train.py` 与 `a2c.model`。**本机 Mac 禁止训练，CLI 会拒绝；仅允许零更新自检、只读评测或 dry-run。** 新方案尚未训练，不能把代码检查通过称为达到 300 分。

## 1. 依据与 Issue #3 诊断

PDF 第 1 页允许引用开源算法，不指定某个必须复现的仓库，也不要求完全原创；第 1–3 页要求 Hardcore 机器人前进、越障、提高平均奖励并展示结果；第 4–6 页规定教师接口和提交格式；第 31–40 页给出 A2C 示例。保留默认 A2C 对照，主方案为有来源说明的 PPO 扩展。PDF 的“300+”是奖励描述，不是明确的统一及格线。

本轮起点是 [Issue #3](https://github.com/Catherine135/cs100709-bipedalwalker-a2c/issues/3)与 [Release `campaign-20261107-results`](https://github.com/Catherine135/cs100709-bipedalwalker-a2c/releases/tag/campaign-20261107-results)，已核对完整诊断与模型：

| 上一轮实测 | 结果 |
| --- | --- |
| 总训练步数 | 49,807,360；因旧平台期规则停止，未用完一亿预算 |
| 选中模型 | `continue-002`，gamma=0.999 |
| 开发筛选集 | 均分 184.85，0/20 到终点 |
| 最终确认集 | 均分 131.53，0/20 到终点 |
| 教师示例 1,3,5,7,9 | 均分 135.76，0/5 到终点 |
| 被旧筛选丢弃的候选 | `continue-004`：172.89、1/20 到终点；`continue-005`：179.43、1/20 到终点 |

不少未通关回合持续前进，但 1600 步仍未走完。例如 seed 3 前进比例约 95.3%，回报 247.37，力矩成本约 98.21；seed 7 是 440 步摔倒，其记录停滞比例为 0，不能归为长时间原地停滞。旧继续阶段反复从 `continue-002` 出发，没有延续出现通关能力的分支。上述是诊断事实；不能据此断言某一个学习率参数已被因果实验验证。

原奖励本身已扣除 `0.028 × Σ|裁剪后关节动作|` 的每步力矩成本。在典型起点到终点的前进项约 364 分时，300 分留给力矩和姿态项的余量约 64 分。因此需要同时提高行进效率、降低无效动作并减少摔倒，单纯延长训练不保证改善。这个计算只是诊断，不改变奖励。

程序只自动下载两个固定 Release 文件并验证 SHA-256：

```text
a2c.model
bc2d1ebca4624635d5395f4d9f80c22927564e64ece1ab544eef644cb9b474e1
final-best-checkpoint.pt
b8317a49372b867f956001a785661a33372eb4875dd617d60b41f7c5750bf731
```

已有这两个文件可分别传 `--campaign-baseline C:\actual\a2c.model` 和 `--campaign-checkpoint C:\actual\final-best-checkpoint.pt`；哈希必须一致。不下载整个旧诊断包，不重装 PyTorch。普通 PPO 从完整检查点保留优化器；gSDE 导入相同 Actor/Critic/归一化统计量，但为新的噪声参数重建优化器。

## 2. 新训练协议：地形覆盖后再提高效率

所有训练和评测仍是 `gym.make('BipedalWalker-v3', hardcore=True)`，Gymnasium 0.29.1、**1600 步**、24 维原始观测、4 维动作和原始奖励。没有奖励归一化、裁剪、额外力矩惩罚或普通模式预训练。训练内部使用旧模型的观测归一化统计量并冻结；归一化随导出模型保存。

| 设置 | A：普通高斯 PPO | B：gSDE PPO |
| --- | --- | --- |
| 起点 | Issue #3 最佳完整检查点 | 同一策略和价值权重；新建 Adam |
| gamma / GAE lambda | 0.999 / 0.95 | 相同 |
| 探索 | 每步独立高斯噪声 | 状态依赖噪声矩阵每 4 步重采样 |
| 网络 | Actor/Critic 各两层 256、Tanh | 相同；噪声参数为 256×4 |
| CPU / 环境 / 线程 | CPU / 16 / 1 | 相同 |
| rollout / minibatch / epochs | 2048/环境 / 256 / 最多 10 | 相同 |
| 学习率 / clip / target KL | 2.5e-5→2e-5 / 0.1 / 0.02 | 相同 |
| 梯度上限 / 熵系数 | 0.5 / 0.001→0.0002 | 相同 |
| 首次训练 / 独立复核种子 | 9873 / 10873 | 相同 |

gSDE 的初始噪声按固定归一化探针校准，初始确定性动作与原模型一致。只在训练采样时维持相关噪声，教师 `eval=True` 仍是无状态确定性动作。`eval=False` 的 portable Agent 提供边际高斯采样；正式 gSDE 训练采样由 SB3 完成。

**训练地形的有限重试：**每个 worker 每回合以 75% 概率生成新随机地形、25% 概率重试自身失败队列（空队列时全部生成新地形）。种子范围 `[1,000,000, 2^31)`，与全部评测种子隔离。最多保留 128 个失败种子，每个最多重试 3 次，成功则移除。每次都从原始起点开始；不缩短地形、不改变观测、奖励或时限。重试的是地形，仍重新采集当前策略轨迹，不是重复使用旧 PPO 样本。

自动流程如下：

1. 在新验证集和开发筛选集分别复测 Release 模型，建立本机基线。
2. A、B 各训练约 1000 万步，再以另一训练随机种子各做约 1000 万步。每约 25 万步在 20 个验证种子上评测。单阶段按分配预算完成，避免只通过验证集就反复零步停止；阶段间才根据两组开发评测决定是否提前结束整个任务。
3. 每阶段同时保存原始均分最佳候选，以及均分距该奖励候选不超过 25 分的通关率候选（已同时达到奖励与通关目标的候选不受此限制），两者都有对应的完整优化器检查点。阶段结束后，两者分别在 40 个开发种子上评测。
4. 保留平均奖励与通关率之间不被其他模型同时超越的候选（Pareto frontier）。继续训练按 `A 奖励 → B 奖励 → A 通关 → B 通关` 轮换，每段约 500 万步；不再仅因均分平台期在约 5000 万步停掉全部路线。候选的合并开发均分最多比 Release 基线低 25 分，避免靠严重牺牲奖励提高通关比例；无合格父模型则从相应 Release 初始化重新开始。
5. 当某个父模型在验证集与开发集**各自**达到至少 80% 通关率，后续该分支进入效率阶段：学习率向 1e-5 降低，熵系数向 0 降低，通过原始奖励已有的力矩成本优化效率。延续阶段不提高父检查点当时的学习率或熵系数。
6. 最终选择的合并开发均分必须不低于 Release 基线。未达到足够通关率前优先通关率；已有两组开发集均达到 80% 的候选时，在这些候选中优先原始均分。两组开发集都满足均分 300 与通关 80%，或总预算耗尽后，锁定模型再打开最终确认集。基线自身也保留为候选。
7. 最终测试无论成败都如实保存结果、打包并退出；不能根据最终测试重新选模型、继续训练或改参数。`goal_reached=false` 的正常完成仍返回退出码 0，安全中断返回 130，运行错误才是失败。

合并开发指标按 20 个验证回合＋40 个筛选回合加权，两个集合都参与开发，不能称为最终测试。80% 是本轮“大部分种子到达终点”的工程目标，不是声称 PDF 规定这个门槛；程序也单独报告严格的全部通关且均分 300。

总预算默认 **100,000,000 次新训练交互**，包含全部试训、复核和继续阶段，不是每组各一亿；向下取整后硬上限 **99,975,168**。每个 1000 万阶段为 **9,994,240** 步，500 万阶段为 **4,980,736** 步，最后阶段使用剩余完整 rollout。评测另计、不更新权重。仅首次启动前可调整 `--campaign-budget`、`--pilot-steps`、`--chunk-steps`；总预算至少覆盖四个初始阶段。不能训练中悄悄增加预算。

## 3. Windows agent：复用环境并开始任务

目标机已验证硬件：i9-14900HX、RTX 4080 Laptop 12 GB、约 32 GB 内存。上一轮实测选择 CPU + 16 环境 + 1 线程；探索方式和评测数量可能改变耗时，应报告本次实际时间，不承诺 24 核占满或固定完成时间。

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
$RlTaskDir = Join-Path $RlRepo 'runs\campaign-issue3-v4'
New-Item -ItemType Directory -Force $RlTaskDir | Out-Null
Invoke-RlPython -u train.py --mode campaign --campaign-dir $RlTaskDir 2>&1 | Tee-Object -FilePath "$RlTaskDir\campaign-console.log" -Append
```

这是新的 v2 协议，必须使用新的 `campaign-issue3-v4` 目录，保留上一轮目录。无需手动分别启动 A/B 或修改 gamma。使用 agent 可持续终端会话，不让工具短超时终止任务；记录 PID、创建时间、会话 ID、源码提交到交接记录。每 30–60 秒查看新增日志和进程，持续跟进至交付物完成，不能以“已经启动”结束任务。程序会在 Windows 任务运行期间申请系统保持唤醒并在退出时释放，不修改永久电源设置。

## 4. 暂停、恢复和预算记录

- Ctrl+C 请求在当前 rollout 更新、评测及保存后停止；也可创建 `$RlTaskDir\STOP` 文件。不要强制杀死子进程。
- 恢复时先确认旧父进程及训练子进程已经退出，然后**重跑同一条 campaign 命令，使用同一个目录**。程序清除此前 STOP 请求并恢复未完成阶段，跳过已完成阶段。
- `campaign.json` 固定源码的 LF 哈希、基线哈希、所有阶段及协议。源码/参数变化会被拒绝；恢复时使用原版本，或保留旧任务并明确新建实验目录。不要训练过程中 `git pull`。
- 父任务和每个训练阶段都有操作系统文件锁。父进程异常退出但子进程仍活着时，恢复会拒绝重复启动；等待子进程结束或用 STOP 请求其安全保存。不要凭残留 `.lock` 文件是否存在判断进程存活。
- 正常中断恢复网络、优化器、归一化统计量、RNG 和每个 worker 的失败地形队列；环境回合重新开始，不保证逐位连续重放。新训练阶段使用新的地形随机流，并重新建立失败队列。
- 异常断电/强杀可能丢弃检查点后正在进行的一个 rollout。恢复时最多额外预留一个 rollout 的预算，并在报告中单列 `lost_steps_upper_bound`；这不是声称精确测得的交互数。已确认完成的训练步数另列，预算上限仍受约束。
- 文件损坏、依赖或数值错误会停止并保留现场，agent 阅读阶段 `console.log` 后处理。最多两次有依据的环境/进程恢复尝试；不无限重试、不静默换奖励、换环境或增加预算。

每阶段 `checkpoint.pt` 为最新完整更新，`best-checkpoint.pt` 为奖励候选的**完整**状态，`coverage-checkpoint.pt` 为到终点比例候选的**完整**状态；分别对应 `a2c.model` 和 `coverage.model`。`a2c.model` 是 CPU 导出对象，不能冒充含优化器的检查点。不要按全盘最大步数从其他实验挑检查点，campaign 已管理来源。

## 5. 种子隔离和验收

| 用途 | 默认种子 | 何时使用 |
| --- | --- | --- |
| 训练地形 | `[1,000,000, 2^31)` 中随机抽取 | 新地形与有限失败重试 |
| 阶段验证 | 11001,11003,…,11039（20 个） | 阶段内选择两种候选 |
| 开发筛选 | 12001,12003,…,12079（40 个） | 阶段比较，明确参与开发 |
| 最终确认 | 13001,13003,…,13199（100 个） | `final-choice.json` 锁定后首次评测 |
| 教师示例 | 1,3,5,7,9 | 锁定后展示；已在历史开发中看过，不能称为新留出集 |

上一轮的确认种子已经用于诊断，本轮换一组。最终验收文件分开报告：

- `coverage_goal_reached`：验证、开发筛选、最终确认集**各自**至少 80% 到达真正终点。
- `reward_goal_reached`：这三个集合各自原始平均奖励至少 300。
- `goal_reached`：以上两项同时成立。
- `strict_300_all_finished`：三个集合都全部通关且均分至少 300。

100 个固定新种子提供更广泛的检查，但样本通关率不是对所有未知地形的保证。`time_limit` 不是通关，低力矩成本也可能只是停止行走，必须连同前进、失败原因和原始奖励一起看。完整诊断保留各奖励分项，重构误差应小于 `1e-4`。

## 6. 自动产物与 agent 收尾

| 路径（相对于任务目录） | 内容 |
| --- | --- |
| `campaign.json` | 固定协议、状态、阶段选择、预算上限、环境版本 |
| `stages/*/config.json` | 实际 gamma、探索分布、学习率/熵调度、地形重试与归一化设置 |
| `stages/*/console.log`、TensorBoard、`episodes.jsonl` | 真实训练过程与原始回合奖励 |
| `stages/*/progress.json` | 完整更新步数、真实 Adam 更新次数、epoch 计数、PID |
| `stages/*/evaluations.jsonl`、`selection-return.json`、`selection-coverage.json` | 验证和开发筛选的逐种子结果 |
| `stages/*/checkpoint.pt`、`best-checkpoint.pt`、`coverage-checkpoint.pt` | 最新、奖励候选、通关候选完整状态 |
| `stages/*/reward-curves.png` | 各阶段训练回报及验证曲线 |
| `final-best-checkpoint.pt` | 与最终导出模型对应的完整状态，供后续明确授权的续训使用 |
| `final-choice.json` | 查看最终测试前锁定的模型及 SHA-256 |
| `confirmation-evaluation.json`、`example-evaluation.json` | 最终完整回合结果 |
| `final-evaluation.json`、`RESULTS.md` | 奖励/通关/组合目标、停止原因、步数和完整成绩 |
| `dependencies.txt` | 目标机实际依赖 |
| `submission.zip` | 仅 `train.py` 和 `a2c.model` |
| `diagnostics.zip` | JSON、日志、曲线和依赖，不含大模型检查点 |

程序执行独立 Python 进程中的教师导入/载入检查，验证 CPU 导出、动作一致性，并打包。agent 收尾时仍需核对 `campaign.json` 状态为 `completed`、所有产物存在、ZIP 内容正确；汇报实际阶段选择、原始均分、通关率、每回合力矩成本、预算及三个目标字段。训练回合的 `episodes.jsonl` 还记录训练地形种子、是否重试、失败原因与力矩成本。还应补记目标机实际墙钟时间和中断情况。若存在 `failed` / `interrupted`，不得称训练已完成。

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

| 原始来源 | 本轮采用或保留的思路 | 适用范围 |
| --- | --- | --- |
| [gSDE 论文：Smooth Exploration for Robotic Reinforcement Learning](https://arxiv.org/abs/2005.05719)；[SB3 2.4.1 分布实现](https://github.com/DLR-RM/stable-baselines3/blob/v2.4.1/stable_baselines3/common/distributions.py) | 用已有 gSDE 实现替代其中一组的逐步独立高斯噪声，每 4 步重采样探索矩阵 | 论文支持探索平滑性的动机，没有证明本项目 1600 步 Hardcore 能达 300。本轮不增加依赖 |
| [LLNL PPO-and-Friends 结果](https://github.com/llnl/ppo_and_friends/blob/6e5dcbacb741d91c82ba72d440d8cbb9017f6301/baselines/README.md)及[训练源码](https://github.com/llnl/ppo_and_friends/blob/6e5dcbacb741d91c82ba72d440d8cbb9017f6301/baselines/gymnasium/bipedal_walker_hardcore.py) | 参考阶段训练与后期降低学习率；作者报告 Hardcore 100 回合均分约 313 | 原实现使用奖励裁剪/归一化和 2000 步注册环境，本项目不照搬这些改变，也不宣称复现其成绩 |
| [RL Zoo PPO 配置](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml) | 一亿步量级的预算与调度作为工程参考 | 增加预算只能增加学习机会，不能保证成功 |
| [dgriff777 A3C](https://github.com/dgriff777/a3c_continuous/tree/ef12d863a982019ace0608f2798947567230d715) | 查阅 LSTM 连续控制实现，保留为后续备选 | 教师接口没有回合重置通知；直接持有记忆会有跨回合串状态风险。原项目也修改训练奖励，本轮未加入 LSTM |
| [SB3-Contrib TQC](https://sb3-contrib.readthedocs.io/en/v2.4.0/modules/tqc.html)及[RL Zoo 配置](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/tqc.yml) | 查阅样本复用与 gSDE 的实现线索 | 官方列出的 Hardcore 200 万步、3 个训练种子结果为 228±18，未证明本课程时限下达到 300；本轮不加入新的离策略学习器和大型经验回放文件 |

失败地形的有限重试、双候选保存、预算分配是针对 Issue #3 设计的工程策略，并非声称复制上述论文的完整算法。比较两条路线也不是严格的单变量消融：普通 PPO 延续原 Adam，gSDE 的噪声参数形状变化需要重建 Adam，应同时报告这个差异。

实际更新算法为 [SB3 2.4.1 PPO](https://github.com/DLR-RM/stable-baselines3/blob/v2.4.1/stable_baselines3/ppo/ppo.py)，引用 [PPO](https://arxiv.org/abs/1707.06347) 与 [GAE](https://arxiv.org/abs/1506.02438)。原始奖励和时限见 [Gymnasium 0.29.1 环境](https://github.com/Farama-Foundation/Gymnasium/blob/v0.29.1/gymnasium/envs/box2d/bipedal_walker.py)与[注册源码](https://github.com/Farama-Foundation/Gymnasium/blob/v0.29.1/gymnasium/envs/__init__.py)。

## 8. 保留的单次训练与 A2C 对照

常规交接使用 campaign；以下仅为明确需要独立实验时的入口：

```powershell
Invoke-RlPython train.py --mode train --algorithm ppo --num-envs 16 --ppo-batch-size 256 --steps 5000000 --warm-start C:\actual\a2c.model --gamma 0.999 --log-dir runs/manual-ppo --model-path runs/manual-ppo/a2c.model
Invoke-RlPython train.py --mode train --algorithm a2c --steps 1000000 --log-dir runs/a2c-baseline --model-path runs/a2c-baseline/a2c.model
Invoke-RlPython train.py --mode train --algorithm a2c --return-steps 8 --max-grad-norm 0.5 --steps 1000000 --log-dir runs/a2c-nstep --model-path runs/a2c-nstep/a2c.model
```

单次 PPO 的 `--resume` 恢复原模型及其已保存的调度；不会因为命令行写了新 gamma 就偷偷改变旧模型。新阶段使用 `--init-checkpoint` 保留优化器，并显式设置新阶段调度；从导出 Agent 开始用 `--warm-start`。campaign 自动处理这些差异。单次训练仍可能向上完成最后一个 rollout，严格总预算仅由 campaign 管理。
