# CS100709：针对慢步态和动作饱和的 PPO / CAPG 自动训练

**交给 Windows 电脑上的 agent：阅读本 README，复用已有 Python 环境，执行 `python train.py --mode campaign --campaign-dir runs/campaign-issue4-v5`，持续跟进至最终评测和打包。** 程序自动下载校验最新基线、完成两组试训与独立种子复核、选择后续方案、恢复中断并生成提交包。启动前先执行第 3 节的进程检查和零更新自检。

公开仓库：[Catherine135/cs100709-bipedalwalker-a2c](https://github.com/Catherine135/cs100709-bipedalwalker-a2c)。源码提交物仍只有 `train.py`，模型为 `a2c.model`。**本机 Mac 不训练；本轮新算法尚无真实训练成绩。** 检查通过不代表奖励已提高或已达到 300。

## 1. 最新证据：Issue #4 是中断结果，不是最终测试

课程 PDF 第 1 页允许引用开源算法，第 1–3 页要求 Hardcore 机器人前进、越障、提高平均奖励并展示结果，第 4–6 页规定教师接口与提交格式，第 31–40 页给出 A2C 教学示例。本项目保留 A2C，对后续 PPO/CAPG 扩展注明来源。“300+”是手册中的奖励描述，不能自行解释成统一及格线。

已读取 [Issue #4](https://github.com/Catherine135/cs100709-bipedalwalker-a2c/issues/4)及 [Release `campaign-v4-interrupted-20261108`](https://github.com/Catherine135/cs100709-bipedalwalker-a2c/releases/tag/campaign-v4-interrupted-20261108) 的模型、完整检查点与全部 13 个阶段诊断：

| 同一轮、同一平台的证据 | 结果 |
| --- | --- |
| 训练状态 | 84,377,600 步时安全中断；最终确认集尚未评测 |
| Release 模型 | `continue-007-gaussian-coverage` 阶段的**奖励候选**，gamma=0.999 |
| 该模型验证集 | 203.47，1/20 到终点 |
| 该模型开发筛选集 | 203.81，3/40 到终点；32 个超时，5 个摔倒 |
| 同一开发集上的旧基线 | 129.46，1/40 到终点 |
| 该阶段另一通关候选 | 验证 189.02、2/20；开发 204.66、4/40；未作为本次 Release 的模型上传 |
| 发布模型的动作诊断 | 约 27.1% 的确定性动作分量超出边界后被裁剪 |
| 已完成地形的回报 | 263.8–271.7；耗时 1521–1589 步，力矩成本约 90.3–96.0 |

因此主要问题是**走得不够快且成本高**，并非只要不摔倒就接近 300。上一轮等到 80% 通关后才进入效率阶段，实际一直未触发。gSDE 没有在本轮显示出稳定优势，继续固定分配一半后续预算缺乏依据。日志里的价值解释方差通常较高，KL 约 0.002–0.003，也不足以断言唯一问题是价值网络坏掉或 KL 早停。

以上是观察，不是严格的因果消融。尤其不能把 Issue #3 的 **184.85（另一开发集）** 和 Issue #4 的 **203.47（验证集）** 当作相同测试集的提升。阶段名称含 `coverage` 也不意味着其 `a2c.model` 就是通关候选。

原奖励已包含每步 `0.028 × Σ|裁剪后动作|` 的力矩成本。典型起点到终点前进项约 364 分，达到 300 需要大幅压缩累计动作/姿态成本，并保持通关。不能只减少力矩后原地不动；必须同时看原始回报、前进、长度与失败原因。

程序只下载下面两个文件，不重新下载 PyTorch，下载后核对 SHA-256，并核对导出模型与完整检查点的网络参数一致：

```text
final-best-a2c.model
8fd9ed360a38a95f287959daf859a0d683697b0b6d0dd127f0341874213a95c5
final-best-checkpoint.pt
20fbb89e90df4b0d4b92c52fb090f017c45b3a11ac1e26cc6b5e6fbd7c1cb5e5
```

可以用 `--campaign-baseline C:\actual\final-best-a2c.model` 和 `--campaign-checkpoint C:\actual\final-best-checkpoint.pt` 指向本地已有的同一文件；哈希必须相符。两个实验分支都保留该检查点的网络与 Adam 状态。新阶段重置阶段交互计数，不把历史训练步数算作本轮新增步数。

## 2. 新协议与算法依据

训练和评测仍使用 `gym.make('BipedalWalker-v3', hardcore=True)`、Gymnasium 0.29.1、**1600 步时限**、原始 24 维观测和奖励。没有奖励塑形、奖励裁剪/归一化、额外力矩惩罚、普通地形预训练或放宽时限。内部观测归一化沿用基线统计量并冻结；教师导出对象包含相同统计量。

新方法是**裁剪动作策略梯度（Clipped Action Policy Gradient, CAPG）**。来源为 [Fujita 与 Maeda，ICML 2018](https://proceedings.mlr.press/v80/fujita18a.html)及[作者的 PPO/TRPO 实现](https://github.com/pfnet-research/capg)。它利用动作会被裁剪这一事实，减少策略梯度估计的方差。原来使用原始高斯采样值的做法仍是有效对照；这次不是把它称为错误。

对每个动作维度，实际执行 `a=clip(u,-1,1)`。新分支对内部动作使用高斯密度，对 `-1` 使用左侧累积概率，对 `+1` 使用右侧尾概率，四个维度的对数概率相加：

```text
log p(a | s) = log Phi((-1-mu)/sigma)      if a = -1
              log Normal(a; mu, sigma)   if -1 < a < 1
              log Phi((mu-1)/sigma)      if a = +1
```

采集时保存的旧概率和 PPO 更新时的新概率都使用这个分布；只改其中一边会破坏概率比。实现使用 PyTorch 的稳定 `log_ndtr`，不是先算极小 CDF 再取 log。CAPG 分支把裁剪后的动作放入 rollout；普通 PPO 仍按 SB3 的原始采样动作计算概率。两者初始的**实际执行动作分布和确定性动作相同**，CAPG 不直接把关节控制量调小，也不改变奖励。

| 设置 | A：普通高斯 PPO | B：CAPG PPO |
| --- | --- | --- |
| 初始网络/优化器 | 同一 Issue #4 完整检查点 | 相同 |
| 更新概率 | 原始高斯采样概率 | 实际执行动作的密度/边界概率 |
| gamma / GAE lambda | 0.999 / **0.98** | 相同 |
| 熵奖励系数 | **0** | **0** |
| 网络 | Actor/Critic 各两层 256、Tanh | 相同 |
| CPU / 并行环境 / 线程 | CPU / 16 / 1 | 相同 |
| 每环境 rollout / minibatch | 2048 / 256 | 相同 |
| 学习率 | 2e-5 线性降至 1e-5 | 相同 |
| PPO clip / 最大 epoch / target KL / 梯度上限 | 0.1 / 10 / 0.02 / 0.5 | 相同 |
| 试训 / 独立复核种子 | 19873 / 20873 | 相同 |

CAPG 诊断使用裁剪后混合分布的熵，不应与原始高斯熵直接比较。两组从开始就不再额外奖励高熵，不等于取消随机探索：基线的可学习高斯标准差仍保留。GAE lambda 从 0.95 提高到 0.98，使 `1/(1-gamma*lambda)` 所表示的优势权重衰减尺度从约 20 步增加至约 48 步；这是改善跨步态信用分配的假设，也可能增加方差。它与熵系数的调整是两组共有设置，因此本轮只直接比较同一新设置下的普通概率与 CAPG 概率；不能单独归因于 λ 或熵。

训练地形沿用有限失败重试：每回合 75% 概率生成新随机地形、25% 重试失败队列（空队列则全部新地形）。训练种子在 `[1,000,000, 2^31)`，最多保存 128 个失败种子，每次入队最多重试 3 次，成功即移除。每次从原始起点重新采集当前策略轨迹；不是复用旧 PPO 样本。诊断额外报告超时率、摔倒率、动作饱和比例和已通关回合平均奖励。

自动执行顺序：

1. 在本轮新验证集、新开发集分别复测 Release 基线。
2. 两种方法各试训约 500 万步，再用独立训练种子各复核约 500 万步，共四个初始阶段。每约 25 万步验证一次；每阶段保留奖励/通关两种完整候选，再在 40 个开发种子上筛选。
3. 根据每个方法两个训练种子的结果分配后续预算。先比较达到“基线合并均分减 25 分”底线的训练种子数量，再比较两个种子的平均开发均分和通关率。若没有方法的两次训练都满足该回报底线，停止继续分配；保留实际最佳模型并完成独立评测。这个规则是工程筛选，不是统计显著性结论。
4. 后续仅继续胜出方法，奖励候选与通关候选交替作为父模型，每段约 500 万步。完整检查点保留相应优化器；学习率不会因进入新阶段提高。失败方法的初始结果仍保留，明确写入 `allocation_decision`，不再固定给弱分支一半预算。
5. 相对上次有意义进展的记录，开发均分提高至少 5 分，或通关率提高至少 5 个百分点，才重置平台期计数。**连续 3 段无此改善就停止**，不反复从同一旧模型无限续训。所有继续候选仍受回报底线约束。
6. 最终模型的合并开发均分不得低于 Release 基线。先改善通关比例；已有验证集与开发集各自达到 80% 的候选时，在这些候选中优先原始均分。两组都达到均分 300 与通关 80%、预算耗尽或触发停止规则时，先锁定模型，再打开 100 个最终确认种子。最终测试失败不会触发重新筛选。

奖励候选按验证原始均分保存；通关候选通常要求距当前奖励候选不超过 25 分，已满足两项目标者不受该差值限制。比较阶段时保留奖励/通关的非支配候选，20 个验证回合和 40 个开发回合按回合数加权。最终的 80% 与 300 目标分别检查各个集合，不能以合并平均掩盖某一集合失败。

默认最多 **60,000,000 次新训练交互**，按 32768 步完整 rollout 取整后硬上限 **59,998,208**；每个约 500 万步阶段为 **4,980,736**。评测不更新权重，单独计数。预算是上限，可能因达标、未通过两种子回报底线筛选或平台期提前结束。只在首次启动前调整 `--campaign-budget`、`--pilot-steps`、`--chunk-steps`，不能训练途中扩大预算。

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
$RlTaskDir = Join-Path $RlRepo 'runs\campaign-issue4-v5'
New-Item -ItemType Directory -Force $RlTaskDir | Out-Null
Invoke-RlPython -u train.py --mode campaign --campaign-dir $RlTaskDir 2>&1 | Tee-Object -FilePath "$RlTaskDir\campaign-console.log" -Append
```

这是新的 v3 协议，必须使用新的 `campaign-issue4-v5` 目录。保留旧的 `runs/campaign-issue3-v4` 和 commit `13fb52b`；旧任务若要恢复，只能用其旧源码。新任务无需手动启动 A/B 或修改参数。使用 agent 可持续终端会话，不让工具短超时终止任务；记录 PID、创建时间、会话 ID、源码提交到交接记录。每 30–60 秒查看新增日志和进程，持续跟进至交付物完成，不能以“已经启动”结束任务。程序会在 Windows 任务运行期间申请系统保持唤醒并在退出时释放，不修改永久电源设置。

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
| 阶段验证 | 21001,21003,…,21039（20 个） | 阶段内选择两种候选 |
| 开发筛选 | 22001,22003,…,22079（40 个） | 阶段比较，明确参与开发 |
| 最终确认 | 23001,23003,…,23199（100 个） | `final-choice.json` 锁定后首次评测 |
| 教师示例 | 1,3,5,7,9 | 锁定后展示；已在历史开发中看过，不能称为新留出集 |

Issue #4 在最终确认前中断；不要擅自打开其旧确认集来调参。本轮预先固定新的种子组。最终验收文件分开报告：

- `coverage_goal_reached`：验证、开发筛选、最终确认集**各自**至少 80% 到达真正终点。
- `reward_goal_reached`：这三个集合各自原始平均奖励至少 300。
- `goal_reached`：以上两项同时成立。
- `strict_300_all_finished`：三个集合都全部通关且均分至少 300。

100 个固定新种子提供更广泛的检查，但样本通关率不是对所有未知地形的保证。`time_limit` 不是通关，低力矩成本也可能只是停止行走，必须连同前进、失败原因和原始奖励一起看。完整诊断保留各奖励分项，重构误差应小于 `1e-4`。

## 6. 自动产物与 agent 收尾

| 路径（相对于任务目录） | 内容 |
| --- | --- |
| `campaign.json` | 固定协议、状态、两训练种子的 allocation_decision、阶段选择和预算上限 |
| `stages/*/config.json` | 实际 gamma、GAE lambda、CAPG 开关、学习率/熵调度及环境设置 |
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

## 7. 本轮检索的公开资料及取舍

| 原始来源 | 查到的依据 | 本轮取舍 |
| --- | --- | --- |
| [CAPG 论文](https://proceedings.mlr.press/v80/fujita18a.html)及[作者实现](https://github.com/pfnet-research/capg) | 对裁剪动作的策略梯度给出无偏、较低方差分析；公开代码包含 PPO/TRPO | 针对约 27% 动作饱和实现 PyTorch/SB3 适配。没有把论文结论当作本项目 300 分保证，也没有照搬旧 Chainer 依赖 |
| [GAE 原论文](https://arxiv.org/abs/1506.02438) | λ 调节优势估计的偏差/方差与时间范围 | 在两组共同测试 λ=0.98；保留单次训练参数可做后续有预算的消融 |
| [SB3 2.4.1 PPO](https://github.com/DLR-RM/stable-baselines3/blob/v2.4.1/stable_baselines3/ppo/ppo.py) | PPO 裁剪目标、GAE、终止处理和恢复机制 | 沿用已有实现，仅适配概率分布和调度；不重装库 |
| [FORK 作者 Hardcore 实现](https://github.com/honghaow/FORK/tree/master/BipedalWalkerHardcore) | 查阅了完整 notebook；训练把 -100 改成 -5，其他奖励乘 5，并使用 Hardcore 专用注册环境 | 不把它的高分当作本任务可直接复现的成绩，本轮不采用这些奖励变更 |
| [SB3-Contrib TQC](https://sb3-contrib.readthedocs.io/en/v2.4.0/modules/tqc.html)及[RL Zoo 配置](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/tqc.yml) | 官方 Hardcore 200 万步、3 种子结果为 228±18；使用另一注册时限 | 有样本效率研究价值，但没有证明当前 1600 步任务可达 300；本轮优先复用已学会行走的模型 |
| [LLNL PPO-and-Friends 源码](https://github.com/llnl/ppo_and_friends/blob/6e5dcbacb741d91c82ba72d440d8cbb9017f6301/baselines/gymnasium/bipedal_walker_hardcore.py) | 分阶段学习率与训练奖励裁剪 | 保留有依据的阶段调度，不复制奖励变更 |

原始奖励及 1600 步限制以 [Gymnasium 0.29.1 环境源码](https://github.com/Farama-Foundation/Gymnasium/blob/v0.29.1/gymnasium/envs/box2d/bipedal_walker.py)和[注册定义](https://github.com/Farama-Foundation/Gymnasium/blob/v0.29.1/gymnasium/envs/__init__.py)为准。gSDE 及旧模型载入支持仍保留，但本轮自动 campaign 不再运行 gSDE。LSTM 仍未引入，因为现有教师接口没有回合重置通知。

## 8. 保留的单次训练与 A2C 对照

常规交接使用 campaign；以下仅为明确需要独立实验时的入口：

```powershell
Invoke-RlPython train.py --mode train --algorithm ppo --num-envs 16 --ppo-batch-size 256 --steps 5000000 --warm-start C:\actual\a2c.model --gamma 0.999 --log-dir runs/manual-ppo --model-path runs/manual-ppo/a2c.model
Invoke-RlPython train.py --mode train --algorithm a2c --steps 1000000 --log-dir runs/a2c-baseline --model-path runs/a2c-baseline/a2c.model
Invoke-RlPython train.py --mode train --algorithm a2c --return-steps 8 --max-grad-norm 0.5 --steps 1000000 --log-dir runs/a2c-nstep --model-path runs/a2c-nstep/a2c.model
```

单次 PPO 的 `--resume` 恢复原模型及其已保存的调度；不会因为命令行写了新 gamma 就偷偷改变旧模型。新阶段使用 `--init-checkpoint` 保留优化器，并显式设置新阶段调度；从导出 Agent 开始用 `--warm-start`。campaign 自动处理这些差异。单次训练仍可能向上完成最后一个 rollout，严格总预算仅由 campaign 管理。
