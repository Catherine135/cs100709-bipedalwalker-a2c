# CS100709：冲击 300 分的 Windows 训练交接

**当前主方案：PPO；保留原 A2C 作为对照。** 用户要求根据 Issue #1 的模型和完整诊断改进训练，尽可能达到原始平均奖励 300 分。接手 agent 应完成环境检查、训练、验证、恢复和交付，不需要原聊天记录或课程 PDF。只有 `train.py` 一个源码文件；最终提交仍为 `train.py` 加 `a2c.model`。

公开仓库：<https://github.com/Catherine135/cs100709-bipedalwalker-a2c>。链接不需要登录，但执行训练需要 agent 在目标 Windows 电脑上具有终端、联网及文件写入能力。只请求阅读／改代码不等于请求启动训练；用户的 Mac 不运行训练。

## 1. 旧模型究竟出了什么问题

[Issue #1](https://github.com/Catherine135/cs100709-bipedalwalker-a2c/issues/1)的 [Release](https://github.com/Catherine135/cs100709-bipedalwalker-a2c/releases/tag/handoff-20261107-results)提供了模型、检查点和完整日志。已确认导出模型与检查点的 `best_agent` 相同；100 万步、821 回合、15625 次更新均相符。

| 指标 | 旧模型的实际证据 |
| --- | --- |
| Windows 五种子原始平均奖励 | −87.350，标准差 1.277 |
| Mac 上同模型只读复测 | −87.269；跨平台物理仿真／浮点结果存在小幅差异 |
| 终止原因 | 5/5 均为时间截断；0/5 到达终点 |
| 最大前进比例 | 全程的约 0.52%～0.57% |
| 前进奖励 | 每回合约 +1.89～+2.09 |
| 力矩成本 | 每回合约 −86.36～−89.75 |
| 失败奖励覆盖调整 | 0；此次评测没有摔倒惩罚 |
| 确定性动作被裁剪的分量比例 | 约 26.4%～26.6% |
| 训练记录 | 821 个训练回合的奖励均低于 −101；Critic 损失有较大尖峰 |

它主要在起点附近持续施力，不能把“活到 1600 步”解释为学会跨越障碍。也不能从日志推断“24 核已打满”或断言梯度爆炸，因为旧日志没有相应测量。源码哈希 `88698c…` 与原版 `895d69…` 的差异仅为 Windows CRLF 与 LF 换行符，内容已核对。

PDF 第 1 页要求控制机器人前进、跨越障碍，尽量提高平均奖励并展示策略；允许有引用地使用开源算法。第 2 页的“300+”描述走到远端的奖励，不是满分上限，也不是明确规定的统一及格线。当前旧模型不能展示任务成功。本轮采用 **300 平均奖励加到达终点** 作为用户指定的工程目标，而非伪称课程评分规则。

## 2. 本轮方案与边界

- 环境保持 `gym.make('BipedalWalker-v3', hardcore=True)`，Gymnasium 0.29.1，24 维观测、4 维动作、原始 1600 步时限。不可替换成时限为 2000 步的 `BipedalWalkerHardcore-v3`。
- 保留原始环境奖励。不加前进奖励、不去掉能耗、不削弱摔倒惩罚，不使用奖励归一化。训练、验证、最终评测均为 Hardcore，不混入普通版成绩。
- 使用 **Stable-Baselines3 2.4.1 的近端策略优化（PPO）**：受限策略更新、广义优势估计（GAE）、优势标准化、梯度裁剪、KL 提前停止。多次使用一批数据，提高样本利用率。
- 策略内部使用观测标准化；这是对旧方案的明确扩展。环境输入输出仍为原始观测，统计量随模型导出，评测冻结；教师不必另装标准化包装器。诊断读取的坐标不输入策略。
- Actor／Critic 各两层 256 单元、Tanh；学习率 `2.5e-4`，`gamma=0.99`，`gae_lambda=0.95`，每个环境采集 2048 步，每批优化 64 个样本，10 个 epoch，裁剪范围 0.2，熵系数 0.001，最大梯度范数 0.5，目标 KL 0.02。
- 默认用 CPU、8 个 spawn 环境、1 个 PyTorch 线程。旧 A2C 的 CPU 测速不能直接视为 PPO 测速结果；报告实际 PPO 训练速度和墙钟时间。
- 一次任务先给 **20,000,000 步累计预算上限**，达到验证目标则提前结束。每个 PPO rollout 为 16384 步，完整 rollout 最多超出请求预算 16383 步，必须报告真实步数。不会承诺 100 万步或一次运行就达到 300。

依据：[PPO 原始论文](https://arxiv.org/abs/1707.06347)、[SB3 2.4.1 PPO 实现](https://github.com/DLR-RM/stable-baselines3/blob/v2.4.1/stable_baselines3/ppo/ppo.py)、[RL Zoo Hardcore 参数](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml)。Zoo 的 Hardcore 配置使用 1 亿步和不同注册时限，不能据此保证本项目 1600 步环境的成绩。这里采用其部分参数并明确保留本项目环境；不是复现 Zoo 的成绩。

## 3. 复用 Windows 已有环境，不重装 PyTorch

目标机是 i9-14900HX、RTX 4080 Laptop 12 GB、约 32 GB 内存。GWTower 已经创建项目 `.conda`，优先复用。先确认现有训练进程状态，避免同时启动两个训练；保留旧模型及日志。

若没有源码，用 `git clone https://github.com/Catherine135/cs100709-bipedalwalker-a2c.git`；没有 Git 可下载 [main.zip](https://github.com/Catherine135/cs100709-bipedalwalker-a2c/archive/refs/heads/main.zip) 解压。已有源码则在无活跃训练、无未保存修改时 `git pull --ff-only`，不使用 `reset --hard`。

以下命令在 **仓库根目录的 Windows PowerShell** 中执行。`$RlConda` 必须为实际 Conda 可执行文件；如果不在 PATH，查 `$env:CONDA_EXE`、已有 Miniconda／Miniforge 的 `Scripts\conda.exe`，不因此重装整个 Conda。

```powershell
$ErrorActionPreference = 'Stop'
if ($env:OS -ne 'Windows_NT') { throw 'Run training on the target Windows computer.' }
$RlRepo = (Get-Location).Path
if (-not (Test-Path '.\train.py')) { throw 'Change into the repository directory first.' }
$RlConda = $env:CONDA_EXE
if (-not $RlConda) { $RlConda = (Get-Command conda.exe -ErrorAction Stop).Source }
$RlPrefix = Join-Path $RlRepo '.conda'
function Invoke-RlConda {
    & $script:RlConda @args
    if ($LASTEXITCODE -ne 0) { throw "Conda command failed with exit code $LASTEXITCODE." }
}
function Invoke-RlPython {
    Invoke-RlConda run --no-capture-output -p $script:RlPrefix python @args
}
```

如果 `.conda` 不在当前仓库，通过 `conda env list --json` 和已有交接记录找到它，修改 `$RlPrefix`；不要凭路径不同重新安装。基线为 Python 3.11、torch 2.8.0、numpy 1.26.4、gymnasium 0.29.1、box2d-py 2.3.8、pygame 2.6.1、tensorboard 2.20.0，另需 matplotlib。

已有环境只增补 PPO 库及其缺失依赖：

```powershell
Invoke-RlPython -m pip install stable-baselines3==2.4.1
Invoke-RlPython -m pip check
```

不要加 `--upgrade`、`--force-reinstall`，也不要重新下载 CUDA 版 PyTorch。该 SB3 版本兼容已有的 Gymnasium 0.29.1 和 PyTorch 2.8.0。教师最终推理不需要 SB3，仅训练／恢复 PPO 时需要。

只有确认没有可复用环境时才创建新环境。未安装 Conda 时，按 [Miniforge 官方说明](https://github.com/conda-forge/miniforge#windows)下载安装器、核对 SHA-256，以用户级静默模式安装；选择可写、无空格的 ASCII 路径，参数 `/InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /S /D=<目录>`，`/D` 最后。然后执行：

```powershell
# First installation only; skip this block on GWTower's configured computer.
Invoke-RlConda create -p $RlPrefix --override-channels -c conda-forge python=3.11 box2d-py=2.3.8 pip -y
Invoke-RlPython -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
Invoke-RlPython -m pip install numpy==1.26.4 gymnasium==0.29.1 pygame==2.6.1 tensorboard==2.20.0 matplotlib stable-baselines3==2.4.1
Invoke-RlPython -m pip check
```

NumPy 使用 pip，Box2D 使用 Conda；不另装 `gymnasium[box2d]`，不使用 `KMP_DUPLICATE_LIB_OK` 掩盖运行库冲突。依赖故障先记录并修复，最多针对同一错误重试两次，不静默修改环境、奖励或算法参数。

## 4. 自检和新实验

先创建独立任务目录并做 **零优化器更新** 自检：

```powershell
$RlTaskDir = Join-Path $RlRepo ('runs\ppo-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 6))
New-Item -ItemType Directory -Force $RlTaskDir | Out-Null
$env:PYTHONUNBUFFERED = '1'
$env:SDL_VIDEODRIVER = 'dummy'
$env:SDL_AUDIODRIVER = 'dummy'
$env:MPLBACKEND = 'Agg'
Invoke-RlPython -B train.py --mode check --algorithm ppo --device cpu --num-envs 8 2>&1 | Tee-Object "$RlTaskDir\check.log"
Invoke-RlPython -m pip freeze | Out-File "$RlTaskDir\dependencies.txt" -Encoding utf8
Get-FileHash '.\train.py' -Algorithm SHA256 | Format-List | Out-File "$RlTaskDir\source-hash.txt"
```

检查退出码必须为 0，末尾包含 PPO 导出／载入／检查点检查通过及 `Optimizer steps: 0`。不使用 `--mode benchmark --algorithm ppo`，因为原 benchmark 测的是 A2C，一组错误测速不如明确记录实际训练时间。

**首次 PPO 训练**，不从旧 A2C 检查点恢复：

```powershell
$RlModel = Join-Path $RlTaskDir 'a2c.model'
Invoke-RlPython -u train.py --mode train --algorithm ppo --device cpu --num-envs 8 --torch-threads 1 --steps 20000000 --log-dir "$RlTaskDir\training" --model-path $RlModel 2>&1 | Tee-Object "$RlTaskDir\train-console.log"
```

使用 agent 可持续终端会话运行，不让工具的短超时杀死训练。记录启动时间、进程 PID 与创建时间、会话 ID、命令、源码版本、日志目录到 `$RlTaskDir\handoff.json`。每 30–60 秒检查新增日志和进程；不要因为进程启动就结束任务。电脑需要保持唤醒，agent 可在训练期间申请系统唤醒请求并在结束时释放，不全局修改电源方案。

程序每个完整 rollout 更新后保存检查点，约每 10 万步用验证种子 `[101,103,105,107,109]` 评测。验证期间进度暂时停止是正常现象。Ctrl+C 会请求完成当前 rollout 和更新，再评测、保存；不要强制杀进程。

## 5. 中断恢复：只选 PPO 检查点

先确认原进程已经退出。读取候选检查点的 `kind` 和 `steps`，选择本次 PPO 实验中步数最多且能读取的文件，不按修改时间猜，也不选 Issue #1 的旧 A2C 检查点：

```powershell
$RlScan = @'
import json
from pathlib import Path
import torch
valid, skipped = [], []
for p in Path('runs').rglob('checkpoint.pt'):
    try:
        c = torch.load(p, map_location='cpu', weights_only=True)
        if c.get('kind') != 'ppo_training_checkpoint':
            continue
        valid.append(dict(path=str(p.resolve()), steps=c['steps'], num_envs=c['config']['num_envs']))
    except Exception as error:
        skipped.append(dict(path=str(p.resolve()), error=repr(error)))
print(json.dumps(dict(candidates=sorted(valid, key=lambda x: x['steps'], reverse=True), skipped=skipped)))
'@
Invoke-RlPython -c $RlScan
```

同目录存在不同实验时，核对 checkpoint 的 `config` 和来源运行，不能跨实验挑最好进度。存在损坏文件时尝试其他有效备份；不要默默丢弃进度重跑。由 agent 将 `$RlCheckpoint` 设为核对后的绝对路径，并令环境数与记录一致：

```powershell
# Set these from the selected checkpoint, not from an old A2C run.
$RlCheckpoint = 'C:\path\to\ppo\checkpoint.pt'
$RlNumEnvs = 8
Invoke-RlPython -u train.py --mode train --algorithm ppo --device cpu --num-envs $RlNumEnvs --steps 20000000 --resume $RlCheckpoint --log-dir "$RlTaskDir\training" --model-path $RlModel 2>&1 | Tee-Object "$RlTaskDir\resume-console.log"
```

`--steps` 是累计预算，不是再训练这么多步。若已有步数略超过预算（完整 rollout），直接评测已有模型，或将恢复预算设为已有步数做零新增更新的最终导出。模型、优化器、观测统计量、随机状态及最佳模型会恢复；Box2D 回合重新开始，不承诺逐位连续重放。恢复目录独立，已有日志不删除。模型文件 `a2c.model` 不包含优化器，不能用于 `--resume`。

## 6. 怎样判断达到 300，而不是碰巧高分

训练期间只使用固定的验证集选模型。候选模型必须同时满足：

1. 5 个验证种子的原始平均奖励至少 300，全部到达终点。
2. 随后在另一组 20 个确认种子 `[201,203,...,239]` 上，原始平均奖励至少 300，全部到达终点。

满足这两个条件才提前停止；确认集会在候选达标时使用，因此它是额外验证集，不宣称从未使用过的最终测试集。最终另用教师示例种子 `[1,3,5,7,9]` 评测并报告，不用它挑选训练模型。三组种子必须互不重叠且各自无重复。

`final-evaluation.json` 分别记录 `validation_and_confirmation_passed` 和 `example_set_passed`。只有两者都通过，才报告“在本次指定种子上达到 300 并到达终点”，不保证教师现场所有未知种子。预算到期但未满足条件时，保存最佳模型，明确报告未达目标，不无限扩预算或伪造成功。

手动复核导出的最佳模型：

```powershell
Invoke-RlPython -B train.py --mode eval --device cpu --model-path $RlModel --eval-seeds 1 3 5 7 9 2>&1 | Tee-Object "$RlTaskDir\evaluation.log"
```

输出开头是环境信息，最后一个 JSON 才是评测结果；不要把整份日志直接当作一个 JSON。保留逐种子奖励、步长、`success_rate`、`diagnostics`、奖励分项和 `reward_reconstruction_error`。误差绝对值应小于 `1e-4`；`time_limit` 不是成功。较低分但通过确认集的候选可以优先于只在验证集得分更高的模型，导出路径因此代表已验证候选优先的选择规则。

## 7. 产物和交付

每个训练目录包含 `config.json`、TensorBoard 事件、`episodes.jsonl`（原始回合奖励）、`evaluations.jsonl`（验证／确认）、`checkpoint.pt`、最佳 `a2c.model`、`final-evaluation.json`。外部 `$RlModel` 是同一所选策略。

agent 应完成：

- 核对实际环境步数、模型文件、检查点可载入、模型参数有限、评测 JSON 和成功标志。低分正常保留，不把完成预算写成任务解决。
- 从原始回合记录及评测记录生成训练奖励和验证平均奖励 PNG；恢复实验保留旧曲线并标明恢复边界。
- 在独立 Python 进程执行教师接口，确认 CPU 载入、动作 `(4,)`、`float32`、范围合法及确定性。导出的 `Agent` 含观测统计量，不需要 SB3、GPU 或训练目录。
- 把本次实际使用的 `train.py` 和所选 `a2c.model` 放进 `$RlTaskDir\submission\`，记录 SHA-256，压缩为 `submission.zip`。检查点和依赖环境不放入提交包。
- 写 `RESULTS.md`，记录源码／依赖／硬件／算法与原 PDF 的差异、累计步数、时间、验证／确认／示例评测完整成绩、曲线、恢复与故障、模型及检查点路径。把 `handoff.json` 状态更新为真实的 `completed`、`interrupted` 或 `failed`。

只有产物与证据齐全才结束任务。不要自动上传新模型／日志到公开仓库；用户已上传的旧 Release 不代表后续实验自动公开。

教师接口：

```python
import torch
from train import *
agent = torch.load('a2c.model')
action = agent.select_action(state, eval=True)
```

## 8. 故障处理和 A2C 对照

CUDA 不可用时用 CPU，不要求重装驱动。内存／spawn 故障可在新实验中减少环境数；恢复 PPO 时需保持检查点的环境数。NaN／Inf、检查点损坏或重复报错时，保存错误和上一有效检查点，修复依赖后最多重试两次；不静默改变奖励函数或模型结构。模型长期低分时按真实诊断报告，不因用户想要 300 就把未达到的结果写成已达到。

原版 A2C 仍是 CLI 默认值，以下为对照入口，不是本轮主训练命令：

```powershell
Invoke-RlPython train.py --mode train --algorithm a2c --steps 1000000 --log-dir runs/a2c-baseline --model-path runs/a2c-baseline/a2c.model
Invoke-RlPython train.py --mode train --algorithm a2c --return-steps 8 --max-grad-norm 0.5 --steps 1000000 --log-dir runs/a2c-nstep --model-path runs/a2c-nstep/a2c.model
```

默认一步 A2C 源于 PDF 第 31–40 页；PPO 是 PDF 第 1 页允许的有引用开源算法扩展，不冒充讲义原实现。奖励解释、环境、接口及提交格式依据 PDF 第 1–6 页。兼容性参考：[环境源码](https://github.com/Farama-Foundation/Gymnasium/blob/v0.29.1/gymnasium/envs/box2d/bipedal_walker.py)、[时间截断](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)、[PyTorch 载入](https://docs.pytorch.org/docs/2.8/generated/torch.load.html)。
