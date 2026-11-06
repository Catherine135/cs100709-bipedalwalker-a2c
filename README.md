# CS100709 BipedalWalker A2C：Windows agent 训练交接

**接手任务：在用户的 Windows 电脑上，自行完成环境准备、自检、CPU／CUDA 测速、累计 1,000,000 步训练、模型评测和结果整理。** 本 README 是完整交接入口，不需要原聊天记录、课程 PDF 或另行询问超参数。训练源码只有 `train.py`，带有注释和内置自检。

公开仓库：<https://github.com/Catherine135/cs100709-bipedalwalker-a2c>。读取、克隆和下载不需要 GitHub 登录。链接本身不会执行程序；接手 agent 必须能在目标电脑上使用终端、联网下载依赖、写入工作目录和维持长时间进程。

## 1. 接手 agent 的执行约定

当用户要求按本仓库开始训练时，按本文顺序执行到验收完成，不要只复述命令或在常规步骤前反复请求确认。若用户只要求阅读、修改代码或检查，则不因此启动训练。

- 目标电脑：Windows x64，i9-14900HX，24 核／32 线程，RTX 4080 Laptop，12 GB 专用显存，约 32 GB 内存。启动时检测实际环境，不必再询问这些已知配置。
- **用户的 Mac 只用于开发和非训练验证，不在 Mac 上运行模型训练。** 如果当前系统不是 Windows，说明平台不符，不把任务默认为在 Mac 上训练。
- 常规依赖安装、项目目录创建、检查、测速、既定预算内的训练和恢复可以直接进行。优先使用用户级环境，无需管理员安装或系统重启。CUDA 不可用时使用已验证的 CPU 路径继续。
- 初始预算固定为累计 **1,000,000 次环境交互**，不是每个环境 100 万步。中断恢复也包含在预算中。达到预算后评测并交付，不因成绩低而无限延长训练、搜索超参数或更换算法。
- 每次开始前检查是否已有本任务的训练进程及检查点。已有进程继续监看；已有有效检查点则恢复；已有完整结果则复核并交付。不要重复开多个训练进程或覆盖以往结果。
- 日志、错误及训练产物保留在本机。默认不把模型、日志或课程 PDF 上传到公开仓库。
- 常见故障按第 8 节修复，同一原因最多重试两次。无法自行解决时报告具体错误、已保存进度及所需外部操作，不把失败当作完成。

## 2. 固定的任务与算法

采用优势演员—评论家算法（Advantage Actor-Critic，A2C），主要依据课程文件 `CS100709_Reinforcement_Learning_Project.pdf`：第 1–6 页为任务和评测接口，第 20–22 页为算法推导，第 31–40 页为连续动作示例。文件鸣谢 Hongtu Zhou、Di Zhang、Hai Zhang（TiEV lab）。下面的信息足以运行，不需要另外取得 PDF。

| 项目 | 本次执行值 |
| --- | --- |
| 环境 | `gym.make("BipedalWalker-v3", hardcore=True)` |
| 观测／动作 | 24 维原始观测；4 维动作，每维 `[-1, 1]` |
| 回合时限 | Gymnasium 0.29.1 注册的 1600 步 |
| Actor／Critic | 两个独立网络；各两层 128 单元；Mish |
| 策略 | 可学习标准差的高斯策略；执行时裁剪动作 |
| 更新 | 一步时序差分（Temporal Difference，TD）；每批 64 个样本 |
| 折扣／学习率 | `gamma=0.99`；Actor `4e-4`，Critic `4e-3`；Adam |
| 采样／线程 | 默认 8 个 `spawn` 环境；1 个 PyTorch 计算线程 |
| 随机种子 | 训练 543；各环境使用相应偏移 |
| 验证 | `[101, 103, 105, 107, 109]`；每完成 10 个训练回合评测 |
| 检查点 | 每完成 1 个训练回合后的更新边界，以及评测／结束时保存 |
| 最终示例评测 | `[1, 3, 5, 7, 9]`；每个种子跑一个完整回合 |

TD 目标为 `r + gamma * (1 - terminated) * V(next_state)`，目标值停止梯度传播。Actor 使用四维动作对数概率之和乘以停止梯度传播的优势。概率计算使用原始采样动作；环境接收裁剪后的动作。真正终止不自举，时间截断使用真实回合末观测自举，自动重置后的初始观测不能替代回合末观测。

保留原始奖励和观测，不添加奖励塑形、观测归一化或熵奖励，也不替换为 PPO。`BipedalWalkerHardcore-v3` 的注册时限不同，不可直接替换环境名称。训练、测速和无人值守评测均不打开图形窗口。

**预算完成不代表已解决 Hardcore。** 文件中的“300+”不是本仓库已达到的成绩；最终报告只能填写实际奖励。

## 3. 获取源码并准备目录

以下命令使用 Windows PowerShell，不是 Bash。同一任务保留 `$RlRepo`、`$RlConda`、`$RlPrefix`、`$RlTaskDir` 等变量；新终端从交接记录恢复这些绝对路径。

首次获取源码：

```powershell
$ErrorActionPreference = 'Stop'
if ($env:OS -ne 'Windows_NT') { throw 'Training must run on the target Windows computer.' }
$RlRepo = Join-Path $env:USERPROFILE 'projects\cs100709-bipedalwalker-a2c'
if (-not (Test-Path $RlRepo)) {
    New-Item -ItemType Directory -Force (Split-Path $RlRepo) | Out-Null
    if (Get-Command git -ErrorAction SilentlyContinue) {
        git clone https://github.com/Catherine135/cs100709-bipedalwalker-a2c.git $RlRepo
        if ($LASTEXITCODE -ne 0) { throw 'Repository clone failed.' }
    } else {
        $RlDownload = Join-Path ([IO.Path]::GetTempPath()) ('a2c-' + [guid]::NewGuid())
        New-Item -ItemType Directory $RlDownload | Out-Null
        Invoke-WebRequest 'https://github.com/Catherine135/cs100709-bipedalwalker-a2c/archive/refs/heads/main.zip' -OutFile "$RlDownload\source.zip"
        Expand-Archive "$RlDownload\source.zip" "$RlDownload\source"
        Move-Item "$RlDownload\source\cs100709-bipedalwalker-a2c-main" $RlRepo
    }
}
Set-Location $RlRepo
if (-not (Test-Path '.\train.py') -or -not (Test-Path '.\README.md')) {
    throw 'The existing directory is incomplete; preserve it and use a new directory.'
}
```

已有目录先检查 `runs/` 的记录及当前进程，不直接重新训练。更新已有 Git checkout 前，确认没有活跃训练、保存已有改动，再使用 `git pull --ff-only`；不要 `reset --hard` 或删除日志。ZIP 副本用源码 SHA-256 标识版本，Git 副本同时记录 commit。

为这次接手创建独立证据目录；保留以前的运行目录：

```powershell
$RlTaskDir = Join-Path $RlRepo ('runs\handoff-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 6))
New-Item -ItemType Directory -Force $RlTaskDir | Out-Null
$env:PYTHONUNBUFFERED = '1'
$env:SDL_VIDEODRIVER = 'dummy'
$env:SDL_AUDIODRIVER = 'dummy'
$env:MPLBACKEND = 'Agg'
Get-FileHash '.\train.py' -Algorithm SHA256 | Format-List | Out-File "$RlTaskDir\source-hash.txt"
```

## 4. 安装独立环境

基线为 Python 3.11、NumPy 1.26.4、Gymnasium 0.29.1、PyTorch 2.8.0、Box2D 2.3.8、pygame 2.6.1、TensorBoard 2.20.0。Box2D 使用 Conda 预编译包；NumPy 使用 pip wheel，避免混用 Conda BLAS/OpenMP 与 PyTorch wheel 的运行库。不另外安装 `gymnasium[box2d]`。

查找已有 Conda 可执行文件，不依赖 `conda activate` 或 shell 初始化：

```powershell
$RlConda = $null
$RlCommand = Get-Command conda.exe -ErrorAction SilentlyContinue
$RlCandidates = @(
    $env:CONDA_EXE,
    $(if ($RlCommand) { $RlCommand.Source }),
    (Join-Path $env:USERPROFILE 'miniforge3\Scripts\conda.exe'),
    (Join-Path $env:USERPROFILE 'miniconda3\Scripts\conda.exe'),
    (Join-Path $env:LOCALAPPDATA 'miniforge3\Scripts\conda.exe'),
    (Join-Path $env:LOCALAPPDATA 'miniconda3\Scripts\conda.exe')
)
foreach ($RlCandidate in $RlCandidates) {
    if ($RlCandidate -and (Test-Path $RlCandidate -PathType Leaf)) {
        $RlConda = $RlCandidate
        break
    }
}
```

如果 `$RlConda` 为空，按 [Miniforge 官方 Windows 安装说明](https://github.com/conda-forge/miniforge#windows)自行下载 `Miniforge3-Windows-x86_64.exe` 及对应 `.sha256`，核对 SHA-256 后安装。选择当前用户可写、只含 ASCII 字符且无空格的专用目录，使用 `/InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /S /D=<绝对安装目录>`，`/D` 放在最后。用 `Start-Process -Wait -PassThru` 检查退出码，将 `$RlConda` 设为该目录的 `Scripts\conda.exe`。不修改系统 PATH、其他 Python 或要求重启。默认用户路径不适合时自行选择其他可写目录；只有安装被系统策略阻止或没有可用目录时才报告阻塞。

创建项目环境并安装固定依赖：

```powershell
if (-not $RlConda) { throw 'Locate or install Miniforge before continuing.' }
$RlPrefix = Join-Path $RlRepo '.conda'
function Invoke-RlConda {
    & $script:RlConda @args
    if ($LASTEXITCODE -ne 0) { throw "Conda command failed with exit code $LASTEXITCODE." }
}
function Invoke-RlPython {
    Invoke-RlConda run --no-capture-output -p $script:RlPrefix python @args
}
if (-not (Test-Path "$RlPrefix\python.exe")) {
    Invoke-RlConda create -p $RlPrefix --override-channels -c conda-forge python=3.11 box2d-py=2.3.8 pip -y
}
Invoke-RlPython -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
Invoke-RlPython -m pip install numpy==1.26.4 gymnasium==0.29.1 pygame==2.6.1 tensorboard==2.20.0 matplotlib ipython
Invoke-RlPython -m pip check
Invoke-RlPython -m pip freeze | Out-File "$RlTaskDir\dependencies.txt" -Encoding utf8
```

函数会检查外部命令退出码；不要只依靠 `$ErrorActionPreference` 判断 Python 成功。版本依据：[PyTorch 2.8.0 官方安装命令](https://pytorch.org/get-started/previous-versions/#v280)。CUDA wheel 也可以运行 CPU，不要为了训练要求更新系统驱动。CUDA wheel 下载或导入失败时，改装同版本 CPU wheel，再执行依赖一致性检查和 CPU 自检：

```powershell
# Run only when the CUDA-wheel installation/import failed.
Invoke-RlPython -m pip install --force-reinstall torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

已有 `.conda` 的 Python／依赖不匹配或损坏时，保留它，把 `$RlPrefix` 改为新的 `.conda-时间戳` 路径，再执行创建和安装步骤，不在无关 base 环境反复修包。下载或安装失败按第 8 节最多重试两次。

## 5. 自检、测速和设备选择

先完成 CPU 自检，退出码必须为 0，末行必须包含 `All checks passed. Optimizer steps: 0.`。CPU 自检失败不能启动训练。

```powershell
$RlNumEnvs = 8
Invoke-RlPython -B train.py --mode check --device cpu --num-envs $RlNumEnvs 2>&1 | Tee-Object "$RlTaskDir\check-cpu.log"
Invoke-RlPython -B train.py --mode benchmark --device cpu --num-envs $RlNumEnvs --benchmark-seconds 20 2>&1 | Tee-Object "$RlTaskDir\benchmark-cpu-1.log"
Invoke-RlPython -B train.py --mode benchmark --device cpu --num-envs $RlNumEnvs --benchmark-seconds 20 2>&1 | Tee-Object "$RlTaskDir\benchmark-cpu-2.log"
```

自检开头打印系统、依赖、`cuda_available` 和 GPU 名称。`cuda_available=true` 时再运行：

```powershell
# Run only if CPU preflight reported cuda_available=true.
Invoke-RlPython -B train.py --mode check --device cuda --num-envs $RlNumEnvs 2>&1 | Tee-Object "$RlTaskDir\check-cuda.log"
Invoke-RlPython -B train.py --mode benchmark --device cuda --num-envs $RlNumEnvs --benchmark-seconds 20 2>&1 | Tee-Object "$RlTaskDir\benchmark-cuda-1.log"
Invoke-RlPython -B train.py --mode benchmark --device cuda --num-envs $RlNumEnvs --benchmark-seconds 20 2>&1 | Tee-Object "$RlTaskDir\benchmark-cuda-2.log"
```

每份 benchmark 输出有两个 JSON 对象：开头是环境信息，最后一个才是测速结果。从最后一个读取 `steps_per_second`，不要把整个日志直接交给 `ConvertFrom-Json`。两次结果取算术平均：**只有 CUDA 自检通过且平均速度至少比 CPU 快 10% 时选择 `cuda`，否则选 `cpu`**。这是运行选择规则，不是课程算法要求。记录两组数值、选择及理由；CUDA 不可用或失败时记录原因，继续 CPU。

测速包含环境采样及前向／反向传播，不包含优化器更新、评测或保存，不能直接预测完整训练时间。`check` 和 `benchmark` 都不改变网络权重。若 8 个环境因资源或 `spawn` 失败，依次用 4、1 个重新自检和测速；只调整环境数。

## 6. 自动发现进度并开始／恢复训练

检查 `runs/**/handoff.json` 中记录的进程，核对 PID 创建时间、命令和本仓库路径，避免 PID 被重用。若本任务仍在运行，继续监看并跳过启动命令。

没有活跃训练时，扫描本仓库 `runs/` 的检查点，读取真正的 `steps`，不要只按修改时间选文件。下面列出默认配置下的有效候选和无法读取的文件，并选择步数最多的候选；不训练或修改权重：

```powershell
$RlScan = @'
import json
from pathlib import Path
import torch
valid, skipped = [], []
expected = dict(state_dim=24, n_actions=4, hidden_dim=128, gamma=0.99, actor_lr=4e-4, critic_lr=4e-3)
for path in Path('runs').rglob('checkpoint.pt'):
    try:
        item = torch.load(path, map_location='cpu', weights_only=True)
        config = item['agent']['config']
        assert item['kind'] == 'a2c_training_checkpoint' and item['format_version'] == 1
        assert all(config.get(key) == value for key, value in expected.items())
        assert item['config']['seed'] == 543 and item['config']['batch_size'] == 64
        assert item['validation_seeds'] == [101, 103, 105, 107, 109]
        assert item['steps'] >= 0
        valid.append(dict(path=str(path.resolve()), steps=item['steps'], updates=item['updates']))
    except Exception as error:
        skipped.append(dict(path=str(path.resolve()), error=repr(error)))
valid.sort(key=lambda row: (row['steps'], row['updates'], row['path']), reverse=True)
print(json.dumps(dict(selected=valid[0] if valid else None, candidates=valid, skipped=skipped)))
'@
$RlScanResult = (Invoke-RlPython -c $RlScan | Out-String) | ConvertFrom-Json
$RlScanResult | ConvertTo-Json -Depth 8 | Out-File "$RlTaskDir\checkpoint-scan.json" -Encoding utf8
```

发现检查点但全部损坏／不兼容时，不静默从零重跑；先检查错误和其他备份。没有旧检查点才从零开始。目录很大时只扫描 `runs/`，不要扫描整个磁盘或 `.conda/`。

根据测速设置设备；下面 `cpu` 是默认值，满足 CUDA 条件时由 agent 改成 `cuda`：

```powershell
$RlDevice = 'cpu'
$RlTarget = 1000000
$RlResume = $null
if ($RlScanResult.selected) {
    $RlResume = $RlScanResult.selected.path
    $RlTarget = [Math]::Max($RlTarget, [int64]$RlScanResult.selected.steps)
    if ((($RlTarget - [int64]$RlScanResult.selected.steps) % $RlNumEnvs) -ne 0 -or ($RlTarget % $RlNumEnvs) -ne 0) {
        $RlNumEnvs = 1
    }
}
$RlModel = Join-Path $RlTaskDir 'a2c.model'
$RlTrainingLogs = Join-Path $RlTaskDir 'training'
$RlTrainArgs = @('-u', 'train.py', '--mode', 'train', '--device', $RlDevice,
    '--num-envs', "$RlNumEnvs", '--torch-threads', '1', '--batch-size', '64',
    '--steps', "$RlTarget", '--seed', '543', '--eval-freq', '10', '--save-freq', '1',
    '--log-dir', $RlTrainingLogs, '--model-path', $RlModel)
if ($RlResume) { $RlTrainArgs += @('--resume', $RlResume) }
```

旧检查点已达到或超过预算时，`--steps` 设为已完成步数，恢复路径只做最终评测和导出，**不会再增加优化器更新**。这是复用完成的训练，不是追加预算。只有 `checkpoint.pt` 用于 `--resume`，`a2c.model` 没有优化器和训练计数。

使用 agent 的可持续终端会话运行以下命令，保留会话 ID、Python／启动器 PID、创建时间及日志路径。工具必须允许长时间进程，不使用到时自动杀死进程的短命令超时：

```powershell
$RlRecord = [ordered]@{
    status = 'prepared'; started_at = (Get-Date).ToString('o'); repo = $RlRepo
    conda = $RlConda; prefix = $RlPrefix; task_dir = $RlTaskDir
    device = $RlDevice; num_envs = $RlNumEnvs; target_steps = $RlTarget
    resume = $RlResume; model = $RlModel; arguments = $RlTrainArgs
    source_sha256 = (Get-FileHash '.\train.py' -Algorithm SHA256).Hash
}
$RlRecord | ConvertTo-Json -Depth 6 | Out-File "$RlTaskDir\handoff.json" -Encoding utf8
Invoke-RlPython @RlTrainArgs 2>&1 | Tee-Object "$RlTaskDir\train-console.log"
```

agent 在启动后将实际进程／会话信息补写到 `handoff.json`，状态改为 `running`；这些值取决于终端工具，不能虚构 PID。每 30–60 秒读取新增日志和进程状态，记录步数、回合数及评测。不要打印了 `Training directory:` 就报告完成或结束监看。长时间运行需要电脑保持唤醒；可在训练期间申请 Windows 系统唤醒请求，结束时释放，不全局修改电源方案。

程序每 100 次更新打印进度；验证时采样进度暂时停止。`active_steps_per_second` 不含验证耗时，不能因一次日志间隔较长就杀掉进程。`--steps` 和 `--batch-size` 必须能被 `--num-envs` 整除。

恢复时，模型超参数、训练随机种子和验证种子来自检查点；设备、环境数、批大小等运行参数来自本次命令。环境回合重新开始，不是 Box2D 世界状态的逐位连续重放。

## 7. 结束后的验收和交付

退出码为 0 本身不代表训练完成，因为正常 Ctrl+C 也会保存并退出。完成下列步骤后才能报告完成：

1. 找到本次 `Training directory:` 的 `checkpoint.pt`，确认 `kind`、格式、实际 `steps >= 1,000,000`、网络参数有限，记录 `episodes`、`updates`、`best_score`。步数不足就恢复到预算。`best_score` 是验证成绩，不是最终示例评测成绩。
2. 确认 `$RlModel` 和训练目录中的 `a2c.model` 存在、可载入，并与检查点的 `best_agent` 一致。不要用最后一轮权重替换验证最佳模型。记录模型／源码 SHA-256 及可用的 Git commit。
3. 用新 Python 进程验证教师导入方式，再运行完整评测：

```powershell
$RlLoadCheck = @'
import sys
import numpy as np
import torch
from train import *
agent = torch.load(sys.argv[1])
state = np.zeros(24, dtype=np.float32)
first = agent.select_action(state, eval=True)
second = agent.select_action(state, eval=True)
assert first.shape == (4,) and first.dtype == np.float32
assert np.isfinite(first).all() and (np.abs(first) <= 1).all()
assert np.array_equal(first, second)
assert all(p.device.type == 'cpu' for n in (agent.actor, agent.critic) for p in n.parameters())
print('PASS: teacher import, default torch.load, CPU tensors and deterministic actions')
'@
Invoke-RlPython -c $RlLoadCheck $RlModel 2>&1 | Tee-Object "$RlTaskDir\teacher-load.log"
Invoke-RlPython -B train.py --mode eval --device cpu --model-path $RlModel --eval-seeds 1 3 5 7 9 2>&1 | Tee-Object "$RlTaskDir\evaluation.log"
```

4. 将评测输出的**最后一个 JSON 对象**保存为 `evaluation.json`，包含 `seeds`、五个 `rewards`、五个 `lengths`、`mean_reward` 和 `std_reward`。确认结果有限，平均值等于五个完整回合原始总奖励的算术平均；低分／负分也如实保留。
5. 从 TensorBoard 事件文件导出训练回合奖励、验证平均奖励两张 PNG 曲线，横轴使用累计环境步数。恢复过训练时合并续训和所引用检查点的来源运行，标明恢复边界；缺少旧事件文件则说明图覆盖范围。平滑线可以辅助展示，但保留原始数据。
6. 在 `$RlTaskDir\submission\` 放入**本次实际训练所用的** `train.py` 和最佳 `a2c.model`，记录 SHA-256，压缩为 `$RlTaskDir\submission.zip`。提交包不含环境、临时文件或检查点；检查点和完整日志单独保留。
7. 生成 `$RlTaskDir\RESULTS.md`，将 `handoff.json` 更新为 `completed`，向用户报告结果与绝对路径，不自动向 GitHub 推送产物。

`RESULTS.md` 至少包含：源码版本／哈希、依赖版本、实际硬件、两种设备测速及选择、训练起止时间／耗时、起始和最终步数、回合／更新数、是否恢复、最佳验证平均奖励、最终逐种子奖励／长度及平均值、两张曲线、教师载入验证、模型哈希、模型／提交包／检查点路径，以及实际错误和修复。没有 GPU 验证就写明没有，不猜测性能。

只有证据齐全才算“本次预算训练与交付完成”，不能把它写成已达到某个未经验证的奖励水平。若 agent 会话提前结束，写真实的 `running`／`interrupted` 状态、进程及恢复信息，并明确未完成。

## 8. 故障处理规则

| 情况 | agent 应执行的处理 |
| --- | --- |
| 链接显示 404 | 仓库已公开；检查 URL、网络及代理，也可下载公开 ZIP；无需索取 GitHub 凭据。 |
| 没有 Conda／Python | 按第 4 节用户级安装 Miniforge，创建 Python 3.11 项目环境。 |
| 下载超时 | 保存错误，最多重试两次；CUDA wheel 失败可以改 CPU wheel。仍失败就报告具体地址和错误。 |
| Box2D 编译失败／缺少 SWIG | 不继续编译 pip Box2D，改用独立 Conda 环境中的 `box2d-py=2.3.8`。 |
| NumPy／OpenMP／DLL 错误 | 在新环境重建固定依赖，NumPy 用 pip；不设置 `KMP_DUPLICATE_LIB_OK` 掩盖冲突。 |
| CUDA 不可用、自检失败或不更快 | 记录原因，使用 CPU；不擅自升级驱动或重启。 |
| 并行启动失败 | 在终端运行 `train.py`，不要直接放入 Notebook；检查错误，依次尝试 4、1 个环境并重新自检。 |
| NaN／Inf 或非有限梯度 | 保留错误与最后有效检查点；排除依赖／硬件问题后最多按相同配置恢复两次，仍复现则报告，不静默改动损失、奖励或学习率。 |
| Ctrl+C／关机／进程中断 | Ctrl+C 请求在更新边界保存。重新接手先确认旧进程退出，再从有效检查点恢复到累计预算；强制结束可能丢失最近尚未保存的进度。 |
| 检查点损坏 | 保留坏文件和错误，尝试其他有效、配置匹配的检查点；全部失效则报告，不删除证据或默默从头训练。 |
| 训练完成但奖励低 | 完成评测和交付，报告真实得分；不自动扩充预算。 |

恢复创建新运行目录，旧目录和历史最佳策略保留。外部模型路径放在本次 `$RlTaskDir`，不会覆盖其他任务的 `a2c.model`。

## 9. 文件、接口和已有验证

| 文件 | 用途 |
| --- | --- |
| `train.py` | 唯一训练源码；`train / eval / check / benchmark` 入口 |
| `README.md` | 完整执行说明、默认决策和验收标准 |
| `AGENTS.md` | agent 入口，指向本文并明确执行范围 |
| `.github/workflows/validate.yml` | Windows CPU 非训练自检，不代替本机训练 |
| `.gitignore` | 忽略模型、检查点、日志和本地环境 |

每个训练目录包含 `config.json`、`events.out.tfevents.*`、`evaluations.jsonl`、`checkpoint.pt` 和最佳 `a2c.model`。可以用项目环境的 `tensorboard --logdir <日志根目录>` 查看曲线；agent 无需依赖图形页面判断进度。

教师接口如下；保留 `train.py` 文件名，与模型放在可导入的位置：

```python
import torch
from train import *

agent = torch.load('a2c.model')
action = agent.select_action(state, eval=True)
```

`torch.save(agent, ...)` 保存完整 `Agent` 对象，其持久化状态只含 CPU 张量和基本类型。导入 `train` 时仅注册本文件的 `Agent`，兼容 PyTorch 2.6 起默认受限载入。评测使用策略均值，返回 `(4,)` 的 `float32` NumPy 数组；`eval=False` 返回原始动作和执行动作。

内置检查覆盖手算 TD／损失、梯度隔离、动作接口、自动重置、剩余批次、真实 Box2D 渲染、8 个 spawn 工作进程、独立进程的教师载入、五个完整评测回合、TensorBoard、恢复和中断保存。本机和 [Windows CPU 检查](https://github.com/Catherine135/cs100709-bipedalwalker-a2c/actions/runs/42240555441)已通过。流程检查使用模拟更新，禁止优化器 `step()`，不证明训练收敛；RTX 4080 验证由接手 agent 在目标电脑完成。

兼容性依据：[Gymnasium 注册源码](https://github.com/Farama-Foundation/Gymnasium/blob/v0.29.1/gymnasium/envs/__init__.py)、[时间限制处理](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)、[PyTorch 模型载入](https://docs.pytorch.org/docs/2.8/generated/torch.load.html)、[A2C CPU 建议](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html)。算法依据仍是课程文件。
