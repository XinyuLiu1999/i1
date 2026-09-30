# 平台多节点 SFT 训练

`run_multinode.sh` 适配平台在**每个节点执行一次**启动脚本的模式。平台提供的
`WORLD_SIZE` 是节点数，`RANK` 是节点编号；脚本将它们传给 torchrun 的
`--nnodes`、`--node_rank`。训练子进程中的同名变量由 torchrun 重新生成，分别表示
总 GPU 进程数、全局进程编号。不要在每张 GPU 上分别执行这个启动脚本。

i1 的 `training.parallel.init_distributed()` 已使用全局 rank 建立 FSDP/TP mesh，
数据采样也按全局 data-parallel rank 划分，因此无需修改模型或损失函数。
这份启动脚本面向 JSONL/Parquet/像素缓存 SFT，默认配置为 `configs/sft_1024.py`。

## 1. 所有节点的准备工作

- 使用相同代码、Python/PyTorch 版本；当前迁移环境是 Python 3.11、PyTorch 2.9.1+cu126。
  CPU 节点安装的环境只有放在共享盘上，且各节点系统兼容时才能直接复用。
- 每个节点有相同数量的可见 GPU。`GPUS_PER_NODE` 是每节点进程数，而不是整个任务的 GPU 总数。
- 所有节点能访问同一份 manifest 及其引用的图片/Parquet/二进制缓存。
  本地数据副本必须内容、记录顺序和路径一致，否则各 rank 的分桶计划可能不一致。
- 初始化 checkpoint、恢复 checkpoint、配置文件在所有节点可读。`SFT_WORKDIR`
  必须是所有节点共同访问的共享输出目录；只有全局 rank 0 写 checkpoint。
- 提前准备 T5Gemma 和 FLUX.2 VAE 的完整缓存。不要把旧机器的 `/cephfs/...` 路径直接照搬。
  脚本默认使用 `/user/lxy8802/.cache/data_juicer/models`，并设置
  `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，不联网下载模型。
  缓存中的 `models--google--t5gemma-2b-2b-ul2-it` 和
  `models--black-forest-labs--FLUX.2-dev` 需保留完整的 snapshots/blobs/refs 结构。
- `MASTER_ADDR` 必须在所有节点可解析且可达，`MASTER_PORT` 用平台分配值。
  NCCL 还需要节点间数据通信，只有 rendezvous 端口可达并不足够。

重启后先重新激活环境，并重新设置下面的变量。建议将它们保存为平台的任务启动脚本。
W&B 默认在线开启。API key 优先使用平台注入的 `WANDB_API_KEY`；未设置时，脚本从
`WANDB_KEY_FILE`（默认 `/user/lxy8802/.bashrc`）中读取 `export WANDB_API_KEY=...` 这一行
（只解析该行，不 source 整个文件，也不打印 key）；两者都没有时，使用之前 `wandb login` 保存的凭据。在线训练启动前，仅节点 0 自动执行
`python -m wandb login --verify`，读取 `WANDB_API_KEY` 或已有凭据；验证失败会退出，
不会等待交互输入。命令预览、通信检查及 offline/disabled 模式跳过登录。
项目名由 `WANDB_PROJECT` 指定，优先于训练配置中的项目名。

## 2. 平台任务启动脚本

以下路径基于迁移机器 `/user/lxy8802`；替换数据和输出路径。
**所有节点执行同一份脚本，保留平台注入的 RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT。**

```bash
#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/user/lxy8802/.cache/huggingface
export HF_HUB_CACHE=/user/lxy8802/.cache/data_juicer/models
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset TRANSFORMERS_CACHE

export SFT_MANIFEST=/shared/datasets/cache_1024/cache.jsonl
export SFT_INIT="$HF_HUB_CACHE/i1-3B/1024_resolution_checkpoint_torch.pt"
export SFT_WORKDIR=/shared/outputs/i1_multinode_run001

# 例如：2 节点 × 每节点 4 卡。节点数由平台 WORLD_SIZE 提供。
export GPUS_PER_NODE=4
export GLOBAL_BATCH_SIZE=32
export GRAD_ACCUM=4
export WANDB_PROJECT="${WANDB_PROJECT:-DenseText-SFT}"
export WANDB_MODE=online  # 离线测试时可改为 offline

exec bash /user/lxy8802/i1/torch_train/run_multinode.sh
```

脚本默认加载 `/user/lxy8802/miniforge3/etc/profile.d/conda.sh` 并激活 `i1_sft`。
可通过 `CONDA_SH` / `CONDA_ENV` 修改路径和环境名。也可设置
`TRAIN_PYTHON=/user/lxy8802/miniforge3/envs/i1_sft/bin/python`，无需依赖 Conda 激活。
未设置 `SFT_INIT` / `SFT_RESUME` 时，默认从
`$HF_HUB_CACHE/i1-3B/1024_resolution_checkpoint_torch.pt` 初始化。
设置 `SFT_RESUME` 时不会补入默认初始化权重。模型离线加载与 W&B 在线记录相互独立；
如需下载模型，显式设置 `HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0`。
默认关闭 `torch.compile`，并显式传入 `--completion-config ""`，不触发旧机器的
单机完成通知/关机接口。任务生命周期由当前平台管理。
训练输出分别追加到 `SFT_WORKDIR/logs/node_0.log` 等文件，进程退出码直接返回平台。

## 3. 先检查命令和节点间通信

在设置好上述变量后，可在 CPU 节点预览命令（不加载模型，不访问数据，不启动训练）：

```bash
WORLD_SIZE=2 RANK=0 MASTER_ADDR=node0.example GPUS_PER_NODE=4 \
  bash /user/lxy8802/i1/torch_train/run_multinode.sh --dry-run
```

在平台上让**每个节点**执行下列命令，验证真实 GPU/NCCL 通信；无需数据或模型权重：

```bash
bash /user/lxy8802/i1/torch_train/run_multinode.sh --check-communication
```

所有 rank 都应打印 `PASS ... backend=nccl` 并正常退出。检查覆盖 BF16 支持、
跨 rank `all_reduce` 和 `all_gather`。它不验证完整模型的显存占用或 FSDP 反向传播。
该检查的 collective 超时默认 120 秒，可追加 `--timeout 300`；等待其他节点加入
torchrun rendezvous 的时间不受此参数控制，缺失节点时应检查平台启动状态。

通信失败时，检查各节点日志、相同节点数/端口设置、DNS、防火墙和平台网络配置。
可设置 `NCCL_DEBUG=INFO`；仅在明确实际网卡名称时设置 `NCCL_SOCKET_IFNAME` /
`GLOO_SOCKET_IFNAME`，不要照搬其他集群的网卡或强制关闭 InfiniBand。

## 4. 三步训练与恢复验证

使用独立的共享测试目录，在所有节点执行：

```bash
export SFT_WORKDIR=/shared/outputs/i1_multinode_smoke001
bash /user/lxy8802/i1/torch_train/run_multinode.sh --total_steps 3 --log_every 1
```

确认 loss 正常、所有节点退出成功、共享目录出现 `checkpoint.pt`，再恢复到第 4 步：

```bash
unset SFT_INIT
export SFT_RESUME="$SFT_WORKDIR/checkpoint.pt"
bash /user/lxy8802/i1/torch_train/run_multinode.sh --total_steps 4 --log_every 1
```

`--total_steps` 是最终总步数，不是额外训练步数。正式新训练需取消 `SFT_RESUME`、
重新设置 `SFT_INIT` 并换新的输出目录；不传 `--total_steps` 时沿用配置中的 epoch/步数设置。
精确恢复请保留同一份数据、配置、拓扑、batch、累积次数和随机种子。
为防止误恢复，输出目录已有 `checkpoint.pt` 时，脚本要求 `SFT_RESUME` 显式指向它。

## 5. 拓扑与 batch 参数

默认 `TP_SIZE=1`，`FSDP_SIZE=节点数 × 每节点GPU数`，模型跨所有 GPU 分片。
这也适用于每个节点只有一张 A100 的情况；需要足够的节点间带宽。

```text
总进程数 = NNODES × GPUS_PER_NODE
数据并行进程数 = 总进程数 / TP_SIZE
每卡 microbatch = GLOBAL_BATCH_SIZE / 数据并行进程数 / GRAD_ACCUM
```

| 节点数 × 每节点 GPU | 全局 batch | 梯度累积 | 每卡 microbatch |
|---|---:|---:|---:|
| 2 × 1 | 32 | 16 | 1 |
| 4 × 1 | 32 | 8 | 1 |
| 2 × 4 | 32 | 4 | 1 |
| 2 × 8 | 32 | 2 | 1 |

这些参数保证 batch 整除，不保证模型一定不 OOM。默认 DataLoader 每 rank 有 4 个 worker，
需检查 CPU 内存和 `/dev/shm`。当前 checkpoint 实现会在各 rank 汇集完整模型、EMA 和优化器
状态到 CPU，保存/恢复时主机内存与共享盘空间也必须足够。

需要节点内分片、节点间复制（HSDP）时，可设 `FSDP_SIZE=GPUS_PER_NODE / TP_SIZE`
的计算结果；默认的全局分片适合先验证功能。`TP_SIZE` 必须整除每节点 GPU 数，
`FSDP_SIZE` 必须整除数据并行进程数，batch 必须整除数据并行进程数乘累积次数。

可覆盖的环境变量：`NNODES`、`NODE_RANK`（优先于平台的 WORLD_SIZE/RANK）、
`MASTER_ADDR`、`MASTER_PORT`、`GPUS_PER_NODE`、`CONDA_SH`、`CONDA_ENV`、`TRAIN_PYTHON`、`SFT_CONFIG`、
`SFT_MANIFEST`、`SFT_INIT`/`SFT_RESUME`、`SFT_WORKDIR`、`FSDP_SIZE`、`TP_SIZE`、
`GLOBAL_BATCH_SIZE`、`GRAD_ACCUM`。其他训练参数直接追加在脚本后，例如 `--token_len 1024`。

## 本地回归测试

```bash
cd /path/to/i1/torch_train
python -m unittest discover -s tests -p 'test_multinode.py' -v
```

测试包括两份独立 torchrun agent、四个 CPU/Gloo worker 的实际 rendezvous 与通信，
用于验证平台节点编号转换、参数校验和节点日志。CPU 测试不代替真实跨机器 NCCL/FSDP 验证。
