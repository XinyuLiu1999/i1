# DenseText SFT → BizGenEval（单机多 GPU）

在 **另一台有 NVIDIA GPU 的机器** 上运行，默认使用 `i1_sft` conda 环境。
入口 `run_densetext.sh` 默认仅准备数据、每卡启动一个推理进程并检查图片，
无需 Gemini API key。显式使用 `--stage evaluate` 可评分已有图片，
`--stage all` 可完成生成、官方评分和汇总。GPU 并行用于生成；官方评分通过 API 并发。

## 环境与路径

按 `torch_train/DENSETEXT_MULTINODE_SFT.md` 配好 `i1_sft` 环境及 T5Gemma、
FLUX.2 模型缓存。把 i1 和 BizGenEval 放在同一父目录；否则设置
`BIZGENEVAL_ROOT`。生成阶段使用现有 `i1_sft` 环境：

```bash
export I1=/path/to/i1
export BIZGENEVAL_ROOT=/path/to/BizGenEval
export HF_HUB_CACHE=/path/to/model/cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

离线开关只控制 Hugging Face 模型加载；Gemini 评分仍需网络和 API 配额。
API key 也可按 BizGenEval 自身约定配置在 YAML 中。评估器的模型、并发量、重试策略
使用 `BizGenEval/config/evaluation_config.yaml`；可用 `--evaluation-config` 指定其他配置。

## 默认生成图片

```bash
export SFT_WORKDIR=/shared/outputs/densetext_sft_1024_run001
export SFT_CHECKPOINT=$SFT_WORKDIR/checkpoint.pt-000010000
export OUTPUT_ROOT=$SFT_WORKDIR/bizgeneval/step10000_1024

# 自动使用本机所有可见 GPU；也可 export GPU_IDS=0,1,2,3
bash "$I1/benchmark_eval/bizgeneval/run_densetext.sh"
```

checkpoint 优先级：`--checkpoint` > `SFT_CHECKPOINT` >
`SFT_WORKDIR/checkpoint.pt`。直接加载训练输出的 `config + model`（EMA 权重），
无需转换或合并多节点分片。优化器状态通过已有 mmap 加载方式避免整份读入内存。
建议使用保留的 `checkpoint.pt-000010000` 等静态副本，避免训练同时覆盖 checkpoint。

`GPU_IDS` / `--gpu-ids` 优先；否则遵循 `CUDA_VISIBLE_DEVICES`，未设置时由
PyTorch 探测。每个进程持有完整模型，不需要 torchrun、NCCL 或多节点环境变量。
每卡需放得下 DiT、文本编码器及 VAE；默认 diffusion/VAE batch 均为 1。
所有子进程使用相同的 `i1_sft` Python。若不用 conda 命令，可设置
`PYTHON_BIN=/path/to/envs/i1_sft/bin/python`。

## 混合分辨率 checkpoint

与训练一致，使用 step=32 的 bucket frontier 和精确 3:2/2:3 anchor，按数据的
aspect_ratio 选择最接近的桶。默认 1024 像素面积档；mixed-resolution checkpoint
可额外测 1536 / 2048。各档使用独立输出目录：

```bash
bash "$I1/benchmark_eval/bizgeneval/run_densetext.sh" \
  --checkpoint /shared/outputs/densetext_sft_multires_run001/checkpoint.pt-000010000 \
  --resolution 2048 --output-root /shared/eval/step10000_2048
```

默认使用 250 步、CFG 12、CFG rescale 1、timestep shift 0.3、seed 42，
原始 prompt 不重写。上下文固定为 1024 token，超长 prompt 默认截断；用
`--caption-overflow error` 可改为报错。`--text-num-tokens` 大于 1024 会扩展上下文，
属于超出训练长度的推理设置，应单独报告。比较 checkpoint 时保持相同分辨率和采样参数。

## 分阶段、检查与续跑

```bash
# 在目标机器先检查命令；不加载模型、不调用 API，但会写输入与运行配置。
bash "$I1/benchmark_eval/bizgeneval/run_densetext.sh" \
  --dry-run --gpu-ids 0,1 --limit 4 --num-steps 2 --output-root /shared/eval/smoke

# 去掉 --dry-run，保持其余参数一致，即运行该 smoke test。

# 仅生成（不需要 Gemini key）
bash "$I1/benchmark_eval/bizgeneval/run_densetext.sh" --stage generate

# 之后评分、验证及汇总；沿用相同 checkpoint、数据和采样参数
conda run -n i1_sft python -m pip install google-genai pyyaml requests
export GEMINI_API_KEY=your-key
bash "$I1/benchmark_eval/bizgeneval/run_densetext.sh" --stage evaluate

# 显式执行生成、评分和汇总全流程
bash "$I1/benchmark_eval/bizgeneval/run_densetext.sh" --stage all

# 仅准备输入，不探测 CUDA，也不加载 checkpoint
bash "$I1/benchmark_eval/bizgeneval/run_densetext.sh" --stage prepare
```

同命令重跑会跳过已有图片和完整评分。GPU 数量决定随机流分片，续跑时须保持
工作进程数不变，GPU 编号可换。checkpoint 大小/修改时间、数据或采样设置改变时，
脚本拒绝复用目录，防止混合结果；smoke 与正式评估也应使用独立目录。
同一目录同时只允许运行一个入口进程。工作进程失败时终止其余进程，不进入评分。
评分不完整时拒绝生成汇总；重复命令可重试不完整评分。改变 judge 配置需要新目录或
`--force-rerun`（重新评分全部图片）；中断的强制评分须继续带此参数。

默认生成阶段输出 `run.json`、`workers.json`、`inputs/`、`images/` 和
`logs/worker_N.log`。显式评分后增加 `judge.json`、`eval_results/` 和 `summaries/`。最终报告是
`summaries/summary_by_domain.csv`、`summary_by_dimension.csv`、`summary.json`，
使用官方 easy/hard/all 评分口径。

CPU 回归测试（不运行模型或访问 API）：

```bash
conda run -n i1_sft python -m unittest discover \
  -s "$I1/benchmark_eval/bizgeneval" -p 'test_densetext_eval.py'
```
