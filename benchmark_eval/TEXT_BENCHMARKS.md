# 自选 checkpoint → LongText / CVTG 多 GPU 生成与评测

统一入口：`benchmark_eval/run_text_benchmarks.sh`。单次选择一个 checkpoint 和一个
benchmark，默认 **只生成并验证图片**。LongText 支持显式评分及汇总；CVTG 只实现生成，
`--stage evaluate` 和 `--stage all` 会直接报错。

在有 NVIDIA GPU 和模型缓存的机器上运行。默认使用 `i1_sft` conda 环境；
设置 `PYTHON_BIN=/path/to/env/bin/python` 可改用指定解释器。生成依赖见
[PyTorch inference](../torch_inference/README.md)，需缓存 T5Gemma 和 FLUX.2 VAE。
checkpoint 使用现有 `generate.py` 支持的 i1 PyTorch `config + model` 格式，
可直接读取训练保存的原始或 SFT checkpoint，无需合并或转换。

## 默认只生成

```bash
export I1=/path/to/i1
export SFT_CHECKPOINT=/shared/train/checkpoint.pt-000010000
export GPU_IDS=0,1,2,3,4,5,6,7

# LongText：全量 160 条 prompt，每条 4 张，共 640 张
bash "$I1/benchmark_eval/run_text_benchmarks.sh" \
  --benchmark longtext --output-root /shared/eval/step10000_longtext

# CVTG：全量 2000 条 prompt，每条 1 张；cvtg-2k 是别名
bash "$I1/benchmark_eval/run_text_benchmarks.sh" \
  --benchmark cvtg --output-root /shared/eval/step10000_cvtg
```

checkpoint 优先级：`--checkpoint` > `SFT_CHECKPOINT` > `SFT_WORKDIR/checkpoint.pt`。
未指定输出目录时，使用 checkpoint 同级的
`<benchmark>/<checkpoint文件名>_<resolution>_<prompt-variant>/`。
`--gpu-ids` / `GPU_IDS` 优先，否则遵循 `CUDA_VISIBLE_DEVICES`，再由 PyTorch 探测 GPU。
每个生成进程只看到一张 GPU 并加载完整模型；每卡需容纳 DiT、文本编码器和 VAE。
按 prompt 连续分片，不需要手动执行 torchrun。

默认：原始 prompt、不在线重写、1024×1024、250 步、CFG 12、CFG rescale 1、
timestep shift 0.3、seed 42、diffusion/VAE batch 均为 1。支持
`--prompt-variant simple_rewrite|complex_rewrite` 使用仓库内预先改写的 prompt。
LongText 评分始终使用原始 benchmark 的目标文字。

默认保留 checkpoint 原生文本上下文，超长文本截断。可设置
`--caption-overflow error` 在超长时报错，或用 `--text-num-tokens 1024` 等显式扩展上下文；
该值不能小于 checkpoint 原生上下文。扩展上下文属于推理设置变更。
`--resolution` 支持 256/512/1024/1536/2048，均为正方形；应选择适合 checkpoint 的分辨率。

## LongText 评分和汇总

评分使用现有 Qwen2.5-VL-7B OCR 和官方 Text Score 计算逻辑，通过 torchrun 在选定 GPU 上分片。
准备评分环境可参照 [benchmark 环境说明](README.md#3-longtext-bench) 的 LongText 部分。
不同环境的依赖版本可能不同，推荐用 `--evaluation-python` 指向已有 LongText 环境，
也可设置 `LONGTEXT_PYTHON`；默认使用生成环境解释器。评分机器需要 Qwen 模型缓存或下载权限。

```bash
# 仅评分已有图片，保持生成时的 checkpoint、limit、prompt variant 和采样参数一致
bash "$I1/benchmark_eval/run_text_benchmarks.sh" \
  --benchmark longtext --stage evaluate \
  --output-root /shared/eval/step10000_longtext \
  --evaluation-python /path/to/envs/longtext/bin/python

# 新目录中执行生成 → 完整性检查 → OCR → 汇总
bash "$I1/benchmark_eval/run_text_benchmarks.sh" \
  --checkpoint /shared/train/checkpoint.pt-000020000 \
  --benchmark longtext --stage all \
  --output-root /shared/eval/step20000_longtext \
  --evaluation-python /path/to/envs/longtext/bin/python
```

LongText 图片直接命名为 `<官方prompt_id>_<repeat>.png`，无需运行 `longtext/process.py`，
也不要对新入口的图片再次执行该重命名脚本。小样本和全量均可直接评分。
评分前检查图片数量、尺寸及可解码性；OCR 后检查结果覆盖、重复项及目标文字，全部通过才发布汇总。
每次显式评分都会重新 OCR 全部所选图片，不复用旧评分分片。失败不发布新汇总；
若已有成功评分，其结果会保留。评分可使用与生成不同的 GPU 数量。

输出：

```text
run.json                       checkpoint、数据摘要和采样设置
workers.json                   生成分片边界
inputs/samples.jsonl           生成 prompt、文件名与原始元数据
inputs/output_names.txt        预期图片清单
inputs/text_prompts.jsonl      LongText 原始评分 prompt
images/                        PNG 图片
logs/worker_N.log              每卡生成日志
logs/evaluation/worker_0.log   LongText torchrun 评分日志
eval_results/results.jsonl    逐图 OCR 与文字匹配计数
eval_results/scores.txt        官方 Text Score
eval_results/summary.json      汇总分数、图片数、prompt 数、checkpoint
```

`eval_results/` 仅在 LongText 显式评分成功后产生。Text Score 是所有图片的
匹配词数之和除以目标词数之和，范围 0–1。CVTG 保留 `00000.png` 等平铺编号，
其 `inputs/samples.jsonl` 记录对应的官方 category/ID；本入口不整理 CVTG 评分目录或执行评分。

## 小样本检查、续跑与参数保护

```bash
# 只检查参数和分片；不加载模型，但写入配置及输入清单
bash "$I1/benchmark_eval/run_text_benchmarks.sh" \
  --checkpoint /shared/train/checkpoint.pt-000010000 \
  --benchmark longtext --gpu-ids 0,1 --limit 4 --num-steps 2 \
  --output-root /shared/eval/longtext_smoke --dry-run

# 去掉 --dry-run 即生成 16 张图片；加 --stage all 则继续评分并汇总
# 只准备清单、不检测 GPU：使用 --stage prepare
```

`--limit` 按 prompt 计数，0 表示全量。dry-run 显式设置 `--gpu-ids` 可完全避免 CUDA 探测。
同命令重新生成会跳过已有图片，保留缺失图片对应的随机流。生成续跑需保持分片数量不变，
GPU 编号可更换。checkpoint 文件大小/修改时间、数据、采样参数或 prompt variant 改变时，
拒绝复用输出目录；smoke 和正式运行使用不同目录。
发现损坏或尺寸错误的图片时先报错，删除对应坏文件后可续跑。
同目录只允许一个入口进程运行；任一生成进程失败会停止其余进程，并阻止进入评分。

CPU 回归测试（需要 Pillow，不加载模型、不访问网络）：

```bash
python -m unittest discover -s "$I1/benchmark_eval" -p 'test_text_benchmarks.py' -v
```
