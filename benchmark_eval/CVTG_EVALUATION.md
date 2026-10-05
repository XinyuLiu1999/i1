# CVTG 模型下载与多 GPU 评估

`run_text_benchmarks.sh --benchmark cvtg` 支持 `--stage evaluate` 和 `--stage all`。
每张 GPU 启动一个独立进程，按图片分片，每卡加载完整 OCR、CLIP、OpenCLIP 和
CLIP-FlanT5-XXL。不是模型跨卡切分；多张小显存卡不会合并成一张大显存卡。
CLIP 默认每批 16 张，VQA 和美学评分逐图执行，模型只加载一次。

建议 Linux + NVIDIA GPU，每卡预留 40–48GB 或更多显存，并先用小样本验证峰值。
[上游建议 XXL 使用 40GB GPU](https://github.com/linzhiqiu/t2v_metrics/blob/main/V_3.0_README.md#notes-on-gpu-and-cache)。
多个 worker 会增加主机内存需求；初始化按共享缓存锁串行进行，避免 Paddle 并发下载
损坏缓存和同时加载 XXL 造成过高内存峰值。初始化完成后各卡独立并行评分。
本地 CPU 测试验证了调度、指标汇总及失败处理，未替代目标 GPU 上的模型推理验收。

## 1. 准备环境并完整下载模型

以下命令在 GPU 服务器上运行；将 `I1` 和 `CVTG_CACHE_DIR` 改为服务器实际路径。
如果已经安装 `textcrafter_eval` 环境，跳过创建环境和安装依赖两步。

```bash
export I1=/path/to/i1
export CVTG_CACHE_DIR=/shared/cache/cvtg

cd "$I1/benchmark_eval/cvtg-2k"
conda env create -f unified_environment.yml
conda activate textcrafter_eval
bash install_paddle_deps.sh
export CVTG_PYTHON="$CONDA_PREFIX/bin/python"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# 无需占用 GPU；不会将 11B VQA 模型加载进内存。
# CLIP/OCR 小模型会短暂在 CPU 初始化。重复执行会复用完整缓存。
"$CVTG_PYTHON" "$I1/benchmark_eval/cvtg-2k/download_models.py" \
  --cache-dir "$CVTG_CACHE_DIR" \
  --no-hf-mirror
```

需要 Hugging Face 镜像时，将 `--no-hf-mirror` 换成 `--use-hf-mirror`。
镜像选项只影响 Hugging Face，CLIP 和 PaddleOCR 原始下载站点仍需可访问。
`--no-hf-mirror` 显式取消已有 `HF_ENDPOINT`，并不表示离线。

下载资产包含：

| 资产 | 用途 |
| --- | --- |
| `zhiqiulin/clip-flant5-xxl`（约 22.9GB） | VQAScore 主模型 |
| `google/flan-t5-xxl` tokenizer | VQA 文本分词；不下载其基座大模型权重 |
| `openai/clip-vit-large-patch14-336` | VQA 视觉编码器及预处理配置 |
| OpenAI CLIP `ViT-L/14` | CLIPScore |
| OpenCLIP `ViT-L-14`, pretrained=`openai` | 美学评分视觉特征 |
| PaddleOCR 英文检测、识别和方向分类模型 | Word Accuracy / NED |
| 仓库中的 `sa_0_4_vit_l_14_linear.pth` | 美学预测头；无需下载 |

建议缓存磁盘预留 40–60GB，环境和图片另计。新下载器与评估使用同一缓存布局：
HF/OpenCLIP 缓存在指定根目录；官方 CLIP 在 `clip/`；OCR 在 `paddleocr/{det,rec,cls}/`。
旧版 `~/.cache/clip` 和 `~/.paddleocr` 的缓存不会自动迁移，首次可能重新下载。
必须沿用项目 requirements 中的评估依赖版本，尤其是 `t2v-metrics==1.2`、
`transformers==4.36.1` 和 PaddleOCR 2.10.0；不要直接升级到最新 t2v-metrics。

## 2. 兼容统一多 GPU benchmark 入口

```bash
export SFT_CHECKPOINT=/shared/train/checkpoint.pt-000010000
export GPU_IDS=0,1,2,3

# 仅评估该入口已生成的图片。保持生成时的 checkpoint、limit、prompt variant、采样设置。
bash "$I1/benchmark_eval/run_text_benchmarks.sh" \
  --benchmark cvtg --stage evaluate \
  --output-root /shared/eval/step10000_cvtg \
  --evaluation-python "$CVTG_PYTHON" \
  --evaluation-cache-dir "$CVTG_CACHE_DIR" \
  --clip-batch-size 16 \
  --no-hf-mirror

# 新目录：生成 → 检查图片 → 多 GPU 评估 → 汇总
bash "$I1/benchmark_eval/run_text_benchmarks.sh" \
  --benchmark cvtg --stage all \
  --output-root /shared/eval/step10000_cvtg_new \
  --evaluation-python "$CVTG_PYTHON" \
  --evaluation-cache-dir "$CVTG_CACHE_DIR" \
  --clip-batch-size 16 \
  --no-hf-mirror
```

默认生成环境仍是 `i1_sft`；可用 `PYTHON_BIN` 指定生成解释器。
评分环境优先级：`--evaluation-python` > `CVTG_PYTHON` > 入口解释器。
缓存优先级：`--evaluation-cache-dir` > `CVTG_CACHE_DIR` > `~/.cache/cvtg`。
入口默认不启用 HF 镜像；需要时添加 `--use-hf-mirror`。

无需 `torchrun`，也**不要运行 `process.py`**。评分直接读取
`inputs/samples.jsonl` 和 `images/00000.png` 等平铺图片，不移动或重命名生成结果。
支持 `--limit` 子集；即使生成使用 rewritten prompt，评分始终使用对应官方原始 prompt。
评估 GPU 数量可以不同于生成；生成续跑仍须保持原分片数量。

小样本验收可在新目录使用 `--stage all --limit 8 --num-steps 2`；
先加 `--dry-run` 检查生成和评分命令。显式指定 GPU ID 的 dry-run 不加载模型或探测 CUDA。
每次显式评分会重新评分所有所选图片，不复用旧分片。

输出：

```text
eval_results/results.json          overall_results + area_results，兼容旧评分 JSON 字段
eval_results/results.jsonl         逐图官方 ID、原始 prompt、各指标、OCR 计数和逐词 NED
eval_results/results_summary.json  总分、checkpoint、图片数、prompt variant、评分模型等
eval_results/logs/worker_N.log     每张 GPU 的模型加载及逐图进度
```

模型缺失、图片损坏、OOM、推理异常和缺失/重复/非有限分数都会失败，不再记成 0 分。
任一 worker 失败会终止其余 worker，不替换已成功发布的评分文件。
Word Accuracy 按总词数计算，NED 按全部词计算；其他指标按全部图片平均，
不平均各 worker 的均值。小批大小变化可能引起正常的浮点舍入差异。

## 3. 直接运行旧评估命令

已通过旧 `process.py` 整理为 `CVTG/2/0.png` 等目录的全量图片仍然支持：

```bash
export CVTG_IMAGES=/shared/eval/legacy_cvtg_images

"$CVTG_PYTHON" "$I1/benchmark_eval/cvtg-2k/unified_metrics_eval.py" \
  --benchmark_dir "$I1/benchmark_eval/cvtg-2k/prompts" \
  --result_dir "$CVTG_IMAGES" \
  --output_file "$CVTG_IMAGES/results.json" \
  --cache_dir "$CVTG_CACHE_DIR" \
  --gpu-ids 0,1,2,3 \
  --clip-batch-size 16 \
  --no_hf_mirror
```

如果还没整理，仅对旧生成方式的全量平铺图片执行一次
`python process.py --root "$CVTG_IMAGES"`（会移动原图）。统一入口生成的图片应使用上一节，
或给上述直接命令添加 `--samples-file /path/to/run/inputs/samples.jsonl`，
同时将 `--result_dir` 指向该 run 的 `images/`。有 manifest 才支持显式子集；
旧目录缺图会报错，避免将不完整数据当成完整 benchmark。

直接 CLI 的 GPU 选择：`--gpu-ids` > `GPU_IDS` > `CUDA_VISIBLE_DEVICES` > PyTorch 自动探测。
未指定 ID 时会使用所有可见 GPU；若要单卡可设 `--gpu-ids 0`。
旧 CLI 保留默认启用镜像的行为，建议显式传 `--no_hf_mirror` 或 `--use_hf_mirror`。
`--device cpu` 为单进程调试选项；完整 XXL 评估不推荐在 CPU 上运行。
结果旁同时写入 `<输出名>.jsonl`、`<输出名>_summary.json` 和 `logs/worker_N.log`。

## 4. CPU 回归测试

仅需 Python 和 Pillow，不下载模型或运行 CUDA：

```bash
python -m unittest discover -s "$I1/benchmark_eval" -p 'test_text_benchmarks.py' -v
python -m unittest discover -s "$I1/benchmark_eval" -p 'test_cvtg_evaluation.py' -v
```
