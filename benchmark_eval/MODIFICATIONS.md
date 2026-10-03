# Modifications to Official Evaluation Code

We take the core evaluation code from the official codebases with minimal modifications that do not influence their functionality. This file documents those modifications.

## DPG-Bench
The core evaluation code is [compute_dpg_bench.py](dpg_bench/compute_dpg_bench.py).<br>
We change the default value of the `--csv` argument because the script is run from a different working directory.
```diff
    parser.add_argument(
        "--csv",
        type=str,
-       default='./dpg_bench/dpg_bench.csv',
+       default='dpg_bench.csv',
    )
```

## PRISM-Bench
The core evaluation code is [evaluation/eval_qwen25.py](prism-bench/evaluation/eval_qwen25.py).<br>
We keep it intact.

## LongText-Bench
The core evaluation code is [evaluate_text_reward.py](longtext/evaluate_text_reward.py).<br>
We add an optional `--prompt_file` manifest for subset evaluation, parse
`--global_seed` as an integer, bind each distributed worker to `LOCAL_RANK`, and
destroy its process group after writing results. OCR prompts, model settings,
and the official `summary_scores.py` scoring formula are unchanged. The
[text benchmark runner](TEXT_BENCHMARKS.md) validates complete result coverage
before invoking the official summary script.

## GenEval
The core evaluation code is [evaluation/evaluate_images.py](geneval/evaluation/evaluate_images.py).<br>
We modify the metadata loading code so that the metadata files do not have to be added to the generated image folders.
```diff
-   for subfolder in os.listdir(args.imagedir):
-       folderpath = os.path.join(args.imagedir, subfolder)
-       if not os.path.isdir(folderpath) or not subfolder.isdigit():
-           continue
-       with open(os.path.join(folderpath, "metadata.jsonl")) as fp:
-           metadata = json.load(fp)
+   with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "evaluation_metadata.jsonl")) as fp:
+       metadatas = [json.loads(line) for line in fp]
+   subfolders = [sf for sf in os.listdir(args.imagedir) 
+                 if os.path.isdir(os.path.join(args.imagedir, sf)) and sf.isdigit()]
+   for subfolder in subfolders:
+       folderpath = os.path.join(args.imagedir, subfolder)
+       metadata = metadatas[int(subfolder)]
```

## CVTG-2K
The core evaluation code is [unified_metrics_eval.py](cvtg-2k/unified_metrics_eval.py).<br>
We keep it intact.

## BizGenEval
The core evaluator remains in the sibling `BizGenEval` checkout and is invoked
without modification. The integration scripts in [bizgeneval](bizgeneval)
prepare filenames expected by the official evaluator, select the closest native
i1 SFT aspect-ratio bucket for each SFT prompt, run the starting checkpoint at
1024x1024, invoke the official evaluation and summary modules, and compare their
summary CSVs. The SFT checkpoint is generated before the starting checkpoint.

The PyTorch inference entry point accepts optional per-prompt height and width
fields from JSONL so all aspect ratios can be generated in one model-loading
session per GPU worker.
