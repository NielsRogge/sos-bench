# Style over Substance: Failure modes of LLM judges in alignment benchmarking

![](./figures/mismo-fig.png)

This codebase stores the complete artifacts and describes how to reproduce or extend the results from the paper "Style over Substance: Failure modes of LLM judges in alignment benchmarking".

# List of Benchmarks

In this table, you can find the complete list of benchmarks we use in **SOS-Bench**, along with the codebase necessary to run them. Below, we describe how to work with each codebase.

| **Benchmark Name**                  | **Reference**                                           | **Test Set Size** | **Metric**                | **Factor** | **Eval Codebase** |
|---------------------------------|-----------------------------------------------------|---------------|-----------------------|--------|---------------|
| LiveBench-Coding                | https://huggingface.co/datasets/livebench/coding                    | 130           | Exact Match Acc       | WK     | LiveBench     |
| LiveBench-Data Analysis         | https://huggingface.co/datasets/livebench/data_analysis                    | 150           | Exact Match Acc       | WK     | LiveBench     |
| LiveBench-Instruction Following | https://huggingface.co/datasets/livebench/instruction_following                    | 200           | Exact Match Acc       | IF     | LiveBench     |
| LiveBench-Language              | https://huggingface.co/datasets/livebench/language                    | 140           | Exact Match Acc       | WK     | LiveBench     |
| LiveBench-Math                  | https://huggingface.co/datasets/livebench/math                    | 230           | Exact Match Acc       | WK     | LiveBench     |
| LiveBench-Reasoning             | https://huggingface.co/datasets/livebench/reasoning                    | 150           | Exact Match Acc       | WK     | LiveBench     |
| IFEval                          | https://huggingface.co/datasets/google/IFEval                    | 540           | Avg of Custom Metrics | IF     | Eleuther      |
| MATH Lvl 5                      | https://huggingface.co/datasets/AI-MO/aimo-validation-math-level-5                    | 1000          | Exact Match Acc       | WK     | Eleuther      |
| MuSR                            | https://huggingface.co/datasets/TAUR-Lab/MuSR                    | 750           | Acc                   | WK     | Eleuther      |
| GPQA                            | https://huggingface.co/datasets/Idavidrein/gpqa                    | 1250          | Acc                   | WK     | Eleuther      |
| MMLU-Pro                        | https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro                    | 12000         | Acc                   | WK     | Eleuther      |
| BBH                             | https://huggingface.co/datasets/maveriq/bigbenchhard                    | 6750          | Acc                   | WK     | Eleuther      |
| BeaverTails                     | https://huggingface.co/datasets/PKU-Alignment/BeaverTails                    | 1400          | Acc                   | Safety | Eleuther      |
| CDNA                            | https://huggingface.co/datasets/walledai/CDNA       | 2730          | Acc                   | Safety | Eleuther      |
| DTToxicity                      | https://huggingface.co/datasets/walledai/DTToxicity | 4800          | Acc                   | Safety | Eleuther      |
| JailbreakHub                    | https://huggingface.co/datasets/walledai/JailbreakHub                    | 15100         | Acc                   | Safety | Eleuther      |
| BBQ                             | https://huggingface.co/datasets/walledai/BBQ                    | 58500         | Acc                   | Safety | Eleuther      |
| WMDP                            | https://huggingface.co/datasets/cais/wmdp                   | 3670          | Inverse Acc           | Safety | Eleuther      |
| XSTest                          | https://huggingface.co/datasets/walledai/XSTest                    | 450           | Acc                   | Safety | Eleuther      |
| WildGuardTest                   | https://huggingface.co/datasets/walledai/WildGuardTest                    | 1730          | Acc                   | Safety | Eleuther      |
| Toxigen                         | https://huggingface.co/datasets/toxigen/toxigen-data                    | 9900          | Acc                   | Safety | Eleuther      |
| StrongREJECT                    | https://huggingface.co/datasets/AlignmentResearch/StrongREJECT                    | 310           | Acc                   | Safety | Eleuther      |
| SGXSTest                        | https://huggingface.co/datasets/walledai/SGXSTest   | 100           | Acc                   | Safety | Eleuther      |
| SaladBench                      | https://huggingface.co/datasets/walledai/SaladBench | 30400         | Acc                   | Safety | Eleuther      |

# List of Artifacts in this Repository

Here is a brief description of our result artifacts.

## Eleuther Results

```
Filenames: eleuther_wandb.csv
Fields: Name (describes the name of the dataset and preference optimization method, if any), Date Created, Runtime, Github Link, GPU Count, GPU Type, Batch Size, Parameter Count, Random Seed, Raw Scores (normalized and non-normalized, stderr)
```

## Arena-Hard-Auto Results

```
Filenames: arena_hard_auto.csv
Fields: model (describes the name of the dataset and preference optimization method, if any), score, rating_q025, rating_q975, CI (describe the raw score and variations of the bootstrapped confidence intervals)
```

## LiveBench Results

```
Filenames: livebench_groups.csv, livebench_tasks.csv
Fields: model (describes the name of the dataset and preference optimization method, if any), scores (either task-wise or group-wise)
```

# How to Run SOS-Bench

The entirety of SOS-Bench can be run as a two-stage process; the first set of benchmarks can be completed using a fork of the Eleuther AI Harness, and the second set can be run using the LiveBench codebase.

## Eleuther

1. `pip install lm_eval[wandb,vllm,math,ifeval], sentencepiece`
2. `python install_nltk_punkt.py`
3. Git clone our [Eleuther AI Harness fork](https://anonymous.4open.science/r/lm-evaluation-harness-24C3/README.md) which contains additional tasks
4. `cd lm-evaluation-harness`
5. `pip install -e .`
6. `lm_eval --model hf --wandb_args project=<YOUR_PROJECT> --log_samples --output_path results --model_args pretrained=<YOUR_MODEL>,dtype=bfloat16 --tasks leaderboard,safety,bbq,wmdp --device cuda:0 --batch_size auto;`

## LiveBench

1. Git clone the [LiveBench repository](https://github.com/livebench/livebench)
2. Follow the instructions provided in the repository readme.

# Citation
