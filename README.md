<p align="center">
  <img src="./asset/awm-icon.png" alt="AWM logo" width="96">
</p>

<!-- <h2 align="center">AWM: Accurate Weight-Matrix Fingerprint for Large Language Models</h2> -->

<h3 align="center">
<b>AWM: Accurate Weight-Matrix Fingerprint for Large Language Models</b>
<br>
<b>ICLR 2026</b>
</h3>

<p align="center">
  Training-free model fingerprint with LAP-aligned unbiased CKA
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2510.06738">
    <img src="https://img.shields.io/badge/arXiv-2510.06738-b31b1b?style=flat-square&logo=arxiv" alt="arXiv"></a>
  &nbsp;
  <a href="https://github.com/LUMIA-Group/AWM">
    <img src="https://img.shields.io/badge/GitHub-Project-181717?style=flat-square&logo=github" alt="GitHub"></a>
  &nbsp;
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  &nbsp;
  <img src="https://img.shields.io/badge/Method-Training--Free-2f855a?style=flat-square" alt="Training Free">
</p>
This repository contains the official implementation for the paper "[AWM: Accurate Weight-Matrix Fingerprint for Large Language Models](https://arxiv.org/abs/2510.06738)"

`AWM` is a fingerprinting method to determine whether one large language model (LLM) is derived from another base model. The method combines Linear Assignment Problem (LAP) based dimension alignment with unbiased CKA on attention weights, and is robust to common post-training changes.

## News
- [2026.1] AWM is accepted at ICLR 2026!
- [2025.10] Code and paper released. 

## Quick Feature Summary

| Feature Category | AWM Capability |
| - | - |
| **Goal** | Independency test of open-source LLMs based on their weights |
| **Method** | 1️⃣ Linear-Assignment-Problem-based (LAP-based) alignment <br> 2️⃣ Unbiased Central Kernel Alignment (UCKA) |
|**Detectable Weights**| ✅ Attention weights <br>✅ (informal) FFN weights|
| **Computation Cost** | ✅ Training-free <br> ✅ ~30s/pair on one RTX3090|
| **Accuracy** | ✅~0 similarity scores for independent models <br> ✅High similarity scores for correlated models|
| **Robustness to <br>training recipes** | ✅ Supervised fine-tuning (SFT) <br> ✅ Continual pre-training (CPT) <br> ✅ Reinforcement learning post-training (RL)<br> ✅ Multimodal tuning <br> ✅ Pruning <br> ✅ MoE upcycling |
|**Robustness to <br>weight manipulations**| ✅ Constant scaling <br> ✅ Permutaion matrices<br> ✅ Signature matrices <br> ✅ Orthogonal matrices|


## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
  - [A. Compare Two Local Models](#a-compare-two-local-models)
  - [B. Run a Predefined Experiment](#b-run-a-predefined-experiment)
- [Model Download](#model-download)
- [Experiment Suites](#experiment-suites)
- [Repository Structure](#repository-structure)
- [Citation](#citation)

## Installation

```bash
git clone https://github.com/LUMIA-Group/AWM
cd AWM
pip install -r requirements.txt
```

## Quick Start

### A. Compare Two Local Models

Download or place two model checkpoints (e.g. Llama-2-7b and CodeLlama-7b-hf) locally, then run:

```bash
python main.py \
  --model_paths ./checkpoints/Llama-2-7b ./checkpoints/CodeLlama-7b-hf \
  --device auto
```

`--device` supports `auto`, `cpu`, and `cuda`.

### B. Run a Predefined Experiment

```bash
python main.py --config sft_pairs --device auto
```

This mode uses model groups and analysis pairs defined in `configs.py`.

## Model Download

`AWM` provides two ways to prepare checkpoints.

1. Direct download with `huggingface-cli`:
```bash
huggingface-cli download meta-llama/Llama-2-7b --local-dir ./checkpoints/Llama-2-7b
huggingface-cli download codellama/CodeLlama-7b-hf --local-dir ./checkpoints/CodeLlama-7b-hf
```

2. Batch download with helper script:
```bash
chmod +x download_models.sh
./download_models.sh             # list supported config keys
./download_models.sh sft_pairs   # download models for one config
./download_models.sh all         # download all mapped models
```

Note: keep `CHECKPOINT_BASE_DIR` in `configs.py` and `download_models.sh` consistent with your local storage path.

## Experiment Suites

Predefined config keys:

- `sft_pairs`
- `cpt_pairs`
- `rl_pairs`
- `multimodal_pairs`
- `pruning_pairs`
- `all_moe_pairs`
- `sft_13b_7b_pairs`
- `independent_pairs`
- `independent_pairs_13b`

Suggested reproducibility flow:

1. Download models for one suite, e.g. `./download_models.sh cpt_pairs`.
2. Run analysis, e.g. `python main.py --config cpt_pairs --device auto`.
3. Compare summary scores in the console output (`Wq_weights`, `Wk_weights`, `Wq_Wk_weights`).

## Repository Structure

```text
AWM/
├── checkpoints/            # model checkpoints
├── asset/                  # static assets (logo/icons)
├── main.py                 # experiment entry point
├── similarity_metrics.py   # CKA, weight loading, LAP alignment
├── configs.py              # experiment definitions + model mapping
├── download_models.sh      # batch downloader for Hugging Face models
└── requirements.txt        # dependencies
```

## Citation
If you find `AWM` useful in your research or applications, we would appreciate it if you could cite our work:
```bibtex
@article{zeng2025awm,
  title={{AWM: Accurate Weight-Matrix Fingerprint for Large Language Models}},
  author={Boyi Zeng and Lin Chen and Ziwei He and Xinbing Wang and Zhouhan Lin},
  year={2025},
  journal={arXiv preprint arXiv:2510.06738},
}
```
