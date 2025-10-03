# AWM <img src="asset/awm-icon.png" width="40" style="vertical-align: middle;">: Accurate Weight-Matrix Fingerprint for Large Language Models

This repository contains the official implementation for the paper "[AWM: Accurate Weight-Matrix Fingerprint for Large Language Models](https://arxiv.org/abs/2405.XXXXX)". AWM is a training-free fingerprinting method to determine if a Large Language Model (LLM) is derived from another base model.

Our method leverages the Linear Assignment Problem (LAP) and an unbiased Centered Kernel Alignment (CKA) to create a similarity metric that is highly robust to common post-training modifications and malicious manipulations, while exhibiting a near-zero risk of false positives.


## Project Structure

```
├── main.py         # Main script to run experiments
├── similarity_metrics.py   # Core functions for CKA, weight loading, and LAP alignment
├── configs.py              # Experiment configurations (model paths, comparison pairs)
├── download_models.sh      # Script to download models from Hugging Face Hub
└── requirements.txt        # Python dependencies
```

## Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/LUMIA-Group/AWM_Fingerprint.git
    cd AWM_Fingerprint
    ```


2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```


## Quick Start: Comparing Two Models

1.  **Download Models**: Use `huggingface-cli` to download the two models you want to compare. For example, Llama-2-7B and CodeLlama-7b.
    ```bash
    huggingface-cli download meta-llama/Llama-2-7b --local-dir ./checkpoints/Llama-2-7b
    huggingface-cli download codellama/CodeLlama-7b-hf --local-dir ./checkpoints/CodeLlama-7b-hf
    ```

2.  **Run the Analysis**: Use the `main.py` script with the `--model_paths` argument pointing to the two model directories.
    ```bash
    python main.py --model_paths ./checkpoints/Llama-2-7b ./checkpoints/CodeLlama-7b-hf
    ```
    The script will output the similarity score to the console.


## Model Download

We provide a convenient script to download all necessary model checkpoints from the Hugging Face Hub.

-   The default download directory is `./checkpoints/`, which is configured inside the `download_models.sh` script. You can change this path as needed.

-   First, make the script executable:
    ```bash
    chmod +x download_models.sh
    ```

-   To see available download configurations:
    ```bash
    ./download_models.sh
    ```

-   To download a **specific model**, you can use `huggingface-cli` directly. For example:
    ```bash
    huggingface-cli download meta-llama/Llama-2-7b --local-dir ./checkpoints/Llama-2-7b
    ```

-   To download **all models** required to reproduce the paper's experiments (Note: this will require significant disk space):
    ```bash
    ./download_models.sh all
    ```

-   To download all models for a **specific experiment configuration** (e.g., `sft_pairs`):
    ```bash
    ./download_models.sh sft_pairs
    ```

## Reproducing Paper Experiments

You can easily reproduce the experiments from the paper using the predefined configurations in `configs.py`.

1.  **Download Models**: Download the models required for the experiment you want to run. For example, to run the `sft_pairs` experiment, you can download the necessary models with:
    ```bash
    ./download_models.sh sft_pairs
    ```

2.  **Run the Experiment**: Use the `--config` flag to specify the desired experiment.
    ```bash
    python main.py --config sft_pairs
    ```

    Available configurations include: `sft_pairs`, `cpt_pairs`, `rl_pairs`, `multimodal_pairs`, `pruning_pairs`, `all_moe_pairs` (for upcycling), `independent_pairs`, and more. See `configs.py` for the full list.

## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@article{zeng2025awm,
  title={{AWM: Accurate Weight-Matrix Fingerprint for Large Language Models}},
  author={Boyi Zeng and Lin Chen and Ziwei He and Xinbing Wang and Zhouhan Lin},
  year={2025},
  journal={arXiv preprint},
}
```
