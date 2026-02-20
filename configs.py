"""
Experiment Configuration File
Contains model paths and analysis pairs for different experimental setups.
"""
import os
from itertools import combinations

# --- CHECKPOINT DIRECTORY ---
# Please set this variable to the directory containing your model checkpoints.
# The script will look for model folders inside this directory.
# Example: "/path/to/my/checkpoints"
CHECKPOINT_BASE_DIR = "./checkpoints"

# --- EXPERIMENT CONFIGURATIONS ---
# The following dictionaries define the model paths and comparison pairs for
# each experiment discussed in the paper.

# SFT Model Comparison (7B and 13B)
SFT_13B_7B_PAIRS_CONFIG = {
    "model_paths": {
        "llama-2-7b": "Llama-2-7B-fp16",
        "Llama-2-7B-Chat-fp16": "Llama-2-7B-Chat-fp16",
        "llama-2-ko-7b": "llama-2-ko-7b",
        "vicuna-7b-v1.5": "vicuna-7b-v1.5",
        "llama-2-7b-finance": "llama-2-7b-finance",
        "selfrag_llama2_7b": "selfrag_llama2_7b",
        "LLaMA-2-7B-32K": "LLaMA-2-7B-32K",
        "WizardMath-7B-V1.0": "WizardMath-7B-V1.0",
        "llama-2-7b-guanaco": "llama-2-7b-guanaco",
        
        "llama-2-13b": "Llama-2-13b-hf",
        "llama2-13b-orca-8k-3319": "llama2-13b-orca-8k-3319",
        "vicuna-13b-v1.5": "vicuna-13b-v1.5",
        "Nous-Hermes-Llama2-13b": "Nous-Hermes-Llama2-13b",
        "LLaMA2-13B-Estopia": "LLaMA2-13B-Estopia",
        "firefly-llama2-13b": "firefly-llama2-13b",
        "llama-2-koen-13b": "llama-2-koen-13b",
        "selfrag_llama2_13b": "selfrag_llama2_13b",
        "Llama-2-13B-Chat-fp16": "Llama-2-13B-Chat-fp16",
    },
    "analysis_pairs": [
        # 8 pairs with llama-2-7b as model1
        {"label": "llama-2-7b vs Llama-2-7B-Chat-fp16", "weights1_name": "llama-2-7b", "weights2_name": "Llama-2-7B-Chat-fp16"},
        {"label": "llama-2-7b vs llama-2-ko-7b", "weights1_name": "llama-2-7b", "weights2_name": "llama-2-ko-7b"},
        {"label": "llama-2-7b vs vicuna-7b-v1.5", "weights1_name": "llama-2-7b", "weights2_name": "vicuna-7b-v1.5"},
        {"label": "llama-2-7b vs llama-2-7b-finance", "weights1_name": "llama-2-7b", "weights2_name": "llama-2-7b-finance"},
        {"label": "llama-2-7b vs selfrag_llama2_7b", "weights1_name": "llama-2-7b", "weights2_name": "selfrag_llama2_7b"},
        {"label": "llama-2-7b vs LLaMA-2-7B-32K", "weights1_name": "llama-2-7b", "weights2_name": "LLaMA-2-7B-32K"},
        {"label": "llama-2-7b vs WizardMath-7B-V1.0", "weights1_name": "llama-2-7b", "weights2_name": "WizardMath-7B-V1.0"},
        {"label": "llama-2-7b vs llama-2-7b-guanaco", "weights1_name": "llama-2-7b", "weights2_name": "llama-2-7b-guanaco"},
        
        # 8 pairs with llama-2-13b as model1
        {"label": "llama-2-13b vs vicuna-13b-v1.5", "weights1_name": "llama-2-13b", "weights2_name": "vicuna-13b-v1.5"},
        {"label": "llama-2-13b vs Nous-Hermes-Llama2-13b", "weights1_name": "llama-2-13b", "weights2_name": "Nous-Hermes-Llama2-13b"},
        {"label": "llama-2-13b vs LLaMA2-13B-Estopia", "weights1_name": "llama-2-13b", "weights2_name": "LLaMA2-13B-Estopia"},
        {"label": "llama-2-13b vs firefly-llama2-13b", "weights1_name": "llama-2-13b", "weights2_name": "firefly-llama2-13b"},
        {"label": "llama-2-13b vs llama-2-koen-13b", "weights1_name": "llama-2-13b", "weights2_name": "llama-2-koen-13b"},
        {"label": "llama-2-13b vs selfrag_llama2_13b", "weights1_name": "llama-2-13b", "weights2_name": "selfrag_llama2_13b"},
        {"label": "llama-2-13b vs llama2-13b-orca-8k-3319", "weights1_name": "llama-2-13b", "weights2_name": "llama2-13b-orca-8k-3319"},
        {"label": "llama-2-13b vs Llama-2-13B-Chat-fp16", "weights1_name": "llama-2-13b", "weights2_name": "Llama-2-13B-Chat-fp16"},
    ]
}


# Independent Models Comparison (7B)
INDEPENDENT_PAIRS_CONFIG = {
    "model_paths": {
        "mpt-7b": "mpt-7b",
        "Baichuan-7B": "Baichuan-7B",        
        "openllama_7b": "open_llama_7b", 
        "llama-7b": "llama-7b",
        "openllama2_7b": "open_llama_7b_v2",
        "llama-2-7b": "Llama-2-7B-fp16",
        "internlm-7b": "internlm-7b",   
        "Mistral-7B": "Mistral-7B-v0.1",
        "OLMo-7B": "OLMo-7B",
        "Qwen-7B": "Qwen-7B"        
    },
    "analysis_pairs": []  # Dynamically generated to include all unique pairs
}


# Comprehensive MoE Model Comparison (Upcycling)
ALL_MOE_PAIRS_CONFIG = {
    "model_paths": {
        "LLaMA-MoE-v1-3_5B-4_16-sft": "LLaMA-MoE-v1-3_5B-4_16-sft",
        "LLaMA-MoE-v1-3_5B-2_8-sft": "LLaMA-MoE-v1-3_5B-2_8-sft",
        "LLaMA-MoE-v1-3_0B-2_16-sft": "LLaMA-MoE-v1-3_0B-2_16-sft",
        "LLaMA-MoE-v1-3_5B-2_8": "LLaMA-MoE-v1-3_5B-2_8",
        "LLaMA-MoE-v1-3_0B-2_16": "LLaMA-MoE-v1-3_0B-2_16",
        "LLaMA-MoE-v1-3_5B-4_16": "LLaMA-MoE-v1-3_5B-4_16",
        "MiniCPM-2B": "MiniCPM-2B-sft-bf16",
        "MiniCPM-MoE-8x2B": "MiniCPM-MoE-8x2B",
        "Qwen1.5-MoE-A2.7B": "Qwen1.5-MoE-A2.7B",
        "Qwen-1_8B": "Qwen-1_8B",
        "Mistral-7B": "Mistral-7B-v0.1",
        "Mixtral-8x7B": "Mixtral-8x7B-v0.1",
        "llama_2_7b": "Llama-2-7B-fp16",
    },
    "analysis_pairs": [
        {"label": "LLaMA-MoE-v1-3_5B-4_16-sft vs llama_2_7b", "weights1_name": "LLaMA-MoE-v1-3_5B-4_16-sft", "weights2_name": "llama_2_7b"}, 
        {"label": "LLaMA-MoE-v1-3_5B-2_8-sft vs llama_2_7b", "weights1_name": "LLaMA-MoE-v1-3_5B-2_8-sft", "weights2_name": "llama_2_7b"}, 
        {"label": "LLaMA-MoE-v1-3_0B-2_16-sft vs llama_2_7b", "weights1_name": "LLaMA-MoE-v1-3_0B-2_16-sft", "weights2_name": "llama_2_7b"}, 
        {"label": "LLaMA-MoE-v1-3_5B-4_16 vs llama_2_7b", "weights1_name": "LLaMA-MoE-v1-3_5B-4_16", "weights2_name": "llama_2_7b"},
        {"label": "LLaMA-MoE-v1-3_0B-2_16 vs llama_2_7b", "weights1_name": "LLaMA-MoE-v1-3_0B-2_16", "weights2_name": "llama_2_7b"},
        {"label": "LLaMA-MoE-v1-3_5B-2_8 vs llama_2_7b", "weights1_name": "LLaMA-MoE-v1-3_5B-2_8", "weights2_name": "llama_2_7b"},
        {"label": "Qwen-1_8B vs Qwen1.5-MoE-A2.7B", "weights1_name": "Qwen-1_8B", "weights2_name": "Qwen1.5-MoE-A2.7B"},
        {"label": "MiniCPM-2B vs MiniCPM-MoE-8x2B", "weights1_name": "MiniCPM-2B", "weights2_name": "MiniCPM-MoE-8x2B"},
        {"label": "Mistral-7B vs Mixtral-8x7B", "weights1_name": "Mistral-7B", "weights2_name": "Mixtral-8x7B"},
        {"label": "Llama-2-7B vs LLaMA-MoE-v1-3_5B-4_16", "weights1_name": "llama_2_7b", "weights2_name": "LLaMA-MoE-v1-3_5B-4_16"},
    ]
}

# Independent Models Comparison (13B)
INDEPENDENT_PAIRS_CONFIG_13B = {
    "model_paths": {
        "open_llama_13b": "open_llama_13b",
        "plamo-13b": "plamo-13b",
        "llama-13b": "llama-13b",
        "Llama-2-13b": "Llama-2-13b-hf",
        "Baichuan-13B": "Baichuan-13B-Base",
        "Qwen-14B": "Qwen-14B",
        "jais-13b": "jais-13b",
        "OLMo-2-1124-13B": "OLMo-2-1124-13B",
        "Baichuan2-13B-Base": "Baichuan2-13B-Base",
        "Qwen3-14B": "Qwen3-14B",
    },
    "analysis_pairs": [] # Dynamically generated
}

# Continued Pre-Training (CPT) Model Comparison
CPT_PAIRS_CONFIG = {
    "model_paths": {
        "llama-2-7b": "Llama-2-7B-fp16",
        "CodeLlama-7b-hf": "CodeLlama-7b-hf",
        "llemma_7b": "llemma_7b",
        "CodeLlama-7b-Python-hf": "CodeLlama-7b-Python-hf",
        "gemma-2b": "gemma-2b",
        "codegemma-2b": "codegemma-2b",
        "gemma-7b": "gemma-7b",
        "codegemma-7b": "codegemma-7b",
        "qwen2.5_7b": "qwen2.5-7b",
        "Qwen2.5-Coder-7B": "Qwen2.5-Coder-7B",
        "Qwen2-7b": "qwen2_7b",
        "Qwen2-Math-7B": "Qwen2-Math-7B",
        "Qwen2.5-Math-7B": "Qwen2.5-Math-7B",
        "CodeLlama-70b-hf": "CodeLlama-70b-hf",
        "CodeLlama-70b-Python-hf": "CodeLlama-70b-Python-hf",
        "Llama-2-70B-fp16": "Llama-2-70B-fp16",
        "Llama-3-8B": "Meta-Llama-3-8B",
    },
    "analysis_pairs": [
        {"label": "Llama-2-7B vs CodeLlama-7B", "weights1_name": "llama-2-7b", "weights2_name": "CodeLlama-7b-hf"},
        {"label": "Llama-2-7B vs Llemma-7B", "weights1_name": "llama-2-7b", "weights2_name": "llemma_7b"},
        {"label": "Llama-2-7B vs CodeLlama-7B-Python", "weights1_name": "llama-2-7b", "weights2_name": "CodeLlama-7b-Python-hf"},
        {"label": "Gemma-2B vs CodeGemma-2B", "weights1_name": "gemma-2b", "weights2_name": "codegemma-2b"},
        {"label": "Gemma-7B vs CodeGemma-7B", "weights1_name": "gemma-7b", "weights2_name": "codegemma-7b"},
        {"label": "Qwen2.5-7B vs Qwen2.5-Coder-7B", "weights1_name": "qwen2.5_7b", "weights2_name": "Qwen2.5-Coder-7B"},
        {"label": "Llama-2-70B vs CodeLlama-70B-Python", "weights1_name": "Llama-2-70B-fp16", "weights2_name": "CodeLlama-70b-Python-hf"},
        {"label": "Llama-2-70B vs CodeLlama-70B", "weights1_name": "Llama-2-70B-fp16", "weights2_name": "CodeLlama-70b-hf"},
        {"label": "Qwen2.5-7B vs Qwen2.5-Math-7B", "weights1_name": "qwen2.5_7b", "weights2_name": "Qwen2.5-Math-7B"},
        {"label": "Qwen2-7B vs Qwen2-Math-7B", "weights1_name": "Qwen2-7b", "weights2_name": "Qwen2-Math-7B"},
    ]
}

# Supervised Fine-Tuning (SFT) Model Comparison
SFT_PAIRS_CONFIG = {
    "model_paths": {
        "llama-2-7b": "Llama-2-7B-fp16",
        "vicuna-7b-v1.5": "vicuna-7b-v1.5",
        "llama-2-7b-finance": "llama-2-7b-finance",
        "selfrag_llama2_7b": "selfrag_llama2_7b",
        "LLaMA-2-7B-32K": "LLaMA-2-7B-32K",
        "WizardMath-7B-V1.0": "WizardMath-7B-V1.0",
        "llama-2-7b-guanaco": "llama-2-7b-guanaco",
        "Llama-2-13b": "Llama-2-13b-hf",
        "vicuna-13b-v1.5": "vicuna-13b-v1.5",
        "Nous-Hermes-Llama2-13b": "Nous-Hermes-Llama2-13b",
        "LLaMA2-13B-Estopia": "LLaMA2-13B-Estopia",
        "firefly-llama2-13b": "firefly-llama2-13b"
    },
    "analysis_pairs": [
        {"label": "Llama-2-7B vs Vicuna-7B-v1.5", "weights1_name": "llama-2-7b", "weights2_name": "vicuna-7b-v1.5"},
        {"label": "Llama-2-7B vs Llama-2-7B-Finance", "weights1_name": "llama-2-7b", "weights2_name": "llama-2-7b-finance"},
        {"label": "Llama-2-7B vs SelfRAG-Llama2-7B", "weights1_name": "llama-2-7b", "weights2_name": "selfrag_llama2_7b"},
        {"label": "Llama-2-7B vs LLaMA-2-7B-32K", "weights1_name": "llama-2-7b", "weights2_name": "LLaMA-2-7B-32K"},
        {"label": "Llama-2-7B vs WizardMath-7B", "weights1_name": "llama-2-7b", "weights2_name": "WizardMath-7B-V1.0"},
        {"label": "Llama-2-7B vs Llama-2-7B-Guanaco", "weights1_name": "llama-2-7b", "weights2_name": "llama-2-7b-guanaco"},
        {"label": "Llama-2-13B vs Vicuna-13B-v1.5", "weights1_name": "Llama-2-13b", "weights2_name": "vicuna-13b-v1.5"},
        {"label": "Llama-2-13B vs Nous-Hermes-Llama2-13B", "weights1_name": "Llama-2-13b", "weights2_name": "Nous-Hermes-Llama2-13b"},
        {"label": "Llama-2-13B vs LLaMA2-13B-Estopia", "weights1_name": "Llama-2-13b", "weights2_name": "LLaMA2-13B-Estopia"},
        {"label": "Llama-2-13B vs Firefly-Llama2-13B", "weights1_name": "Llama-2-13b", "weights2_name": "firefly-llama2-13b"},
    ]
}

# Multimodal Model Comparison
MULTIMODAL_PAIRS_CONFIG = {
    "model_paths": {
        "llama-2-7b": "Llama-2-7B-fp16",
        "llava-v1.5-7b": "llava-v1.5-7b",
        "Video-LLaVA-7B-hf": "Video-LLaVA-7B-hf",
        "Qwen-7b": "Qwen-7B",
        "Qwen-VL": "Qwen-VL",
        "Qwen-Audio": "Qwen-Audio",
        "Qwen2.5-3B": "Qwen2.5-3B",
        "Qwen2.5-VL-3B-Instruct": "Qwen2.5-VL-3B-Instruct",
        "llama3-llava-next-8b-hf": "llama3-llava-next-8b-hf",
        "Qwen2-7b": "qwen2_7b",
        "Qwen2-VL-7B-Instruct": "Qwen2-VL-7B-Instruct",
        "Qwen2-Audio-7B": "Qwen2-Audio-7B",
        "Llama-3-8B": "Meta-Llama-3-8B",
        "qwen2.5_7b": "qwen2.5-7b",
        "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B-Instruct",
        "Llama-2-13b": "Llama-2-13b-hf",
        "llava-v1.5-13b": "llava-v1.5-13b"
    },
    "analysis_pairs": [
        {"label": "Llama-2-7B vs LLaVA-v1.5-7B", "weights1_name": "llama-2-7b", "weights2_name": "llava-v1.5-7b"},
        {"label": "Llama-2-7B vs Video-LLaVA-7B", "weights1_name": "llama-2-7b", "weights2_name": "Video-LLaVA-7B-hf"},
        {"label": "Qwen2-7B vs Qwen2-VL-7B", "weights1_name": "Qwen2-7b", "weights2_name": "Qwen2-VL-7B-Instruct"},
        {"label": "Qwen-7B vs Qwen2-Audio-7B", "weights1_name": "Qwen-7b", "weights2_name": "Qwen2-Audio-7B"},
        {"label": "Llama3-LLaVA-Next-8B vs Llama-3-8B", "weights1_name": "llama3-llava-next-8b-hf", "weights2_name": "Llama-3-8B"},
        {"label": "Qwen2.5-7B vs Qwen2.5-VL-7B", "weights1_name": "qwen2.5_7b", "weights2_name": "Qwen2.5-VL-7B-Instruct"},
        {"label": "Qwen2.5-3B vs Qwen2.5-VL-3B-Instruct", "weights1_name": "Qwen2.5-3B", "weights2_name": "Qwen2.5-VL-3B-Instruct"},
        {"label": "Qwen-7B vs Qwen-Audio", "weights1_name": "Qwen-7b", "weights2_name": "Qwen-Audio"},
        {"label": "Qwen-7B vs Qwen-VL", "weights1_name": "Qwen-7b", "weights2_name": "Qwen-VL"},
        {"label": "Llama-2-13B vs LLaVA-v1.5-13B", "weights1_name": "Llama-2-13b", "weights2_name": "llava-v1.5-13b"},
    ]
}

# Reinforcement Learning (RL) Model Comparison
RL_PAIRS_CONFIG = {
    "model_paths": {
        "chatglm-6b": "chatglm-6b",
        "chatglm-fitness-RLHF": "chatglm-fitness-RLHF",
        "Qwen3-4B-Base": "Qwen3-4B-Base",
        "Qwen3_Medical_GRPO": "Qwen3_Medical_GRPO",
        "Mistral-7B-v0.1": "Mistral-7B-v0.1",
        "Nous-Hermes-2-Mistral-7B-DPO": "Nous-Hermes-2-Mistral-7B-DPO",
        "dolphin-2.6-mistral-7b-dpo": "dolphin-2.6-mistral-7b-dpo",
        "Mixtral-8x7B-v0.1": "Mixtral-8x7B-v0.1",
        "Nous-Hermes-2-Mixtral-8x7B-DPO": "Nous-Hermes-2-Mixtral-8x7B-DPO",
        "MiniCPM-2B": "MiniCPM-2B-sft-bf16",
        "MiniCPM-2B-dpo": "MiniCPM-2B-dpo-bf16",
        "Llama-3-8B": "Meta-Llama-3-8B",
        "LLaMA3-iterative-DPO": "LLaMA3-iterative-DPO-final",
        "Qwen2.5-1.5B": "Qwen2.5-1.5B",
        "Open-Reasoner-Zero-1.5B": "Open-Reasoner-Zero-1.5B",
        "Nemotron-Research-Reasoning-Qwen-1.5B": "Nemotron-Research-Reasoning-Qwen-1.5B",
        "qwen2.5_7b": "qwen2.5-7b",
        "Open-Reasoner-Zero-7B": "Open-Reasoner-Zero-7B",
        "hh_rlhf_rm_open_llama_3b": "hh_rlhf_rm_open_llama_3b",
        "open_llama_3b": "open_llama_3b",
    },
    "analysis_pairs": [
        {"label": "Qwen2.5-1.5B vs Open-Reasoner-Zero-1.5B", "weights1_name": "Qwen2.5-1.5B", "weights2_name": "Open-Reasoner-Zero-1.5B"},
        {"label": "Qwen2.5-7B vs Open-Reasoner-Zero-7B", "weights1_name": "qwen2.5_7b", "weights2_name": "Open-Reasoner-Zero-7B"},
        {"label": "ChatGLM-6B vs ChatGLM-fitness-RLHF", "weights1_name": "chatglm-6b", "weights2_name": "chatglm-fitness-RLHF"},
        {"label": "Qwen3-4B-Base vs Qwen3_Medical_GRPO", "weights1_name": "Qwen3-4B-Base", "weights2_name": "Qwen3_Medical_GRPO"},
        {"label": "Mistral-7B-v0.1 vs Nous-Hermes-2-Mistral-7B-DPO", "weights1_name": "Mistral-7B-v0.1", "weights2_name": "Nous-Hermes-2-Mistral-7B-DPO"},
        {"label": "Mistral-7B-v0.1 vs dolphin-2.6-mistral-7b-dpo", "weights1_name": "Mistral-7B-v0.1", "weights2_name": "dolphin-2.6-mistral-7b-dpo"},
        {"label": "Mixtral-8x7B-v0.1 vs Nous-Hermes-2-Mixtral-8x7B-DPO", "weights1_name": "Mixtral-8x7B-v0.1", "weights2_name": "Nous-Hermes-2-Mixtral-8x7B-DPO"},
        {"label": "MiniCPM-2B vs MiniCPM-2B-dpo", "weights1_name": "MiniCPM-2B", "weights2_name": "MiniCPM-2B-dpo"},
        {"label": "Llama-3-8B vs LLaMA3-iterative-DPO", "weights1_name": "Llama-3-8B", "weights2_name": "LLaMA3-iterative-DPO"},
        {"label": "hh_rlhf_rm_open_llama_3b vs open_llama_3b", "weights1_name": "hh_rlhf_rm_open_llama_3b", "weights2_name": "open_llama_3b"},
    ]
}


# Pruning Alignment Experiment
PRUNING_PAIRS_CONFIG = {
    "model_paths": {
        "Llama-3.2-1B": "Llama-3.2-1B",
        "Llama-3.2-3B": "Llama-3.2-3B",
        "Llama-3-8B": "Meta-Llama-3-8B",
        "Llama-3.1-Minitron-4B-Width-Base": "Llama-3.1-Minitron-4B-Width-Base",
        "Llama-3.1-Minitron-4B-Depth-Base": "Llama-3.1-Minitron-4B-Depth-Base",
        "Sheared-LLaMA-1.3B": "Sheared-LLaMA-1.3B",
        "Sheared-LLaMA-2.7B": "Sheared-LLaMA-2.7B",
        "Sheared-LLaMA-1.3B-Pruned": "Sheared-LLaMA-1.3B-Pruned",
        "Sheared-LLaMA-2.7B-Pruned": "Sheared-LLaMA-2.7B-Pruned",
        "Sheared-LLaMA-1.3B-ShareGPT": "Sheared-LLaMA-1.3B-ShareGPT",
        "Sheared-LLaMA-2.7B-ShareGPT": "Sheared-LLaMA-2.7B-ShareGPT",
        "llama-2-7b": "Llama-2-7B-fp16",
    },
    "analysis_pairs": [
        {"label": "Llama-3.2-1B vs Llama-3-8B", "weights1_name": "Llama-3.2-1B", "weights2_name": "Llama-3-8B"},
        {"label": "Llama-3-8B vs Llama-3.1-Minitron-4B-Width-Base", "weights1_name": "Llama-3-8B", "weights2_name": "Llama-3.1-Minitron-4B-Width-Base"},
        {"label": "Llama-3-8B vs Llama-3.1-Minitron-4B-Depth-Base", "weights1_name": "Llama-3-8B", "weights2_name": "Llama-3.1-Minitron-4B-Depth-Base"},
        {"label": "Llama-3.2-3B vs Llama-3-8B", "weights1_name": "Llama-3.2-3B", "weights2_name": "Llama-3-8B"},
        {"label": "Sheared-LLaMA-1.3B vs llama-2-7b", "weights1_name": "Sheared-LLaMA-1.3B", "weights2_name": "llama-2-7b"},
        {"label": "Sheared-LLaMA-1.3B-Pruned vs llama-2-7b", "weights1_name": "Sheared-LLaMA-1.3B-Pruned", "weights2_name": "llama-2-7b"},
        {"label": "Sheared-LLaMA-1.3B-ShareGPT vs llama-2-7b", "weights1_name": "Sheared-LLaMA-1.3B-ShareGPT", "weights2_name": "llama-2-7b"},
        {"label": "Sheared-LLaMA-2.7B vs llama-2-7b", "weights1_name": "Sheared-LLaMA-2.7B", "weights2_name": "llama-2-7b"},
        {"label": "Sheared-LLaMA-2.7B-Pruned vs llama-2-7b", "weights1_name": "Sheared-LLaMA-2.7B-Pruned", "weights2_name": "llama-2-7b"},
        {"label": "Sheared-LLaMA-2.7B-ShareGPT vs llama-2-7b", "weights1_name": "Sheared-LLaMA-2.7B-ShareGPT", "weights2_name": "llama-2-7b"},
    ]
}

# --- Hugging Face Model ID Mapping ---
# Maps the local folder names used in the configurations above to their
# corresponding Hugging Face Hub repository IDs. This is used by the
# `download_models.sh` script.
# Note: Some mappings are best-effort based on available models in `models.txt`.
# Please verify and add any missing models if needed.
MODEL_HF_MAP = {
    # SFT / Base Models
    "Llama-2-7B-fp16": "meta-llama/Llama-2-7b",
    "Llama-2-7B-Chat-fp16": "meta-llama/Llama-2-7b-chat",
    "llama-2-ko-7b": "beomi/llama-2-ko-7b",
    "vicuna-7b-v1.5": "lmsys/vicuna-7b-v1.5",
    "llama-2-7b-finance": "Abira1/llama-2-7b-finance",
    "selfrag_llama2_7b": "selfrag/selfrag_llama2_7b",
    "LLaMA-2-7B-32K": "togethercomputer/LLaMA-2-7B-32K",
    "WizardMath-7B-V1.0": "WizardLMTeam/WizardMath-7B-V1.0",
    "llama-2-7b-guanaco": "mlabonne/llama-2-7b-guanaco",
    "Llama-2-13b-hf": "meta-llama/Llama-2-13b",
    "llama2-13b-orca-8k-3319": "OpenAssistant/llama2-13b-orca-8k-3319",
    "vicuna-13b-v1.5": "lmsys/vicuna-13b-v1.5",
    "Nous-Hermes-Llama2-13b": "NousResearch/Nous-Hermes-Llama2-13b",
    "LLaMA2-13B-Estopia": "KoboldAI/LLaMA2-13B-Estopia",
    "firefly-llama2-13b": "YeungNLP/firefly-llama2-13b",
    "llama-2-koen-13b": "beomi/llama-2-koen-13b",
    "selfrag_llama2_13b": "selfrag/selfrag_llama2_13b",
    "Llama-2-13B-Chat-fp16": "meta-llama/Llama-2-13b-chat",
    "Llama-2-70B-fp16": "meta-llama/Llama-2-70b",
    "Meta-Llama-3-8B": "meta-llama/Llama-3.1-8B", 

    # Independent Models
    "mpt-7b": "mosaicml/mpt-7b",
    "Baichuan-7B": "baichuan-inc/Baichuan-7B",
    "open_llama_7b_v2": "openlm-research/open_llama_7b_v2",
    "llama-7b": "huggyllama/llama-7b",
    "internlm-7b": "internlm/internlm-7b", 
    "Mistral-7B-v0.1": "mistralai/Mistral-7B-v0.1", 
    "Qwen-7B": "Qwen/Qwen-7B",
    "open_llama_13b": "openlm-research/open_llama_13b",
    "plamo-13b": "pfnet/plamo-13b",
    "llama-13b": "huggyllama/llama-13b",
    "Baichuan-13B-Base": "baichuan-inc/Baichuan13B-Base",
    "Qwen-14B": "Qwen/Qwen-14B",
    "jais-13b": "inceptionai/jais-13b",
    "OLMo-2-1124-13B": "allenai/OLMo-2-1124-13B",
    "Baichuan2-13B-Base": "baichuan-inc/Baichuan2-13B-Base",
    "Qwen3-14B": "Qwen/Qwen3-14B",

    # MoE / Upcycling Models
    "LLaMA-MoE-v1-3_5B-4_16-sft": "llama-moe/LLaMA-MoE-v1-3_5B-4_16-sft",
    "LLaMA-MoE-v1-3_5B-2_8-sft": "llama-moe/LLaMA-MoE-v1-3_5B-2_8-sft",
    "LLaMA-MoE-v1-3_0B-2_16-sft": "llama-moe/LLaMA-MoE-v1-3_0B-2_16-sft",
    "LLaMA-MoE-v1-3_5B-2_8": "llama-moe/LLaMA-MoE-v1-3_5B-2_8",
    "LLaMA-MoE-v1-3_0B-2_16": "llama-moe/LLaMA-MoE-v1-3_0B-2_16",
    "LLaMA-MoE-v1-3_5B-4_16": "llama-moe/LLaMA-MoE-v1-3_5B-4_16",
    "MiniCPM-2B-sft-bf16": "openbmb/MiniCPM-2B-sft-bf16",
    "MiniCPM-MoE-8x2B": "openbmb/MiniCPM-MoE-8x2B",
    "Qwen1.5-MoE-A2.7B": "Qwen/Qwen1.5-MoE-A2.7B",
    "Qwen-1_8B": "Qwen/Qwen-1_8B",
    "Mixtral-8x7B-v0.1": "mistralai/Mixtral-8x7B-v0.1",

    # CPT Models
    "CodeLlama-7b-hf": "codellama/CodeLlama-7b-hf",
    "llemma_7b": "EleutherAI/llemma_7b",
    "CodeLlama-7b-Python-hf": "meta-llama/CodeLlama-7b-Python-hf",
    "gemma-2b": "google/gemma-2b",
    "codegemma-2b": "google/codegemma-2b",
    "gemma-7b": "google/gemma-7b",
    "codegemma-7b": "google/codegemma-7b",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B", 
    "Qwen2.5-Coder-7B": "Qwen/Qwen2.5-Coder-7B",
    "qwen2_7b": "Qwen/Qwen2-7B", 
    "Qwen2-Math-7B": "Qwen/Qwen2-Math-7B",
    "Qwen2.5-Math-7B": "Qwen/Qwen2.5-Math-7B",
    "CodeLlama-70b-hf": "meta-llama/CodeLlama-70b-hf",
    "CodeLlama-70b-Python-hf": "meta-llama/CodeLlama-70b-Python-hf",

    # Multimodal Models
    "llava-v1.5-7b": "liuhaotian/llava-v1.5-7b",
    "Video-LLaVA-7B-hf": "LanguageBind/Video-LLaVA-7B-hf",
    "Qwen-VL": "Qwen/Qwen-VL",
    "Qwen-Audio": "Qwen/Qwen-Audio",
    "Qwen2.5-3B": "Qwen/Qwen2.5-3B",
    "Qwen2.5-VL-3B-Instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
    "llama3-llava-next-8b-hf": "llava-hf/llama3-llava-next-8b-hf",
    "Qwen2-VL-7B-Instruct": "Qwen/Qwen2-VL-7B-Instruct",
    "Qwen2-Audio-7B": "Qwen/Qwen2-Audio-7B",
    "Qwen2.5-VL-7B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
    "llava-v1.5-13b": "liuhaotian/llava-v1.5-13b",

    # RL Models
    "chatglm-6b": "zai-org/chatglm-6b", 
    "chatglm-fitness-RLHF": "fb700/chatglm-fitness-RLHF",
    "Qwen3-4B-Base": "Qwen/Qwen3-4B-Base",
    "Qwen3_Medical_GRPO": "lastmass/Qwen3_Medical_GRPO",
    "Nous-Hermes-2-Mistral-7B-DPO": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
    "dolphin-2.6-mistral-7b-dpo": "dphn/dolphin-2.6-mistral-7b-dpo",
    "Nous-Hermes-2-Mixtral-8x7B-DPO": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    "MiniCPM-2B-dpo-bf16": "openbmb/MiniCPM-2B-dpo-bf16",
    "LLaMA3-iterative-DPO-final": "RLHFlow/LLaMA3-iterative-DPO-final",
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B",
    "Open-Reasoner-Zero-1.5B": "Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B",
    "Nemotron-Research-Reasoning-Qwen-1.5B": "nvidia/Nemotron-Research-Reasoning-Qwen-1.5B",
    "Open-Reasoner-Zero-7B": "Open-Reasoner-Zero/Open-Reasoner-Zero-7B",
    "hh_rlhf_rm_open_llama_3b": "weqweasdas/hh_rlhf_rm_open_llama_3b",
    "open_llama_3b": "openlm-research/open_llama_3b",

    # Pruning Models
    "Llama-3.2-1B": "meta-llama/Llama-3.2-1B", 
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B", 
    "Llama-3.1-Minitron-4B-Width-Base": "nvidia/Llama-3.1-Minitron-4B-Width-Base",
    "Llama-3.1-Minitron-4B-Depth-Base": "nvidia/Llama-3.1-Minitron-4B-Depth-Base",
    "Sheared-LLaMA-1.3B": "princeton-nlp/Sheared-LLaMA-1.3B",
    "Sheared-LLaMA-2.7B": "princeton-nlp/Sheared-LLaMA-2.7B",
    "Sheared-LLaMA-1.3B-Pruned": "princeton-nlp/Sheared-LLaMA-1.3B-Pruned",
    "Sheared-LLaMA-2.7B-Pruned": "princeton-nlp/Sheared-LLaMA-2.7B-Pruned",
    "Sheared-LLaMA-1.3B-ShareGPT": "princeton-nlp/Sheared-LLaMA-1.3B-ShareGPT",
    "Sheared-LLaMA-2.7B-ShareGPT": "princeton-nlp/Sheared-LLaMA-2.7B-ShareGPT",

    #pangu vs qwen
    "Pangu": "IntervitensInc/pangu-pro-moe-model",
    "Qwen2.5-14B": "Qwen/Qwen2.5-14B",
}

# All available configurations, mapped to their experiment names.
AVAILABLE_CONFIGS = {
    # Main experiments
    "sft_pairs": SFT_PAIRS_CONFIG,
    "cpt_pairs": CPT_PAIRS_CONFIG,
    "rl_pairs": RL_PAIRS_CONFIG,
    "multimodal_pairs": MULTIMODAL_PAIRS_CONFIG,
    "pruning_pairs": PRUNING_PAIRS_CONFIG,
    "all_moe_pairs": ALL_MOE_PAIRS_CONFIG, # Upcycling
    
    # Additional / alternative configs
    "sft_13b_7b_pairs": SFT_13B_7B_PAIRS_CONFIG,
    
    # Negative set experiments
    "independent_pairs": INDEPENDENT_PAIRS_CONFIG,
    "independent_pairs_13b": INDEPENDENT_PAIRS_CONFIG_13B,
}

def get_config(config_name: str) -> dict:
    """
    Retrieves the experiment configuration by name and dynamically generates pairs if needed.
    
    For 'independent_pairs' and 'independent_pairs_13b', this function generates
    all possible unique pairs of models from the provided `model_paths`.

    Args:
        config_name (str): The name of the configuration (e.g., "sft_pairs").
        
    Returns:
        dict: A dictionary containing 'model_paths' and 'analysis_pairs'.
        
    Raises:
        ValueError: If the specified config_name is not found in AVAILABLE_CONFIGS.
    """
    if config_name not in AVAILABLE_CONFIGS:
        raise ValueError(f"Unknown config name: {config_name}. Available configs: {list(AVAILABLE_CONFIGS.keys())}")
    
    config = AVAILABLE_CONFIGS[config_name].copy()
    
    # For independent_pairs, dynamically generate all possible pairs
    if config_name in ["independent_pairs", "independent_pairs_13b"]:
        model_names = list(config["model_paths"].keys())
        analysis_pairs = []
        for model1, model2 in combinations(model_names, 2):
            analysis_pairs.append({
                "label": f"{model1} vs {model2}",
                "weights1_name": model1,
                "weights2_name": model2
            })
        config["analysis_pairs"] = analysis_pairs
    
    return config
