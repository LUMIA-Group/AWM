"""
Main script for QK CKA Similarity Calculation.
This script computes the CKA similarity for attention Q and K weights between different models.
It uses word embedding alignment to handle different feature dimensions.
Usage:
    python main.py --config moe_pairs
"""

import argparse
import torch
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from similarity_metrics import (
    load_all_weights_from_dir,
    get_attention_weights,
    calculate_attention_cka_similarities,
    generate_negative_sample,
    get_word_embedding_weight,
    load_vocab_from_dir,
    find_overlapping_vocab
)
from configs import get_config, AVAILABLE_CONFIGS, CHECKPOINT_BASE_DIR

def run_experiment(config_name: str = None, model1_path: str = None, model2_path: str = None, device: str = 'cpu'):
    """
    Runs the similarity analysis experiment.

    This function can operate in two modes:
    1. Config Mode: Uses a predefined experiment configuration from `configs.py`.
    2. Quick Comparison Mode: Compares two models directly from their local paths.

    Args:
        config_name (str, optional): The name of the predefined experiment configuration.
        model1_path (str, optional): The local path to the first model's checkpoint directory.
        model2_path (str, optional): The local path to the second model's checkpoint directory.
        device (str, optional): The computing device to use ('cpu' or 'cuda'). Defaults to 'cpu'.
    """
    
    is_quick_comparison = False
    if model1_path and model2_path:
        # Quick Comparison Mode
        is_quick_comparison = True
        model1_name = os.path.basename(os.path.normpath(model1_path))
        model2_name = os.path.basename(os.path.normpath(model2_path))
        
        experiment_name = f"{model1_name}_vs_{model2_name}"
        
        model_paths = {
            model1_name: model1_path,
            model2_name: model2_path,
        }
        analysis_pairs = [{
            "label": f"{model1_name} vs {model2_name}",
            "weights1_name": model1_name,
            "weights2_name": model2_name,
        }]
    elif config_name:
        # Config Mode
        config = get_config(config_name)
        model_paths = config["model_paths"]
        analysis_pairs = config["analysis_pairs"]
        experiment_name = config_name
    else:
        raise ValueError("Invalid execution: Please provide either a --config name or two --model_paths.")
        
    print(f"\n{'='*60}")
    print(f"Starting Experiment: {experiment_name}")
    print(f"Using device: {device}")
    print(f"{'='*60}")
    
    print(f"\n--- Pre-loading model weights, vocabularies, and embeddings ---")
    all_attention_weights = {}
    all_vocabs = {}
    all_embeddings = {}

    for model_name, model_folder in model_paths.items():
        print(f"Loading model: {model_name}")
        if is_quick_comparison:
            model_path = model_folder
        else:
            model_path = os.path.join(CHECKPOINT_BASE_DIR, model_folder)
            
        if not os.path.isdir(model_path):
            print(f"Error: Directory not found for model '{model_name}': {model_path}")
            print("Please ensure the model has been downloaded and the path is correct.")
            all_attention_weights[model_name] = None
            all_vocabs[model_name] = None
            all_embeddings[model_name] = None
            continue

        state_dict = load_all_weights_from_dir(model_path)
        if state_dict:
            all_attention_weights[model_name] = get_attention_weights(state_dict)
            all_vocabs[model_name] = load_vocab_from_dir(model_path)
            all_embeddings[model_name] = get_word_embedding_weight(state_dict)
        else:
            print(f"Warning: Failed to load model {model_name} from path: {model_path}")
            all_attention_weights[model_name] = None
            all_vocabs[model_name] = None
            all_embeddings[model_name] = None
    
    # Check if any model failed to load, which might affect negative sample generation
    if "random" in [p["weights2_name"] for p in analysis_pairs] or "random" in [p["weights1_name"] for p in analysis_pairs]:
        print(f"\n--- Generating random negative sample ---")
        ref_model_name = next((name for name, weights in all_attention_weights.items() if weights), None)
        
        if ref_model_name:
            print(f"Using {ref_model_name} as reference to generate random sample")
            all_attention_weights["random"] = generate_negative_sample(all_attention_weights[ref_model_name])
            all_vocabs["random"] = all_vocabs.get(ref_model_name)
            all_embeddings["random"] = all_embeddings.get(ref_model_name)
        else:
            print("Warning: Cannot generate random sample, no suitable reference model found.")
            all_attention_weights["random"], all_vocabs["random"], all_embeddings["random"] = None, None, None

    print(f"\n--- Starting QK CKA Similarity Analysis ---")
    qk_results = []
    
    for pair in analysis_pairs:
        name1, name2, label = pair["weights1_name"], pair["weights2_name"], pair["label"]
        
        if not all([
            name1 in all_attention_weights, name2 in all_attention_weights,
            all_attention_weights[name1], all_attention_weights[name2]
        ]):
            print(f"\nSkipping analysis for '{label}' due to missing model weights for '{name1}' or '{name2}'.")
            continue

        weights1, weights2 = all_attention_weights.get(name1), all_attention_weights.get(name2)
        vocab1, embedding1 = all_vocabs.get(name1), all_embeddings.get(name1)
        vocab2, embedding2 = all_vocabs.get(name2), all_embeddings.get(name2)

        if not all([weights1, weights2]):
            print(f"Skipping analysis for '{label}' - missing model weights.")
            continue
        
        # --- Initialize alignment parameters ---
        subselect_indices = None
        subselect_signs = None
        base_model_is_first = None

        # --- Unified Alignment Logic ---
        if all([vocab1, embedding1 is not None, vocab2, embedding2 is not None]):
            print(f"\nApplying unified dimension alignment for '{label}'...")
            _, indices1, indices2 = find_overlapping_vocab(vocab1, vocab2)

            if len(indices1) > 0:
                emb1 = torch.index_select(embedding1, 0, torch.tensor(indices1)).to(torch.float32).numpy()
                emb2 = torch.index_select(embedding2, 0, torch.tensor(indices2)).to(torch.float32).numpy()

                # 1. Automatically determine base and target models
                if emb1.shape[1] >= emb2.shape[1]:
                    emb_base, emb_target = emb1, emb2
                    base_model_is_first = True
                else:
                    emb_base, emb_target = emb2, emb1
                    base_model_is_first = False
                
                print(f"  Base model embedding: {emb_base.shape}, Target model embedding: {emb_target.shape}")
                
                # 2. Compute cost matrix and run LAP
                use_cuda = torch.cuda.is_available() and device == 'cuda'
                compute_device = torch.device("cuda" if use_cuda else "cpu")
                print(f"  Using device for alignment computation: {compute_device}")

                if use_cuda:
                    emb_base_t = torch.from_numpy(emb_base).to(compute_device)
                    emb_target_t = torch.from_numpy(emb_target).to(compute_device)
                    emb_base_norm = F.normalize(emb_base_t.T, p=2, dim=1)
                    emb_target_norm = F.normalize(emb_target_t.T, p=2, dim=1)
                    similarity_matrix_t = torch.mm(emb_base_norm, emb_target_norm.T)
                    cost_matrix_t = 1 - torch.abs(similarity_matrix_t)
                    cost_matrix = cost_matrix_t.cpu().numpy()
                    similarity_matrix = similarity_matrix_t.cpu().numpy()
                else:
                    similarity_matrix = cosine_similarity(emb_base.T, emb_target.T)
                    cost_matrix = 1 - np.abs(similarity_matrix)

                base_indices, target_indices = linear_sum_assignment(cost_matrix)

                # 3. Create dimension index map and sign vector
                perm = np.argsort(target_indices)
                subselect_indices = base_indices[perm]
                
                d_target = emb_target.shape[1]
                subselect_signs = np.sign(similarity_matrix[subselect_indices, np.arange(d_target)])
                
                print(f"  Found {len(subselect_indices)} matching dimensions, with {(subselect_signs == -1).sum()} needing sign flips.")
            else:
                print("  Warning: No overlapping vocabulary, cannot perform alignment.")
        else:
            print(f"Warning: Missing embeddings or vocabularies, skipping alignment for '{label}'.")

        print(f"\nAnalyzing pair: {label} ({name1} vs {name2})")
        result = calculate_attention_cka_similarities(
            weights1, weights2,
            device=device,
            subselect_indices=subselect_indices,
            subselect_signs=subselect_signs,
            base_model_is_first=base_model_is_first
        )
        
        if result and 'averages' in result and result['averages']:
            result['label'] = label
            qk_results.append(result)

    if qk_results:
        print(f"\n{'*'*70}")
        print(f"*** {experiment_name} Experiment Results Summary ***")
        print(f"{'*'*70}")
        sort_key = 'avg_direct' if any('avg_direct' in res['averages'] for res in qk_results) else 'q_direct'
        qk_results.sort(key=lambda x: x['averages'].get(sort_key, 0), reverse=True)
        for res in qk_results:
            print(f"--- {res['label']} ---")
            
            order = ['Wq_weights', 'Wk_weights', 'Wq_Wk_weights']
            sorted_averages = sorted(
                res['averages'].items(),
                key=lambda item: order.index(item[0]) if item[0] in order else float('inf')
            )
            for metric, avg_score in sorted_averages:
                print(f"  Average {metric:<15} Similarity(%) = {avg_score*100:.2f}")
            
            wq_wk_avg = res['averages'].get('Wq_Wk_weights')
            if wq_wk_avg is not None:
                neg_pair_mean = 0.00375
                # NOTE: The standard deviation for negative pairs was not specified.
                neg_pair_std = 0.0028
                z_score = abs((wq_wk_avg - neg_pair_mean) / neg_pair_std)
                print(f"  Reference Similarity(%) for 90 negative pairs (mean) = {neg_pair_mean*100:.2f}")
                print(f"  Absolute Z-Score vs negative pairs = {z_score:.2f}")


    print(f"\n{'='*60}")
    print("Experiment finished!")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(
        description="Run QK CKA Similarity Analysis Experiments.\n"
                    "You can either run a predefined experiment with --config or a quick comparison with --model_paths.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--config", 
        choices=list(AVAILABLE_CONFIGS.keys()),
        help=f"Select a predefined experiment configuration from:\n{', '.join(AVAILABLE_CONFIGS.keys())}"
    )
    group.add_argument(
        "--model_paths",
        nargs=2,
        metavar=("MODEL1_PATH", "MODEL2_PATH"),
        default=["/data0/byzeng/checkpoint/Qwen-7B","/data0/byzeng/checkpoint/OLMo-7B"],
        help="Run a quick comparison between two local model directories."
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Select the computation device ('auto' detects CUDA availability)."
    )
    
    args = parser.parse_args()
    
    device_to_use = 'cuda' if args.device == "auto" and torch.cuda.is_available() else 'cpu'
    
    if args.config:
        run_experiment(config_name=args.config, device=device_to_use)
    elif args.model_paths:
        run_experiment(model1_path=args.model_paths[0], model2_path=args.model_paths[1], device=device_to_use)

if __name__ == "__main__":
    main() 