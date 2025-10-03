#!/bin/bash
#
# Script to download language model checkpoints from the Hugging Face Hub.
#
# This script facilitates the download of models required for the AWM (Accurate Weight-Matrix Fingerprint)
# analysis. It can download models for a specific experiment configuration or all models at once.
#
# Usage:
#   # Show available experiment configurations
#   ./download_models.sh
#
#   # Download all models for a specific experiment (e.g., sft_pairs)
#   ./download_models.sh sft_pairs
#
#   # Download all models listed across all experiments
#   ./download_models.sh all

# --- Configuration ---
# The default directory to save model checkpoints.
# This should match CHECKPOINT_BASE_DIR in configs.py
CHECKPOINT_BASE_DIR="../checkpoints"


# --- Helper Functions ---

# Function to get the list of available config keys from configs.py
get_config_keys() {
    python3 -c "from configs import AVAILABLE_CONFIGS; print(' '.join(AVAILABLE_CONFIGS.keys()))"
}

# Function to get a unique list of Hugging Face IDs for a given config or 'all'
get_hf_ids() {
    local config_name=$1
    python3 -c "
import sys
from configs import AVAILABLE_CONFIGS, MODEL_HF_MAP

config_name = '$config_name'

hf_ids = set()

if config_name == 'all':
    configs_to_process = AVAILABLE_CONFIGS.values()
elif config_name in AVAILABLE_CONFIGS:
    configs_to_process = [AVAILABLE_CONFIGS[config_name]]
else:
    print(f'Error: Config \"{config_name}\" not found.', file=sys.stderr)
    sys.exit(1)

for config in configs_to_process:
    for folder_name in config.get('model_paths', {}).values():
        if folder_name in MODEL_HF_MAP:
            hf_ids.add(MODEL_HF_MAP[folder_name])
        else:
            print(f'Warning: No Hugging Face ID mapping found for folder \"{folder_name}\" in configs.py', file=sys.stderr)

print(' '.join(sorted(list(hf_ids))))
"
}

# Function to download a single model using huggingface-cli
download_model() {
    local hf_id=$1
    local folder_name=$(echo "$hf_id" | awk -F/ '{print $NF}')
    local target_dir="$CHECKPOINT_BASE_DIR/$folder_name"

    if [ -d "$target_dir" ]; then
        echo "--> Model '$hf_id' already exists at '$target_dir'. Skipping."
    else
        echo "--> Downloading '$hf_id' to '$target_dir'..."
        huggingface-cli download "$hf_id" --local-dir "$target_dir" --local-dir-use-symlinks False --resume-download
        if [ $? -eq 0 ]; then
            echo "--> Successfully downloaded '$hf_id'."
        else
            echo "--> Error downloading '$hf_id'. Please check the error message above."
        fi
    fi
}


# --- Main Logic ---

# Ensure the script is run from the correct directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
if [ ! -f "$SCRIPT_DIR/configs.py" ]; then
    echo "Error: This script must be run from the 'analysis_code' directory."
    exit 1
fi
cd "$SCRIPT_DIR"

CONFIG_KEYS=$(get_config_keys)
TARGET_CONFIG=$1

# If no argument is provided, print usage info
if [ -z "$TARGET_CONFIG" ]; then
    echo "Usage: $0 [config_name | all]"
    echo ""
    echo "Available configurations:"
    for key in $CONFIG_KEYS; do
        echo "  - $key"
    done
    echo "  - all (to download all models from all configurations)"
    exit 0
fi

# Get the list of models to download
echo "Parsing configuration '$TARGET_CONFIG'..."
MODELS_TO_DOWNLOAD=$(get_hf_ids "$TARGET_CONFIG")
if [ -z "$MODELS_TO_DOWNLOAD" ]; then
    echo "No models found to download for '$TARGET_CONFIG'. This could be due to an incorrect config name or missing mappings in MODEL_HF_MAP."
    exit 1
fi

# Create the base checkpoints directory if it doesn't exist
mkdir -p "$CHECKPOINT_BASE_DIR"

# Convert space-separated string to array
read -r -a model_array <<< "$MODELS_TO_DOWNLOAD"

echo "Found ${#model_array[@]} unique models to download for configuration '$TARGET_CONFIG'."
echo "------------------------------------------------------------------"

# Loop through and download each model
i=1
for model_id in $MODELS_TO_DOWNLOAD; do
    echo "[${i}/${#model_array[@]}] Processing model: $model_id"
    download_model "$model_id"
    i=$((i+1))
done

echo "------------------------------------------------------------------"
echo "All download tasks completed."
