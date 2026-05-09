#!/bin/bash

# 1. Cleanup Function (Runs on exit/crash)
cleanup() {
    echo "🔓 Re-enabling sleep and suspension..."
    sudo systemctl unmask sleep.target suspend.target hibernate.target hybrid-sleep.target
}
trap cleanup EXIT

# 2. Environment Tweaks for 8GB VRAM
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"

echo "🚀 Starting SLM Pipeline..."

# 3. Disable Sleep
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'

# 4. Your Python Tasks
python3 release_RAM.py
python3 collect_sft_data.py