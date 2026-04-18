#!/bin/bash

# 1. Environment Tweaks for 8GB VRAM
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"

echo "============================================================"
echo "🚀 SLM ULTIMATE TRAINING PIPELINE"
echo "============================================================"

# 2. Prevent Sleep & Suspension (Ironman Mode)
echo "🔒 Disabling sleep and suspension..."
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'

# 3. Clean RAM & Swap
echo "🧹 Cleaning System RAM and Swap..."
if sudo python3 release_RAM.py; then
    echo "✅ RAM cleared."
else
    echo "❌ RAM cleaning failed. Aborting."
    # Re-enable sleep before exiting if failed
    sudo systemctl unmask sleep.target suspend.target hibernate.target hybrid-sleep.target
    exit 1
fi

clear

# 4. Launch Training
echo "🔥 Phase 2: Starting 36-Layer Training..."
echo "Note: You are currently seeing live logs. Do not close this terminal."
echo "------------------------------------------------------------"

python3 pretrain.py

# 5. Restore System Settings after training completes or crashes
echo ""
echo "🔓 Training finished. Restoring power settings..."
sudo systemctl unmask sleep.target suspend.target hibernate.target hybrid-sleep.target
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'suspend'

echo "============================================================"
echo "🏁 Pipeline Complete."
echo "============================================================"