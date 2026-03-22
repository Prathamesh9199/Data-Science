import subprocess
import os

def check_hardware():
    print("🔍 Scanning Pop!_OS Hardware for ML Training...\n")
    print("-" * 50)
    
    # 1. Check GPU / VRAM
    try:
        # Calls nvidia-smi natively on Linux
        nvidia_smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"], 
            encoding='utf-8'
        ).strip().split('\n')
        
        for i, gpu in enumerate(nvidia_smi):
            name, total_ram, free_ram = gpu.split(', ')
            total_mb = int(total_ram.replace(' MiB', ''))
            print(f"🖥️  GPU {i}: {name}")
            print(f"📊 Total VRAM: {total_mb} MB")
            
            # MATH FOR MODEL SIZE:
            # We reserve 20% of VRAM for batch size, context window, and system overhead.
            usable_vram_mb = total_mb * 0.8
            usable_vram_bytes = usable_vram_mb * 1024 * 1024
            
            # Rule of thumb for full training (fp16/bf16 + AdamW): ~18 bytes per parameter
            max_params_full = usable_vram_bytes / 18
            
            # Rule of thumb for LoRA fine-tuning: ~6 bytes per parameter
            max_params_lora = usable_vram_bytes / 6
            
            print(f"  -> Max Model Size (Full Pre-training): ~{max_params_full / 1e6:.0f} Million parameters")
            print(f"  -> Max Model Size (LoRA Fine-tuning): ~{max_params_lora / 1e6:.0f} Million parameters\n")
            
    except FileNotFoundError:
        print("❌ No NVIDIA GPU detected or nvidia-smi not installed.")
        print("Training locally will rely on CPU, which is highly impractical for building an SLM from scratch.\n")

    # 2. Check System RAM (Crucial for loading datasets)
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.readlines()
        total_ram_kb = int([line for line in meminfo if 'MemTotal' in line][0].split()[1])
        total_ram_gb = total_ram_kb / (1024 * 1024)
        print(f"🧠 System RAM: {total_ram_gb:.1f} GB")
        if total_ram_gb < 16:
            print("  ⚠️ Warning: Less than 16GB RAM might bottleneck dataset tokenization.")
        else:
            print("  ✅ System RAM looks sufficient for data loading pipelines.\n")
    except Exception:
        print("Could not read system RAM.\n")
        
    print("-" * 50)

if __name__ == "__main__":
    check_hardware()