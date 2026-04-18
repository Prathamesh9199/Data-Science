import torch
import subprocess
import shutil

def run_recon():
    print("="*60)
    print("🔍 SYSTEM HARDWARE RECON")
    print("="*60)

    # 1. System RAM & Swap (Pop!_OS/Linux native)
    print("\n🖥️  CPU SYSTEM MEMORY (RAM & SWAP)")
    print("-" * 60)
    if shutil.which("free"):
        # Runs the standard Linux memory check
        subprocess.run(["free", "-h"])
    else:
        print("Could not find 'free' command.")

    # 2. GPU VRAM (The AI Limit)
    print("\n🎮 GPU VIDEO MEMORY (VRAM)")
    print("-" * 60)
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"GPUs detected: {device_count}")
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            total_vram_gb = props.total_memory / (1024**3)
            print(f"\nGPU [{i}]: {props.name}")
            print(f"  Physical VRAM: {total_vram_gb:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            
        print("\n📊 Current GPU Allocation (nvidia-smi):")
        if shutil.which("nvidia-smi"):
            subprocess.run([
                "nvidia-smi", 
                "--query-gpu=index,memory.total,memory.used,memory.free", 
                "--format=csv"
            ])
    else:
        print("❌ No CUDA GPU detected by PyTorch!")
        
    print("\n" + "="*60)
    print("Recon Complete.")

if __name__ == "__main__":
    run_recon()