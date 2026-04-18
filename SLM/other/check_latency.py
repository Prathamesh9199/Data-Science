import torch
import time
import os
import torch.nn as nn
import numpy as np

# Simulation Parameters based on your 600M Model
BATCH_SIZE = 2
SEQ_LEN = 4096
D_MODEL = 1024
LAYERS = 36
DTYPE = torch.bfloat16 

device = torch.device("cuda")
print(f"🚀 Hardware: {torch.cuda.get_device_name(0)}")
print(f"📊 Testing Data Size: ~{(BATCH_SIZE * SEQ_LEN * D_MODEL * 2) / 1e6:.2f} MB per layer")

# 1. TEST RECALCULATION (GPU MATH)
def test_gpu_math():
    layer = nn.Linear(D_MODEL, D_MODEL * 4).to(device).to(DTYPE)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, device=device, dtype=DTYPE)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    # Simulate re-running 36 layers
    for _ in range(LAYERS):
        _ = layer(x)
        
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 # ms

# 2. TEST SSD OFFLOADING (I/O)
def test_ssd_io():
    # We use float32 for the CPU/SSD part because numpy doesn't support bfloat16
    # This actually makes the SSD look slightly FASTER than it would be with bfloat16 
    # because it doesn't have to handle specialized bit-conversion overhead.
    data = torch.randn(LAYERS, BATCH_SIZE, SEQ_LEN, D_MODEL).numpy()
    file_path = "vram_dump.bin"
    
    start = time.perf_counter()
    
    # Write to SSD
    with open(file_path, "wb") as f:
        f.write(data.tobytes())
    
    # Read back from SSD
    with open(file_path, "rb") as f:
        _ = f.read()
        
    duration = (time.perf_counter() - start) * 1000 # ms
    if os.path.exists(file_path):
        os.remove(file_path)
    return duration

print("\n--- Starting Latency Battle ---")
gpu_time = test_gpu_math()
print(f"⚡ GPU Recalculation (36 layers): {gpu_time:.2f} ms")

ssd_time = test_ssd_io()
print(f"💾 SSD Write/Read (36 layers):  {ssd_time:.2f} ms")

print("\n--- Final Verdict ---")
if gpu_time < ssd_time:
    ratio = ssd_time / gpu_time
    print(f"🏆 GPU RECALCULATION is {ratio:.1f}x FASTER than your SSD.")
else:
    ratio = gpu_time / ssd_time
    print(f"🏆 SSD OFFLOADING is {ratio:.1f}x FASTER than recalculating.")