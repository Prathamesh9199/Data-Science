import subprocess
import time

def read_meminfo():
    with open('/proc/meminfo', 'r') as f:
        info = {}
        for line in f:
            key, val = line.split(':')
            info[key.strip()] = int(val.strip().split()[0]) / 1024 / 1024  # GB
    return info

def print_ram(label):
    m = read_meminfo()
    used = m['MemTotal'] - m['MemAvailable']
    print(f"  {label}")
    print(f"    Total RAM : {m['MemTotal']:.1f} GB")
    print(f"    Used RAM  : {used:.1f} GB")
    print(f"    Available : {m['MemAvailable']:.1f} GB")
    print(f"    Cached    : {m.get('Cached', 0):.1f} GB")
    print(f"    Swap Total: {m.get('SwapTotal', 0):.1f} GB")
    print(f"    Swap Free : {m.get('SwapFree', 0):.1f} GB")

print("=" * 60)
print("  Deep RAM & Swap Cleaner")
print("=" * 60)

print("\nBefore:")
print_ram("Status")

print("\n[1/4] Syncing file systems...")
subprocess.run(["sync"], check=True)
print("  ✓ sync done")

print("[2/4] Dropping page cache, dentries, and inodes (Requires sudo)...")
subprocess.run(
    ["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
    check=True
)
print("  ✓ caches dropped")

print("[3/4] Compacting physical memory...")
subprocess.run(
    ["sudo", "sh", "-c", "echo 1 > /proc/sys/vm/compact_memory"],
    check=True
)
print("  ✓ memory compacted")

print("[4/4] Flushing Swapfile (This might take 10-30 seconds)...")
try:
    # Turning swap off forces the OS to dump stale swap data
    subprocess.run(["sudo", "swapoff", "-a"], check=True)
    time.sleep(2) # Give the kernel a moment to settle
    # Turn it back on fresh
    subprocess.run(["sudo", "swapon", "-a"], check=True)
    print("  ✓ Swap flushed and reset to 0% used")
except subprocess.CalledProcessError as e:
    print(f"  ⚠ Failed to flush swap. Error: {e}")
    print("  Ensure you have enough free physical RAM to absorb the swap contents.")

print("\nAfter:")
print_ram("Status")
print(f"\n{'='*60}")