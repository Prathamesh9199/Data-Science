import subprocess

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
    print(f"    Total     : {m['MemTotal']:.1f} GB")
    print(f"    Used      : {used:.1f} GB")
    print(f"    Available : {m['MemAvailable']:.1f} GB")
    print(f"    Cached    : {m.get('Cached', 0):.1f} GB")

print("=" * 50)
print("  RAM Cleaner")
print("=" * 50)

print("\nBefore:")
print_ram("RAM status")

# Step 1: sync writes to disk
subprocess.run(["sync"], check=True)
print("\n  ✓ sync done")

# Step 2: drop page cache, dentries, inodes (needs root)
subprocess.run(
    ["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
    check=True
)
print("  ✓ page cache, dentries, inodes dropped")

# Step 3: compact memory (needs root)
subprocess.run(
    ["sudo", "sh", "-c", "echo 1 > /proc/sys/vm/compact_memory"],
    check=True
)
print("  ✓ memory compacted")

print("\nAfter:")
print_ram("RAM status")

print(f"\n{'='*50}")