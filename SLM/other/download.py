from huggingface_hub import snapshot_download

print("🚀 Starting PeS2o download (Grabbing all available Parquet chunks)...")

snapshot_download(
    repo_id="allenai/peS2o",
    repo_type="dataset",
    revision="refs/convert/parquet",
    allow_patterns=["**/*.parquet", "*.parquet"], 
    local_dir="data/pes2o_raw",
    local_dir_use_symlinks=False
)

print("\n✅ Download complete! The files are in data/pes2o_raw/")