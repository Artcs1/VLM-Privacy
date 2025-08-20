from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="bop-benchmark/hot3d",
    repo_type="dataset",
    local_dir="./hot3d_dataset",
    # Optionally restrict file types (e.g. only zip files) if necessary:
    allow_patterns="object_models/*.glb"
)

snapshot_download(
    repo_id="bop-benchmark/hot3d",
    repo_type="dataset",
    local_dir="./hot3d_dataset",
    # Optionally restrict file types (e.g. only zip files) if necessary:
    allow_patterns="object_models/*.json"
)
