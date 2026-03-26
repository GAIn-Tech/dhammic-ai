# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch==2.6.0",
#     "pytorch-lightning>=2.6.0",
#     "tiktoken>=0.11.0",
#     "rustbpe>=0.1.0",
#     "requests>=2.32",
#     "pyarrow>=18.0",
#     "huggingface-hub>=0.35.0",
#     "datasets>=3.0",
#     "einops>=0.8.0",
#     "tensorboard>=2.18",
# ]
# ///
"""HF Job: g2_elite_d192_exp3 (gen 2)"""
import os, json, sys, time, re
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

MUTATION = "g2_elite_d192_exp3"
GEN_ID = 2
OVERRIDES = {"d_model": 192, "n_layers": 6, "mamba_expand": 3}
CODE_REPO = "icarus112/dhammic-ai-code"
RESULTS_REPO = "icarus112/dhammic-ai-results"

# Download code from HF Hub
print(f"[job] Downloading code from {CODE_REPO} ...")
for fname in ["prepare.py", "train.py"]:
    hf_hub_download(CODE_REPO, fname, repo_type="dataset",
                    local_dir=".", token=os.environ.get("HF_TOKEN"))

# Download src/ directory
print(f"[job] Downloading src/ directory ...")
snapshot_download(
    CODE_REPO, repo_type="dataset",
    local_dir=".", allow_patterns="src/*",
    token=os.environ.get("HF_TOKEN"),
)

# Prepare data (download shards + tokenizer)
print(f"[job] Mutation: {MUTATION}")
print(f"[job] Overrides: {OVERRIDES}")

sys.path.insert(0, ".")
sys.path.insert(0, "./src")

# Patch requests to include HF auth header (FineWeb-Edu requires auth)
import requests as _req
_orig_get = _req.get
def _authed_get(url, **kw):
    token = os.environ.get("HF_TOKEN")
    if token and "huggingface.co" in url:
        kw.setdefault("headers", {})["Authorization"] = f"Bearer {token}"
    return _orig_get(url, **kw)
_req.get = _authed_get

print("[job] Downloading data ...")
from prepare import download_data, train_tokenizer
download_data(num_shards=3)
train_tokenizer()

# Call main(overrides) — train.py has DhammicConfig dataclass that accepts overrides
print("[job] Starting training (time budget) ...")
from train import main
results = main(OVERRIDES)
results["mutation"] = MUTATION
results["gen_id"] = GEN_ID

# results dict already populated by main() with val_bpb, tok_per_sec, peak_vram_mb, config, eval

# Push results
try:
    token = os.environ.get("HF_TOKEN")
    if token:
        api = HfApi(token=token)
        api.create_repo(RESULTS_REPO, repo_type="dataset", exist_ok=True, private=True)
        result_path = f"/tmp/result_gen{GEN_ID}_{MUTATION}.json"
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        api.upload_file(
            path_or_fileobj=result_path,
            path_in_repo=f"gen{GEN_ID}/{MUTATION}.json",
            repo_id=RESULTS_REPO, repo_type="dataset")
        print(f"[job] Results pushed to {RESULTS_REPO}")
except Exception as e:
    print(f"[job] Push failed: {e}")

print(f"\nval_bpb: {results.get('val_bpb', 'N/A')}")
print(f"peak_vram_mb: {results.get('peak_vram_mb', 'N/A')}")
print(f"tok_per_sec: {results.get('tok_per_sec', 'N/A')}")
