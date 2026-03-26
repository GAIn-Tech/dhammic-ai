"""
Genetic Autoresearch Orchestrator — Dhammic Cognitive Architecture

Launches parallel experiments on HF Jobs, collects results,
selects winners, plans crossover.

Usage:
    uv run evolve.py launch <gen>     # Launch N parallel HF Jobs
    uv run evolve.py status <gen>     # Check job status
    uv run evolve.py collect <gen>    # Collect results
    uv run evolve.py select <gen>     # Select winners, plan next gen
"""

import json
import sys
import time
import os
from pathlib import Path

from huggingface_hub import HfApi, get_token, run_uv_job

# ─── Config ──────────────────────────────────────────────────────────────────

CODE_REPO = "icarus112/dhammic-ai-code"
RESULTS_REPO = "icarus112/dhammic-ai-results"
FLAVOR = "a100-large"
TIMEOUT = "15m"
GENERATIONS_DIR = Path("generations")
RESULTS_FILE = Path("results.tsv")

# ─── Generation 1: 10 orthogonal mutations for A100 80GB ────────────────────

GENERATION_1 = [
    {"name": "baseline",         "overrides": {}},
    {"name": "wide_d256_l4",     "overrides": {"d_model": 256, "n_layers": 4}},
    {"name": "deep_d128_l8",     "overrides": {"d_model": 128, "n_layers": 8}},
    {"name": "big_sdr",          "overrides": {"sdr_dim": 2048, "sdr_k_active": 40}},
    {"name": "dense_engram",     "overrides": {"engram_n_columns": 2048, "engram_cells_per_col": 8, "engram_k_active": 20}},
    {"name": "high_lr",          "overrides": {"lr": 5e-2}},
    {"name": "wide_expand3",     "overrides": {"d_model": 192, "mamba_expand": 3}},
    {"name": "many_heads",       "overrides": {"n_heads": 8, "d_model": 256}},
    {"name": "big_lora",         "overrides": {"lora_rank": 32}},
    {"name": "max_throughput",   "overrides": {"d_model": 96, "n_layers": 2, "lr": 5e-2}},
]

# ─── Job Script Template ─────────────────────────────────────────────────────

JOB_TEMPLATE = '''# /// script
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
"""HF Job: {name} (gen {gen_id})"""
import os, json, sys, time, re
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

MUTATION = "{name}"
GEN_ID = {gen_id}
OVERRIDES = {overrides_json}
CODE_REPO = "{code_repo}"
RESULTS_REPO = "{results_repo}"

# Download code from HF Hub
print(f"[job] Downloading code from {{CODE_REPO}} ...")
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
print(f"[job] Mutation: {{MUTATION}}")
print(f"[job] Overrides: {{OVERRIDES}}")

sys.path.insert(0, ".")
sys.path.insert(0, "./src")

# Patch requests to include HF auth header (FineWeb-Edu requires auth)
import requests as _req
_orig_get = _req.get
def _authed_get(url, **kw):
    token = os.environ.get("HF_TOKEN")
    if token and "huggingface.co" in url:
        kw.setdefault("headers", {{}})["Authorization"] = f"Bearer {{token}}"
    return _orig_get(url, **kw)
_req.get = _authed_get

print("[job] Preparing data (streaming) ...")
from prepare import download_data, train_tokenizer
download_data(num_shards=1)  # Minimal download, data recycles via infinite dataloader
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
        result_path = f"/tmp/result_gen{{GEN_ID}}_{{MUTATION}}.json"
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        api.upload_file(
            path_or_fileobj=result_path,
            path_in_repo=f"gen{{GEN_ID}}/{{MUTATION}}.json",
            repo_id=RESULTS_REPO, repo_type="dataset")
        print(f"[job] Results pushed to {{RESULTS_REPO}}")
except Exception as e:
    print(f"[job] Push failed: {{e}}")

print(f"\\nval_bpb: {{results.get('val_bpb', 'N/A')}}")
print(f"peak_vram_mb: {{results.get('peak_vram_mb', 'N/A')}}")
print(f"tok_per_sec: {{results.get('tok_per_sec', 'N/A')}}")
'''


# ─── Functions ────────────────────────────────────────────────────────────────

def upload_code():
    """Upload prepare.py, train.py, and src/ directory to HF Hub for jobs to download."""
    api = HfApi(token=get_token())
    api.create_repo(CODE_REPO, repo_type="dataset", exist_ok=True, private=True)

    for fname in ["prepare.py", "train.py"]:
        api.upload_file(
            path_or_fileobj=fname,
            path_in_repo=fname,
            repo_id=CODE_REPO,
            repo_type="dataset",
        )
        print(f"[evolve] Uploaded {fname} to {CODE_REPO}")

    # Upload entire src/ directory
    src_dir = Path("src")
    if src_dir.exists():
        api.upload_folder(
            folder_path=str(src_dir),
            path_in_repo="src",
            repo_id=CODE_REPO,
            repo_type="dataset",
        )
        print(f"[evolve] Uploaded src/ directory to {CODE_REPO}")
    else:
        print(f"[evolve] WARNING: src/ directory not found!")


GENERATION_6 = [
    # Champion d128/4L variants (high throughput ~220k tok/s)
    {"name": "g6_d128_l4_lr5e3", "overrides": {"d_model": 128, "n_layers": 4, "mamba_expand": 3, "lr": 5e-3, "n_heads": 8}},
    {"name": "g6_d128_l4_lr2e2", "overrides": {"d_model": 128, "n_layers": 4, "mamba_expand": 3, "lr": 2e-2, "n_heads": 8}},
    {"name": "g6_d128_l6_lr1e2", "overrides": {"d_model": 128, "n_layers": 6, "mamba_expand": 3, "lr": 1e-2, "n_heads": 8}},
    # d192/4L variants (best bpb ~1.303)
    {"name": "g6_d192_l4_lr5e3", "overrides": {"d_model": 192, "n_layers": 4, "mamba_expand": 3, "lr": 5e-3, "n_heads": 8}},
    {"name": "g6_d192_l4_lr2e2", "overrides": {"d_model": 192, "n_layers": 4, "mamba_expand": 3, "lr": 2e-2, "n_heads": 8}},
    {"name": "g6_d192_l6_lr1e2", "overrides": {"d_model": 192, "n_layers": 6, "mamba_expand": 3, "lr": 1e-2, "n_heads": 8}},
    # Width exploration
    {"name": "g6_d160_l4_lr1e2", "overrides": {"d_model": 160, "n_layers": 4, "mamba_expand": 3, "lr": 1e-2, "n_heads": 8}},
    {"name": "g6_d256_l4_lr1e2", "overrides": {"d_model": 256, "n_layers": 4, "mamba_expand": 3, "lr": 1e-2, "n_heads": 8}},
    # Engram/SDR variations on champion
    {"name": "g6_d128_l4_eng4k", "overrides": {"d_model": 128, "n_layers": 4, "mamba_expand": 3, "lr": 1e-2, "engram_n_columns": 4096, "engram_cells_per_col": 8}},
    # Head count
    {"name": "g6_d192_l4_h16", "overrides": {"d_model": 192, "n_layers": 4, "mamba_expand": 3, "lr": 1e-2, "n_heads": 16}},
]

GENERATION_7 = [
    # d160 champion variants (best balance: 1.603 bpb at 195k tok/s)
    {"name": "g7_d160_l4_lr8e3", "overrides": {"d_model": 160, "n_layers": 4, "mamba_expand": 3, "lr": 8e-3, "n_heads": 8}},
    {"name": "g7_d160_l4_lr1.5e2", "overrides": {"d_model": 160, "n_layers": 4, "mamba_expand": 3, "lr": 1.5e-2, "n_heads": 8}},
    {"name": "g7_d160_l4_eng4k", "overrides": {"d_model": 160, "n_layers": 4, "mamba_expand": 3, "lr": 1e-2, "engram_n_columns": 4096, "engram_cells_per_col": 8}},
    {"name": "g7_d160_l4_sdr4k", "overrides": {"d_model": 160, "n_layers": 4, "mamba_expand": 3, "lr": 1e-2, "sdr_dim": 4096, "sdr_k_active": 80}},
    {"name": "g7_d160_l4_ds32", "overrides": {"d_model": 160, "n_layers": 4, "mamba_expand": 3, "lr": 1e-2, "d_state": 32}},
    # Cross d160 x d192 (merge best bpb with best throughput)
    {"name": "g7_d176_l4_lr1e2", "overrides": {"d_model": 176, "n_layers": 4, "mamba_expand": 3, "lr": 1e-2, "n_heads": 8}},
    {"name": "g7_d176_l4_lr1.5e2", "overrides": {"d_model": 176, "n_layers": 4, "mamba_expand": 3, "lr": 1.5e-2, "n_heads": 8}},
    # Wider exploration
    {"name": "g7_d192_l4_lr1.5e2", "overrides": {"d_model": 192, "n_layers": 4, "mamba_expand": 3, "lr": 1.5e-2, "n_heads": 8}},
    # Depth with throughput-friendly width
    {"name": "g7_d128_l8_lr1e2", "overrides": {"d_model": 128, "n_layers": 8, "mamba_expand": 3, "lr": 1e-2, "n_heads": 8}},
    # Speed king: can we get better bpb at max throughput?
    {"name": "g7_d128_l4_lr1.5e2", "overrides": {"d_model": 128, "n_layers": 4, "mamba_expand": 3, "lr": 1.5e-2, "n_heads": 8}},
]

GENERATIONS = {1: GENERATION_1, 6: GENERATION_6, 7: GENERATION_7}


def get_generation(gen_id: int) -> list:
    if gen_id in GENERATIONS:
        return GENERATIONS[gen_id]

    gen_file = GENERATIONS_DIR / f"gen_{gen_id}.md"
    if not gen_file.exists():
        print(f"[evolve] Gen {gen_id} not defined. Run 'select {gen_id - 1}' first.")
        sys.exit(1)

    mutations = []
    for line in gen_file.read_text().split("\n"):
        if line.startswith("|") and "---" not in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 5 and parts[2] and parts[2] not in ("Name", ""):
                name = parts[2]
                cfg_str = parts[4] if len(parts) > 4 else "{}"
                overrides = {}
                if cfg_str.strip() and cfg_str.strip() != "{}":
                    for pair in cfg_str.strip().strip("`").split(","):
                        if "=" in pair:
                            k, v = pair.split("=", 1)
                            try:
                                overrides[k.strip()] = eval(v.strip())
                            except Exception:
                                pass
                mutations.append({"name": name, "overrides": overrides})
    return mutations


def launch_generation(gen_id: int):
    mutations = get_generation(gen_id)
    print(f"[evolve] Uploading code to HF Hub ...")
    upload_code()

    print(f"[evolve] Launching Gen {gen_id}: {len(mutations)} mutations on {FLAVOR}")
    token = get_token()
    api = HfApi(token=token)
    job_ids = {}

    for m in mutations:
        name = m["name"]
        script = JOB_TEMPLATE.format(
            name=name, gen_id=gen_id,
            overrides_json=json.dumps(m["overrides"]),
            code_repo=CODE_REPO, results_repo=RESULTS_REPO,
        )

        script_path = Path(f"jobs/gen{gen_id}_{name}.py")
        script_path.parent.mkdir(exist_ok=True)
        script_path.write_text(script)

        try:
            job = run_uv_job(
                str(script_path),
                flavor=FLAVOR,
                timeout=TIMEOUT,
                secrets={"HF_TOKEN": token},
            )
            jid = job.id if hasattr(job, "id") else str(job)
            job_ids[name] = jid
            print(f"  [{name}] -> {jid}")
        except Exception as e:
            job_ids[name] = f"ERROR: {e}"
            print(f"  [{name}] FAILED: {e}")

    tracker = {"gen_id": gen_id, "jobs": job_ids, "launched": time.time()}
    tracker_path = GENERATIONS_DIR / f"gen_{gen_id}_jobs.json"
    GENERATIONS_DIR.mkdir(exist_ok=True)
    tracker_path.write_text(json.dumps(tracker, indent=2))

    print(f"\n[evolve] {len(job_ids)} jobs launched. Track at: https://huggingface.co/settings/jobs")
    print(f"[evolve] Job IDs saved to {tracker_path}")


def check_status(gen_id: int):
    tracker_path = GENERATIONS_DIR / f"gen_{gen_id}_jobs.json"
    if not tracker_path.exists():
        print(f"[evolve] No jobs for gen {gen_id}.")
        return

    tracker = json.loads(tracker_path.read_text())
    api = HfApi(token=get_token())

    print(f"\n{'Name':<20} {'Status':<15} {'Job ID':<30}")
    print("-" * 65)
    for name, jid in tracker["jobs"].items():
        if jid.startswith("ERROR"):
            status = "LAUNCH_FAILED"
        else:
            try:
                j = api.inspect_job(job_id=jid)
                status = j.status.stage
            except Exception:
                status = "POLL_ERR"
        print(f"{name:<20} {status:<15} {jid:<30}")


def collect_results(gen_id: int):
    api = HfApi(token=get_token())
    print(f"[evolve] Collecting gen {gen_id} results from {RESULTS_REPO} ...")

    try:
        files = list(api.list_repo_tree(RESULTS_REPO, repo_type="dataset",
                                         path_in_repo=f"gen{gen_id}"))
        result_files = [f for f in files if f.rfilename.endswith(".json")]
    except Exception as e:
        print(f"[evolve] No results yet: {e}")
        return

    results = []
    for rf in result_files:
        path = api.hf_hub_download(RESULTS_REPO, rf.rfilename, repo_type="dataset")
        with open(path) as f:
            r = json.load(f)
        results.append(r)
        print(f"  [{r['mutation']}] tok/s={r.get('tok_per_sec','?')} | "
              f"val_bpb={r.get('val_bpb','?')} | "
              f"vram={r.get('peak_vram_mb','?')}MB")

    # Sort by: 1) val_bpb (lower=better), 2) throughput (higher=better)
    results.sort(key=lambda r: (r.get("val_bpb", float("inf")), -r.get("tok_per_sec", 0)))

    # Append to results.tsv
    with open(RESULTS_FILE, "a") as f:
        for r in results:
            f.write(f"gen{gen_id}_{r['mutation']}\t{r.get('val_bpb','')}\t"
                    f"{r.get('peak_vram_mb','')}\t{r.get('tok_per_sec','')}\t"
                    f"gen{gen_id}\t{r['mutation']}\n")

    if results:
        print(f"\n[evolve] Best: {results[0]['mutation']} (val_bpb={results[0].get('val_bpb')})")

    # Update gen markdown
    gen_file = GENERATIONS_DIR / f"gen_{gen_id}_results.md"
    md = f"# Generation {gen_id} Results\n\n"
    md += "| # | Name | val_bpb | vram_mb | tok/s | params | steps |\n"
    md += "|---|------|---------|---------|-------|--------|-------|\n"
    for i, r in enumerate(results, 1):
        md += (f"| {i} | {r['mutation']} | {r.get('val_bpb','?')} | "
               f"{r.get('peak_vram_mb','?')} | {r.get('tok_per_sec','?')} | "
               f"{r.get('config',{}).get('d_model','?')}d | {r.get('total_steps','?')} |\n")
    gen_file.write_text(md)
    print(f"[evolve] Written to {gen_file}")


def select_and_crossover(gen_id: int):
    api = HfApi(token=get_token())
    try:
        files = list(api.list_repo_tree(RESULTS_REPO, repo_type="dataset",
                                         path_in_repo=f"gen{gen_id}"))
        result_files = [f for f in files if f.rfilename.endswith(".json")]
    except Exception:
        print(f"[evolve] No results for gen {gen_id}.")
        return

    results = []
    for rf in result_files:
        path = api.hf_hub_download(RESULTS_REPO, rf.rfilename, repo_type="dataset")
        with open(path) as f:
            results.append(json.load(f))

    results.sort(key=lambda r: r.get("val_bpb", float("inf")))
    baseline = next((r for r in results if r["mutation"] == "baseline"), results[-1])
    baseline_bpb = baseline.get("val_bpb", float("inf"))

    winners = [r for r in results if r.get("val_bpb", float("inf")) < baseline_bpb]
    if not winners:
        winners = results[:3]
        print(f"[evolve] Nothing beat baseline ({baseline_bpb}). Taking top 3.")

    print(f"\n[evolve] Winners ({len(winners)}):")
    for w in winners:
        delta = ((baseline_bpb - w["val_bpb"]) / baseline_bpb * 100) if baseline_bpb else 0
        print(f"  {w['mutation']}: val_bpb={w['val_bpb']} ({delta:+.2f}%)")

    # Crossover: pairwise + elites
    # Extract ONLY params that differ from DhammicConfig defaults (the actual mutations)
    from dataclasses import fields as dc_fields, MISSING
    try:
        from train import DhammicConfig
        default_cfg = {}
        for f in dc_fields(DhammicConfig):
            if f.default is not MISSING:
                default_cfg[f.name] = f.default
    except Exception:
        default_cfg = {}

    CROSSOVER_KEYS = (
        "d_model", "n_layers", "d_state", "n_heads", "mamba_expand",
        "sdr_dim", "sdr_k_active",
        "engram_n_columns", "engram_cells_per_col", "engram_k_active", "engram_layer",
        "lora_rank", "lr",
    )

    def extract_overrides(result):
        """Extract only params that differ from defaults — the actual mutation."""
        full_cfg = result.get("config", {})
        overrides = {}
        for k in CROSSOVER_KEYS:
            if k in full_cfg and full_cfg[k] != default_cfg.get(k):
                overrides[k] = full_cfg[k]
        return overrides

    next_id = gen_id + 1
    crossovers = []

    # Elites (top 3 with their actual overrides only)
    for w in winners[:3]:
        cfg = extract_overrides(w)
        crossovers.append({"name": f"{w['mutation']}_elite", "overrides": cfg})

    # Pairwise crossover — take architecture from w1, training/memory from w2
    ARCH_KEYS = ("d_model", "n_layers", "mamba_expand", "n_heads", "d_state")
    MEMORY_KEYS = ("sdr_dim", "sdr_k_active", "engram_n_columns", "engram_cells_per_col",
                   "engram_k_active", "engram_layer", "lora_rank")
    TRAIN_KEYS = ("lr",)

    for i, w1 in enumerate(winners):
        for w2 in winners[i + 1:]:
            o1 = extract_overrides(w1)
            o2 = extract_overrides(w2)
            # Architecture from w1, memory/training from w2
            merged = {}
            for k in ARCH_KEYS:
                if k in o1: merged[k] = o1[k]
                elif k in o2: merged[k] = o2[k]
            for k in (*MEMORY_KEYS, *TRAIN_KEYS):
                if k in o2: merged[k] = o2[k]
                elif k in o1: merged[k] = o1[k]
            crossovers.append({"name": f"{w1['mutation']}+{w2['mutation']}", "overrides": merged})

    crossovers = crossovers[:10]

    gen_file = GENERATIONS_DIR / f"gen_{next_id}.md"
    md = f"# Generation {next_id}: Crossover of Gen {gen_id} Winners\n\n"
    md += f"Winners: {', '.join(w['mutation'] for w in winners)}\n\n"
    md += "## Mutations\n\n"
    md += "| # | Name | Config Override |\n"
    md += "|---|------|----------------|\n"
    for i, m in enumerate(crossovers, 1):
        o = ", ".join(f"{k}={v}" for k, v in m["overrides"].items())
        md += f"| {i} | {m['name']} | `{o}` |\n"

    gen_file.write_text(md)
    print(f"\n[evolve] Gen {next_id}: {len(crossovers)} mutations planned -> {gen_file}")
    print(f"[evolve] Run: uv run evolve.py launch {next_id}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    cmd, gen = sys.argv[1], int(sys.argv[2])
    GENERATIONS_DIR.mkdir(exist_ok=True)

    {"launch": launch_generation, "status": check_status,
     "collect": collect_results, "select": select_and_crossover}[cmd](gen)
