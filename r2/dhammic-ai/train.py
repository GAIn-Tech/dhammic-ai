"""
Dhammic Cognitive Architecture v3 — Training Script
Inline dharmic architecture targeting 100k+ tok/s.
SDR Embedding -> [Mamba-3 SSD + SwiGLU] x N -> HTMEngram(layer 2) -> LM Head

Uses autoresearch data infrastructure (FineWeb via climbmix-400b).
Dual-mode: local training (default config) OR HF Jobs entry point (overrides from evolve.py).
"""

import os
import sys
import time
import math
import json
import torch
import torch.nn.functional as F
from dataclasses import dataclass, asdict, fields

sys.path.insert(0, os.path.expanduser("~/autoresearch"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from prepare import (
    Tokenizer, make_dataloader, get_token_bytes,
    MAX_SEQ_LEN, TIME_BUDGET, EVAL_TOKENS, VOCAB_SIZE,
)
from citta_vithi_pipeline import CittaVithiPipeline

# Performance: TF32 + cuDNN auto-tune (from autoresearch)
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True


# -- Config dataclass -- all tunable parameters in one place ----------------

@dataclass
class DhammicConfig:
    # Model architecture
    d_model: int = 256              # A100 baseline, up from 128
    n_layers: int = 6               # Deeper backbone for A100 80GB
    d_state: int = 16
    mamba_expand: int = 3
    n_heads: int = 8
    chunk_size: int = 256           # A100 can handle larger chunks

    # SDR Embedding
    sdr_dim: int = 2048
    sdr_k_active: int = 40

    # HTM Engram
    engram_n_columns: int = 2048
    engram_cells_per_col: int = 8
    engram_k_active: int = 20
    engram_layer: int = 3

    # Hebbian LoRA
    lora_rank: int = 16

    # Training
    batch_size: int = 32            # A100 has 80GB
    seq_len: int = 1024
    lr: float = 2e-2                # High LR proven optimal in mamba-edge-sdr evolution
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    max_steps: int = -1             # -1 means use TIME_BUDGET
    use_grad_checkpoint: bool = False  # A100 has plenty of VRAM
    eval_interval: int = 200
    eval_steps: int = 5
    cosine_period: int = 10000      # SGDR warm restart cycle

    # Derived (not user-set)
    vocab_size: int = 0             # Set at runtime from tokenizer

    def __post_init__(self):
        # Ensure seq_len is divisible by chunk_size
        if self.seq_len % self.chunk_size != 0:
            self.chunk_size = min(self.chunk_size, self.seq_len)


# -- Factual eval prompts ---------------------------------------------------

FACTUAL_PROMPTS = [
    ("The capital of France is", "Paris"),
    ("Water boils at", "100"),
    ("The largest planet in our solar system is", "Jupiter"),
    ("The chemical symbol for gold is", "Au"),
    ("Albert Einstein developed the theory of", "relativity"),
    ("The speed of light is approximately", "300"),
    ("DNA stands for", "deoxyribonucleic"),
    ("The Great Wall of China is located in", "China"),
    ("Photosynthesis converts sunlight into", "energy"),
    ("The human body has 206", "bones"),
    ("Shakespeare wrote", "plays"),
    ("The Amazon is the world's largest", "river"),
    ("Oxygen has atomic number", "8"),
    ("The moon orbits the", "Earth"),
    ("Gravity was described by", "Newton"),
    ("The Mona Lisa was painted by", "Leonardo"),
    ("The mitochondria is the powerhouse of the", "cell"),
    ("Pi is approximately equal to", "3.14"),
    ("The first president of the United States was", "George"),
    ("The periodic table organizes", "elements"),
]


# -- Model creation ---------------------------------------------------------

def create_model(config: DhammicConfig, device: str):
    model = CittaVithiPipeline(
        d_model=config.d_model, n_layers=config.n_layers, d_state=config.d_state,
        mamba_expand=config.mamba_expand, n_heads=config.n_heads, chunk_size=config.chunk_size,
        vocab_size=config.vocab_size, sdr_dim=config.sdr_dim, sdr_k_active=config.sdr_k_active,
        engram_n_columns=config.engram_n_columns, engram_cells_per_col=config.engram_cells_per_col,
        engram_k_active=config.engram_k_active, engram_layer=config.engram_layer,
        lora_rank=config.lora_rank, use_grad_checkpoint=config.use_grad_checkpoint,
    )
    return model.to(device)


# -- LR schedule: cosine annealing with warm restarts (SGDR) ---------------

def get_lr(step: int, warmup: int, max_lr: float, cosine_period: int):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    cycle_step = (step - warmup) % cosine_period
    decay_ratio = cycle_step / max(1, cosine_period)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * decay_ratio))


# -- Evaluation -------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_loader, token_bytes, device, steps: int = 5):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_bytes = 0

    for _ in range(steps):
        x, y, _ = next(val_loader)
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()
        total_bytes += token_bytes[y].sum().item()

    avg_loss = total_loss / total_tokens
    val_bpb = total_loss / (total_bytes * math.log(2)) if total_bytes > 0 else float("inf")
    model.train()
    return avg_loss, val_bpb


# -- Factual eval (English knowledge + perplexity) --------------------------

@torch.no_grad()
def run_factual_eval(model, config: DhammicConfig, max_gen_tokens: int = 10):
    """Evaluate factual prompts with greedy generation + perplexity."""
    import tiktoken

    enc = tiktoken.get_encoding("o200k_base")
    model.eval()
    device = next(model.parameters()).device

    results = []
    correct = 0
    gen_prompts = FACTUAL_PROMPTS[:5]
    total = len(gen_prompts)

    print(f"\n{'='*60}\nFACTUAL ENGLISH EVAL ({total} generation + {len(FACTUAL_PROMPTS)} perplexity)\n{'='*60}")

    for prompt_text, expected_keyword in gen_prompts:
        tokens = enc.encode(prompt_text, disallowed_special=())
        tokens = [t if t < config.vocab_size else 0 for t in tokens]

        # Greedy autoregressive generation
        generated = list(tokens)
        for _ in range(max_gen_tokens):
            if len(generated) >= config.seq_len:
                break
            x_in = torch.tensor([generated[-config.seq_len:]], dtype=torch.long, device=device)
            logits = model(x_in)
            if isinstance(logits, tuple):
                logits = logits[0]
            next_logits = logits[0, -1]
            next_token = next_logits.argmax().item()
            generated.append(next_token)
            decoded_last = enc.decode([next_token])
            if "\n" in decoded_last:
                break

        completion = enc.decode(generated[len(tokens):])
        hit = expected_keyword.lower() in completion.lower()
        if hit:
            correct += 1

        status = "PASS" if hit else "FAIL"
        results.append({
            "prompt": prompt_text,
            "expected": expected_keyword,
            "completion": completion.strip()[:80],
            "correct": hit,
        })
        print(f"  [{status}] {prompt_text}")
        print(f"         -> {completion.strip()[:80]}")

    accuracy = correct / total if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"FACTUAL EVAL: {correct}/{total} = {accuracy:.1%}")
    print(f"{'='*60}\n")

    # Prompt perplexity (how confident is the model on factual text?)
    total_nll = 0
    total_prompt_tokens = 0
    for prompt_text, _ in FACTUAL_PROMPTS[:5]:
        tokens = enc.encode(prompt_text, disallowed_special=())
        tokens = [t if t < config.vocab_size else 0 for t in tokens]
        if len(tokens) < 2:
            continue
        x = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        shift_logits = logits[0, :-1]
        shift_labels = torch.tensor(tokens[1:], device=device)
        nll = F.cross_entropy(shift_logits, shift_labels, reduction="sum")
        total_nll += nll.item()
        total_prompt_tokens += len(tokens) - 1

    prompt_ppl = math.exp(total_nll / max(total_prompt_tokens, 1))
    print(f"Prompt perplexity (5 samples): {prompt_ppl:.2f}")

    return {
        "factual_accuracy": accuracy,
        "factual_correct": correct,
        "factual_total": total,
        "prompt_perplexity": prompt_ppl,
        "samples": results[:5],
    }


# -- Main entry point -------------------------------------------------------

def main(overrides: dict | None = None):
    """Run training. Accepts optional config overrides dict from evolve.py.

    Args:
        overrides: Dict of DhammicConfig field names -> values.
                   None means use defaults (local mode).

    Returns:
        Dict with val_bpb, tok_per_sec, peak_vram_mb, total_steps, config, eval.
    """
    # Build config from defaults + overrides
    config = DhammicConfig()
    if overrides:
        for k, v in overrides.items():
            if k in {f.name for f in fields(DhammicConfig)}:
                setattr(config, k, v)
        config.__post_init__()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, VRAM: {props.total_memory / 1024**3:.1f} GB, SMs: {props.multi_processor_count}")

    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size > 32768:
        vocab_size = 32768
    config.vocab_size = vocab_size
    print(f"Vocab size: {vocab_size}")

    token_bytes = get_token_bytes(device=device)
    batch_size = config.batch_size

    train_loader = iter(make_dataloader(tokenizer, batch_size, config.seq_len, "train"))
    val_loader = iter(make_dataloader(tokenizer, batch_size, config.seq_len, "val"))

    model = create_model(config, device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count / 1e6:.2f}M")
    print(f"Architecture: d={config.d_model} n_layers={config.n_layers} expand={config.mamba_expand} "
          f"heads={config.n_heads} sdr={config.sdr_dim} engram={config.engram_n_columns}")

    # VRAM check + auto batch-size reduction on OOM
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        dummy_x = torch.randint(0, vocab_size, (batch_size, config.seq_len), device=device)
        dummy_y = torch.randint(0, vocab_size, (batch_size, config.seq_len), device=device)
        try:
            logits = model(dummy_x)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), dummy_y.reshape(-1))
            loss.backward()
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak VRAM (fwd+bwd): {peak:.2f} GB")
            model.zero_grad(set_to_none=True)
            del dummy_x, dummy_y, logits, loss
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print("OOM on probe! Reducing batch size...")
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            batch_size = max(4, batch_size // 2)
            config.batch_size = batch_size
            print(f"Reduced batch_size to {batch_size}")
            train_loader = iter(make_dataloader(tokenizer, batch_size, config.seq_len, "train"))
            val_loader = iter(make_dataloader(tokenizer, batch_size, config.seq_len, "val"))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, betas=(0.9, 0.95),
        weight_decay=config.weight_decay, fused=(device == "cuda"),
    )

    # Checkpoint resume (local mode only)
    ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoint.pt")
    start_step = 0
    total_tokens_prev = 0
    best_val_bpb = float("inf")
    if overrides is None and os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step = ckpt.get("step", 0)
            total_tokens_prev = ckpt.get("total_tokens", 0)
            best_val_bpb = ckpt.get("best_val_bpb", float("inf"))
            print(f"Resumed from step {start_step}, best_val_bpb {best_val_bpb:.6f}")
        except Exception as e:
            print(f"Fresh start (checkpoint incompatible: {e})")

    # Decide stopping criterion: time budget (evolve) or max_steps (pretrain)
    use_time_budget = config.max_steps <= 0
    time_budget = TIME_BUDGET if use_time_budget else float("inf")
    max_steps = config.max_steps if config.max_steps > 0 else float("inf")

    print(f"\nTraining: B={batch_size} T={config.seq_len} d={config.d_model} layers={config.n_layers}")
    if use_time_budget:
        print(f"Time budget: {TIME_BUDGET}s")
    else:
        print(f"Max steps: {config.max_steps}")

    # --- GPU Optimizations ---
    if device == "cuda":
        torch.set_float32_matmul_precision('high')  # TF32 on Ampere+
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # AMP scaler for mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))
    autocast_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device == "cuda"))

    # torch.compile for fused kernels (skip on old drivers)
    try:
        if device == "cuda":
            model = torch.compile(model)
            print("torch.compile: enabled")
    except Exception as e:
        print(f"torch.compile: skipped ({e})")

    model.train()
    step = start_step
    tokens_processed = 0
    t0 = time.time()
    log_interval = 10

    while True:
        elapsed = time.time() - t0
        if elapsed >= time_budget:
            break
        if step >= start_step + max_steps:
            break

        lr = get_lr(step, config.warmup_steps, config.lr, config.cosine_period)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y, _ = next(train_loader)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        step += 1
        tokens_processed += batch_size * config.seq_len

        # Periodic CUDA defragmentation
        if step % 500 == 0 and device == "cuda":
            torch.cuda.empty_cache()

        if step % log_interval == 0:
            elapsed = time.time() - t0
            tps = tokens_processed / elapsed
            if use_time_budget:
                print(f"step {step:>5d} | loss {loss.item():.4f} | lr {lr:.2e} | "
                      f"tok/s {tps:.0f} | elapsed {elapsed:.0f}s/{TIME_BUDGET}s")
            else:
                steps_done = step - start_step
                print(f"step {step:>5d} | loss {loss.item():.4f} | lr {lr:.2e} | "
                      f"tok/s {tps:.0f} | {steps_done}/{config.max_steps} steps")

        if step % config.eval_interval == 0:
            val_loss, val_bpb = evaluate(model, val_loader, token_bytes, device, config.eval_steps)
            is_best = val_bpb < best_val_bpb
            if is_best:
                best_val_bpb = val_bpb
            marker = " *BEST*" if is_best else ""
            print(f"  >> val_loss {val_loss:.4f} | val_bpb: {val_bpb:.6f}{marker}")

    # Final eval
    val_loss, val_bpb = evaluate(model, val_loader, token_bytes, device, config.eval_steps)
    if val_bpb < best_val_bpb:
        best_val_bpb = val_bpb

    elapsed = time.time() - t0
    tps = tokens_processed / elapsed if elapsed > 0 else 0
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0

    total_tokens_now = total_tokens_prev + tokens_processed

    # Save checkpoint (local mode only)
    if overrides is None:
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "total_tokens": total_tokens_now,
            "best_val_bpb": best_val_bpb,
        }, ckpt_path)

    # Factual eval stage
    eval_results = run_factual_eval(model, config)

    # Summary
    print(f"\n{'='*60}")
    print(f"Steps: {step} ({start_step}->{step}), {elapsed:.1f}s")
    print(f"Tokens: {tokens_processed:,} this run | {total_tokens_now:,} total")
    print(f"Throughput: {tps:.0f} tok/s")
    print(f"val_bpb: {val_bpb:.6f}")
    print(f"best_val_bpb: {best_val_bpb:.6f}")
    if device == "cuda":
        print(f"Peak VRAM: {peak_vram_mb:.0f} MB ({peak_vram_mb / 1024:.2f} GB)")
    print(f"Factual accuracy: {eval_results['factual_accuracy']:.1%}")
    print(f"Prompt perplexity: {eval_results['prompt_perplexity']:.2f}")
    print(f"{'='*60}")

    # Machine-readable output for evolve.py
    print(f"\nval_bpb: {val_bpb:.6f}")
    print(f"peak_vram_mb: {peak_vram_mb:.1f}")
    print(f"tok_per_sec: {tps:.0f}")
    print(f"total_steps: {step}")

    return {
        "val_bpb": val_bpb,
        "tok_per_sec": tps,
        "peak_vram_mb": peak_vram_mb,
        "total_steps": step,
        "config": asdict(config),
        "eval": {
            "factual_accuracy": eval_results["factual_accuracy"],
            "prompt_perplexity": eval_results["prompt_perplexity"],
        },
    }


if __name__ == "__main__":
    overrides = json.loads(sys.argv[1]) if len(sys.argv) > 1 else None
    result = main(overrides)
    print(json.dumps(result, indent=2, default=str))
