#!/usr/bin/env python3
"""
Sprint 3 Task 3.3: Catastrophic forgetting measurement.

Protocol:
  1. Train backbone on Task A (bigrams from domain A: tokens 0-15)
  2. Adapt via Hebbian LoRA on Task B (bigrams from domain B: tokens 16-31)
  3. Measure Task A accuracy after N steps of Task B adaptation
  4. Compare: Hebbian LoRA (sparse) vs no adaptation (frozen) vs dense activation Hebbian

Acceptance: Task A retention >= 90% after 500 steps of Task B.
Output: benchmarks/forgetting_results.json
"""

import json
import math
import random
import sys
from pathlib import Path


def relu(x):
    return max(0.0, x)


def sp_sparse(v, sparsity=0.02):
    return v if abs(v) > 0.8 else 0.0


def softmax(logits):
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]


def cross_entropy(probs, target):
    return -math.log(max(probs[target], 1e-10))


def accuracy_top1(probs, target):
    return 1 if probs.index(max(probs)) == target else 0


class MicroLM:
    def __init__(self, vocab=32, hidden=32, rank=4, seed=42):
        random.seed(seed)
        self.vocab = vocab
        self.hidden = hidden
        self.rank = rank

        def rand_mat(rows, cols, scale=0.02):
            return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]

        self.embed = rand_mat(vocab, hidden)
        self.proj = rand_mat(vocab, hidden)
        self.lora_a = rand_mat(rank, hidden, scale=0.01)
        self.lora_b = [[0.0] * rank for _ in range(hidden)]

    def forward(self, tok, use_sparse=False):
        h = self.embed[tok][:]
        h_mid = [
            sum(self.lora_a[r][d] * h[d] for d in range(self.hidden))
            for r in range(self.rank)
        ]
        delta_h = [
            sum(self.lora_b[d][r] * h_mid[r] for r in range(self.rank))
            for d in range(self.hidden)
        ]
        h_out = [h[d] + delta_h[d] for d in range(self.hidden)]
        if use_sparse:
            h_act = [sp_sparse(relu(v)) for v in h_out]
        else:
            h_act = [relu(v) for v in h_out]
        logits = [
            sum(self.proj[v][d] * h_act[d] for d in range(self.hidden))
            for v in range(self.vocab)
        ]
        probs = softmax(logits)
        return probs, h_act, h_mid

    def gradient_update(self, tok, tok_target, lr=0.3):
        probs, h_act, _ = self.forward(tok)
        delta = probs[:]
        delta[tok_target] -= 1.0
        for v in range(self.vocab):
            for d in range(self.hidden):
                self.proj[v][d] -= lr * delta[v] * h_act[d]
        for d in range(self.hidden):
            grad_h = sum(delta[v] * self.proj[v][d] for v in range(self.vocab))
            self.embed[tok][d] -= lr * 0.05 * grad_h

    def hebbian_update(self, tok, h_mid, h_act, eta=0.02, lam=0.05, use_sparse=False):
        h_in = self.embed[tok]
        if use_sparse:
            h_act = [sp_sparse(v) for v in h_act]
        for r in range(self.rank):
            for d in range(self.hidden):
                w = self.lora_a[r][d]
                self.lora_a[r][d] = w + eta * (h_mid[r] * h_in[d] - lam * w)
        for d in range(self.hidden):
            for r in range(self.rank):
                w = self.lora_b[d][r]
                self.lora_b[d][r] = w + eta * (h_act[d] * h_mid[r] - lam * w)


def train_backbone(model, bigrams, n_steps=300, lr=0.3):
    keys = list(bigrams.keys())
    for step in range(n_steps):
        tok = keys[step % len(keys)]
        model.gradient_update(tok, bigrams[tok], lr)


def evaluate(model, bigrams, use_sparse=False):
    total, correct = 0, 0
    for tok, target in bigrams.items():
        probs, _, _ = model.forward(tok, use_sparse=use_sparse)
        correct += accuracy_top1(probs, target)
        total += 1
    return correct / total if total > 0 else 0.0


def adapt_hebbian(model, bigrams, n_steps=500, eta=0.02, lam=0.05, use_sparse=False):
    keys = list(bigrams.keys())
    for step in range(n_steps):
        tok = keys[step % len(keys)]
        target = bigrams[tok]
        eta_t = eta / (1.0 + step / 200.0)
        probs, h_act, h_mid = model.forward(tok, use_sparse=use_sparse)
        model.hebbian_update(tok, h_mid, h_act, eta_t, lam, use_sparse)


def run_forgetting_experiment(use_sparse=False, n_adapt_steps=500, seed=42):
    random.seed(seed)

    task_a = {i: (i + 3) % 16 for i in range(16)}
    task_b = {i + 16: (i + 5) % 16 + 16 for i in range(16)}

    model = MicroLM(vocab=32, hidden=32, rank=4, seed=seed)

    train_backbone(model, task_a, n_steps=400, lr=0.4)
    baseline_a = evaluate(model, task_a, use_sparse)

    adapt_hebbian(model, task_b, n_steps=n_adapt_steps, use_sparse=use_sparse)
    post_a = evaluate(model, task_a, use_sparse)

    retention = post_a / baseline_a if baseline_a > 0 else 0.0
    return {
        "use_sparse": use_sparse,
        "baseline_task_a_acc": round(baseline_a, 4),
        "post_taskb_task_a_acc": round(post_a, 4),
        "task_a_retention": round(retention, 4),
        "pass": retention >= 0.90,
    }


def run_no_adaptation_baseline(seed=42):
    random.seed(seed)

    task_a = {i: (i + 3) % 16 for i in range(16)}
    model = MicroLM(vocab=32, hidden=32, rank=4, seed=seed)
    train_backbone(model, task_a, n_steps=400, lr=0.4)
    acc = evaluate(model, task_a)
    return {
        "use_sparse": False,
        "baseline_task_a_acc": round(acc, 4),
        "post_taskb_task_a_acc": round(acc, 4),
        "task_a_retention": 1.0,
        "pass": True,
        "note": "no adaptation — retention always 100%",
    }


if __name__ == "__main__":
    print("Sprint 3 — Catastrophic Forgetting Measurement Suite")
    print("=" * 60)

    dense_result = run_forgetting_experiment(use_sparse=False, n_adapt_steps=500)
    sparse_result = run_forgetting_experiment(use_sparse=True, n_adapt_steps=500)
    frozen_result = run_no_adaptation_baseline()

    for name, r in [
        ("Hebbian LoRA (dense)", dense_result),
        ("Hebbian LoRA (sparse)", sparse_result),
        ("Frozen (no adaptation)", frozen_result),
    ]:
        status = "PASS" if r["pass"] else "FAIL"
        print(f"  {name}:")
        print(f"    Baseline Task A: {r['baseline_task_a_acc']:.2%}")
        print(f"    After Task B:    {r['post_taskb_task_a_acc']:.2%}")
        print(f"    Retention:       {r['task_a_retention']:.2%}  [{status}]")
        print()

    retention_dense = dense_result["task_a_retention"]
    retention_sparse = sparse_result["task_a_retention"]

    results = {
        "hebbian_dense": dense_result,
        "hebbian_sparse": sparse_result,
        "frozen_baseline": frozen_result,
        "task_a_retention": retention_sparse,
        "sparse_vs_dense_retention": round(retention_sparse - retention_dense, 4),
        "sparse_better": retention_sparse >= retention_dense,
        "pass": dense_result["pass"] and sparse_result["pass"],
    }

    out = Path(__file__).parent / "forgetting_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Results written to {out}")

    overall = results["pass"]
    print(f"\nOverall: {'PASS' if overall else 'FAIL'}")
    print(f"  Sparse activations better retention: {results['sparse_better']}")
    print(
        f"  Task A retention (Hebbian LoRA sparse): {retention_sparse:.2%} (threshold: ≥90%)"
    )
    sys.exit(0 if overall else 1)
