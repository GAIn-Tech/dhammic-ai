#!/usr/bin/env python3
"""
Sprint 3 Task 3.1 acceptance criterion:
micro-model convergence test with Hebbian LoRA (Python simulation).

Verifies: loss decreases over 1000 steps, final_loss < initial_loss * 0.8.
"""

import math
import random
import sys


def relu(x):
    return max(0.0, x)


def softmax(logits):
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]


def cross_entropy(probs, target):
    return -math.log(max(probs[target], 1e-10))


class MicroLM:
    def __init__(self, vocab=16, hidden=32, rank=4, seed=42):
        random.seed(seed)
        self.vocab = vocab
        self.hidden = hidden
        self.rank = rank

        def rand_mat(rows, cols, scale=0.1):
            return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]

        self.embed = rand_mat(vocab, hidden, scale=0.02)
        self.proj = rand_mat(vocab, hidden, scale=0.02)

        self.lora_a = rand_mat(rank, hidden, scale=0.01)
        self.lora_b = [[0.0] * rank for _ in range(hidden)]

    def mat_vec(self, M, x):
        return [sum(M[i][j] * x[j] for j in range(len(x))) for i in range(len(M))]

    def forward(self, token_id):
        h = self.embed[token_id][:]
        h_lora_in = h[:]

        h_mid = self.mat_vec(self.lora_a, h_lora_in)
        delta_h = self.mat_vec(self.lora_b, h_mid)
        h_out = [h[i] + delta_h[i] for i in range(self.hidden)]
        h_act = [relu(v) for v in h_out]

        logits = self.mat_vec(self.proj, h_act)
        probs = softmax(logits)
        return probs, h_act, h_mid

    def hebbian_update(self, h_in, h_mid, h_out, eta, lam):
        for r in range(self.rank):
            for d in range(self.hidden):
                w = self.lora_a[r][d]
                delta = eta * (h_mid[r] * h_in[d] - lam * w)
                self.lora_a[r][d] = w + delta

        for d in range(self.hidden):
            for r in range(self.rank):
                w = self.lora_b[d][r]
                delta = eta * (h_out[d] * h_mid[r] - lam * w)
                self.lora_b[d][r] = w + delta


def run_micro(n_steps=1000, report_every=100):
    """
    Hebbian LoRA convergence on bigram prediction task.

    Phase 1 (steps 0-499): Gradient-based backbone warmup (embed + proj, NO LoRA).
                            This simulates a pretrained backbone.
    Phase 2 (steps 500-999): Backbone frozen, only Hebbian LoRA adapts.
                              Measures whether Hebbian LoRA can further reduce loss.
    Acceptance: Phase 2 final loss < Phase 1 baseline * 0.80.
    """
    model = MicroLM(vocab=64, hidden=32, rank=4, seed=42)

    bigrams = {i: (i + 3) % 16 for i in range(16)}
    sequence = list(range(16)) * (n_steps // 16 + 2)

    window = 50
    loss_history = []
    phase1_end = n_steps // 2
    lr_backbone = 0.5

    for step in range(n_steps):
        tok_in = sequence[step % len(sequence)]
        tok_target = bigrams[tok_in]

        probs, h_act, h_mid = model.forward(tok_in)
        loss = cross_entropy(probs, tok_target)
        loss_history.append(loss)

        if step < phase1_end:
            delta_logit = probs[:]
            delta_logit[tok_target] -= 1.0
            for v in range(model.vocab):
                for d in range(model.hidden):
                    model.proj[v][d] -= lr_backbone * delta_logit[v] * h_act[d]
            for d in range(model.hidden):
                grad_h = sum(
                    delta_logit[v] * model.proj[v][d] for v in range(model.vocab)
                )
                model.embed[tok_in][d] -= lr_backbone * 0.05 * grad_h
        else:
            eta_t = 0.05 / (1.0 + (step - phase1_end) / 200.0)
            model.hebbian_update(model.embed[tok_in], h_mid, h_act, eta_t, 0.05)

        if (step + 1) % report_every == 0:
            recent = loss_history[-window:]
            avg = sum(recent) / len(recent)
            phase = "backbone" if step < phase1_end else "hebbian"
            print(
                f"  step {step + 1:4d} [{phase}]: loss={loss:.4f}, avg_{window}={avg:.4f}"
            )

    phase1_losses = loss_history[phase1_end - window : phase1_end]
    phase2_losses = loss_history[-window:]
    phase1_avg = sum(phase1_losses) / len(phase1_losses)
    phase2_avg = sum(phase2_losses) / len(phase2_losses)
    ratio = phase2_avg / phase1_avg if phase1_avg > 0 else 1.0

    print(f"\nConvergence summary:")
    print(f"  Phase 1 (backbone) final avg loss: {phase1_avg:.4f}")
    print(f"  Phase 2 (Hebbian LoRA) final avg loss: {phase2_avg:.4f}")
    print(f"  Ratio (phase2/phase1): {ratio:.4f}")

    converged = ratio <= 0.80
    print(
        f"  Threshold: phase2 < phase1 * 0.80 → {'PASS' if converged else 'FAIL'} (ratio={ratio:.4f})"
    )
    return converged, phase1_avg, phase2_avg


if __name__ == "__main__":
    print("Sprint 3 — Hebbian LoRA micro-model convergence test")
    print("=" * 60)
    converged, initial, final = run_micro(n_steps=1000)
    if converged:
        print(
            f"\nConverged: loss decreased from {initial:.4f} to {final:.4f} over 1000 steps"
        )
        sys.exit(0)
    else:
        print(
            f"\nFAIL: loss did not decrease sufficiently ({initial:.4f} → {final:.4f})"
        )
        sys.exit(1)
