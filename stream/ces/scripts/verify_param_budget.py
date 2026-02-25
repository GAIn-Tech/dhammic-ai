#!/usr/bin/env python3
"""
CES Parameter Budget Verifier

Parses the CES architecture spec for the JSON param_budget block
and verifies the total is 130M ± 1M.

Usage:
    python scripts/verify_param_budget.py docs/architecture-spec.md
"""

import sys
import json
import re
import argparse


TARGET = 130_000_000
TOLERANCE = 1_000_000


def parse_budget_from_spec(path: str) -> dict:
    """Extract the param_budget JSON block from the architecture spec."""
    with open(path, "r") as f:
        content = f.read()

    # Find JSON block containing "param_budget"
    pattern = r'```json\s*(\{[^`]*"param_budget"[^`]*\})\s*```'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError(
            f"No JSON block with 'param_budget' found in {path}.\n"
            "Expected a fenced ```json block containing 'param_budget' key."
        )

    raw_json = match.group(1)
    data = json.loads(raw_json)
    return data["param_budget"]


def verify_budget(budget: dict) -> tuple[int, bool]:
    """Verify the param_budget dict. Returns (total, passed)."""
    total = budget.get("total", None)
    if total is None:
        raise ValueError("param_budget must contain a 'total' key.")

    # Also verify sum of components matches stated total
    component_sum = sum(v for k, v in budget.items() if k != "total")
    stated_total = total

    if abs(component_sum - stated_total) > 1000:
        print(
            f"WARNING: Component sum ({component_sum:,}) differs from stated total "
            f"({stated_total:,}) by {component_sum - stated_total:,}"
        )

    passed = abs(stated_total - TARGET) <= TOLERANCE
    return stated_total, passed


def main():
    parser = argparse.ArgumentParser(description="Verify CES parameter budget.")
    parser.add_argument(
        "spec_path",
        nargs="?",
        default="docs/architecture-spec.md",
        help="Path to architecture-spec.md (default: docs/architecture-spec.md)",
    )
    args = parser.parse_args()

    try:
        budget = parse_budget_from_spec(args.spec_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    total, passed = verify_budget(budget)

    print("\nCES Parameter Budget Verification")
    print("=" * 50)
    for component, params in budget.items():
        if component == "total":
            continue
        pct = 100.0 * params / total if total > 0 else 0
        print(f"  {component:<30s}: {params:>14,}  ({pct:5.1f}%)")

    print("-" * 50)
    print(f"  {'TOTAL':<30s}: {total:>14,}  ({total / 1e6:.3f}M)")
    print(f"  {'TARGET':<30s}: {TARGET:>14,}  ({TARGET / 1e6:.0f}M)")
    print(f"  {'TOLERANCE':<30s}: ±{TOLERANCE:>13,}  (±{TOLERANCE / 1e6:.0f}M)")
    print(f"  {'DELTA':<30s}: {total - TARGET:>+14,}  ({(total - TARGET) / 1e6:+.3f}M)")
    print("=" * 50)

    if passed:
        print(f"\nTOTAL: {total / 1e6:.3f}M, PASS ✓")
        sys.exit(0)
    else:
        print(
            f"\nTOTAL: {total / 1e6:.3f}M, FAIL ✗  "
            f"(off by {(total - TARGET) / 1e6:+.3f}M)"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
