#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parent.parent


def gate_0(model_name: str) -> dict:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ces_train.test_loop",
            "--steps",
            "3",
            "--model",
            model_name,
        ],
        capture_output=True,
        text=True,
        cwd=ROOT / "ces-train",
    )
    passed = result.returncode == 0 and "complete" in result.stdout.lower()
    return {
        "sprint": 0,
        "model": model_name,
        "pass": passed,
        "metrics": {"test_loop_exit_code": result.returncode},
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip() if not passed else "",
    }


def gate_stub(sprint: int, model_name: str) -> dict:
    return {
        "sprint": sprint,
        "model": model_name,
        "pass": False,
        "metrics": {},
        "note": f"Gate G{sprint} not yet implemented (Sprint {sprint} not started)",
    }


GATES = {0: gate_0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sprint", type=int, required=True)
    parser.add_argument("--model", default="dummy")
    args = parser.parse_args()

    gate_fn = GATES.get(args.sprint, lambda m: gate_stub(args.sprint, m))
    result = gate_fn(args.model)

    print(json.dumps(result, indent=2))
    sys.exit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
