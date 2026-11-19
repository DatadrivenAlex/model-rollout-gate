"""Read offline prediction exports and produce a versioned strict JSON report."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from rollout_gate.evaluation import Policy, Prediction, evaluate


def read_predictions(path: Path) -> tuple[list[Prediction], str]:
    rows = []
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        line = 0
        while raw := stream.readline(8193):
            line += 1
            if len(raw) > 8192:
                raise ValueError(f"prediction line {line}: exceeds 8192 bytes")
            digest.update(raw)
            if not raw.strip():
                continue
            if len(rows) >= 10000:
                raise ValueError("at most 10000 predictions are supported")
            try:
                rows.append(Prediction.model_validate_json(raw))
            except ValueError as error:
                raise ValueError(f"prediction line {line}: invalid record") from error
    return rows, digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument(
        "--out", type=Path, help="new output file; existing files are not replaced"
    )
    args = parser.parse_args(argv)
    try:
        policy = Policy()
        if args.policy:
            with args.policy.open("rb") as stream:
                raw_policy = stream.read(65537)
            if len(raw_policy) > 65536:
                raise ValueError("policy exceeds 65536 bytes")
            policy = Policy.model_validate_json(raw_policy)
        rows, digest = read_predictions(args.predictions)
        result = {**evaluate(rows, policy), "input_sha256": digest}
        report = json.dumps(result, indent=2, allow_nan=False, sort_keys=True) + "\n"
        if args.out:
            with args.out.open("x", encoding="utf-8") as stream:
                stream.write(report)
        else:
            print(report, end="")
        return {"pass": 0, "block": 2, "hold": 3}[result["verdict"]]
    except (ValueError, OSError) as error:
        print(f"rollout-gate: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
