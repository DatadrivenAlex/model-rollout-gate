"""Train two real scikit-learn models on seeded synthetic data; export held-out scores."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--regress", action="store_true", help="simulate a segment mapping bug"
    )
    args = parser.parse_args()
    rng = np.random.default_rng(49)
    x = rng.normal(size=(2500, 4))
    segment = rng.choice(["standard", "priority"], size=len(x))
    logits = x[:, 0] + 3 * x[:, 1] * x[:, 2] + 0.8 * (segment == "priority")
    y = rng.binomial(1, 1 / (1 + np.exp(-logits)))
    a = LogisticRegression().fit(x[:2000], y[:2000]).predict_proba(x[2000:])[:, 1]
    b = (
        HistGradientBoostingClassifier(random_state=49)
        .fit(x[:2000], y[:2000])
        .predict_proba(x[2000:])[:, 1]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        for i in range(len(a)):
            name = str(segment[2000 + i])
            score = float(b[i])
            if args.regress and name == "priority":
                score = 1 - score
            # Synthetic timings; not a benchmark of the estimators.
            row = {
                "sample_id": f"example-{i:04}",
                "segment": name,
                "label": int(y[2000 + i]),
                "champion": float(a[i]),
                "challenger": score,
                "champion_ms": 4.0,
                "challenger_ms": 4.4,
            }
            stream.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
