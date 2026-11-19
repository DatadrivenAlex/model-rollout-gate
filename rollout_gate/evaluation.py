"""Deterministic paired evaluation; positive Brier delta favors the challenger."""

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

Tag = Annotated[str, Field(min_length=1, max_length=80)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    sample_id: Annotated[str, Field(min_length=1, max_length=160)]
    segment: Tag
    label: StrictInt = Field(ge=0, le=1)
    champion: Probability
    challenger: Probability
    champion_ms: float = Field(gt=0, allow_inf_nan=False)
    challenger_ms: float = Field(gt=0, allow_inf_nan=False)


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    min_rows: int = Field(default=100, ge=20, le=10000)
    min_segment_rows: int = Field(default=40, ge=20, le=10000)
    required_segments: list[Tag] = Field(default_factory=list, max_length=30)
    max_global_regression: Probability = 0.005
    max_segment_regression: Probability = 0.025
    max_p95_latency_ratio: float = Field(default=1.25, gt=0, allow_inf_nan=False)
    bootstrap_draws: int = Field(default=1500, ge=250, le=5000)
    seed: int = Field(default=23, ge=0, le=2**32 - 1)

    @field_validator("required_segments")
    @classmethod
    def unique_segments(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values):
            raise ValueError("required_segments must be unique")
        return values


@dataclass(frozen=True)
class Metrics:
    count: int
    champion_brier: float
    challenger_brier: float
    delta: float
    lower_95: float
    upper_95: float


def paired_metrics(rows: list[Prediction], policy: Policy) -> Metrics:
    labels = np.array([r.label for r in rows])
    before = (np.array([r.champion for r in rows]) - labels) ** 2
    after = (np.array([r.challenger for r in rows]) - labels) ** 2
    delta = before - after
    # Bound memory at ~3 MiB per chunk even at the maximum dataset size.
    rng = np.random.default_rng(policy.seed)
    means = np.empty(policy.bootstrap_draws)
    for start in range(0, len(means), 32):
        count = min(32, len(means) - start)
        indices = rng.integers(0, len(rows), size=(count, len(rows)), dtype=np.int32)
        means[start : start + count] = delta[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return Metrics(
        len(rows),
        float(before.mean()),
        float(after.mean()),
        float(delta.mean()),
        float(low),
        float(high),
    )


def evaluate(rows: list[Prediction], policy: Policy) -> dict:

    if len(rows) > 10000:
        raise ValueError("at most 10000 predictions are supported")
    rows = sorted(rows, key=lambda r: r.sample_id)
    ids = Counter(r.sample_id for r in rows)
    if any(count != 1 for count in ids.values()):
        raise ValueError(
            "duplicate sample_id; every row must represent a distinct paired sample"
        )
    groups: dict[str, list[Prediction]] = {}
    for row in rows:
        groups.setdefault(row.segment, []).append(row)
    evidence: list[str] = []
    failures: list[str] = []
    overall = None
    slices: dict[str, dict] = {}
    latency = None
    if len(rows) < policy.min_rows or len({r.label for r in rows}) < 2:
        evidence.append("overall sample lacks minimum count or both label classes")
    else:
        overall = paired_metrics(rows, policy)
        if overall.lower_95 < -policy.max_global_regression:
            failures.append("overall Brier non-inferiority margin not met")
        before, after = (
            float(np.quantile([getattr(r, field) for r in rows], 0.95))
            for field in ("champion_ms", "challenger_ms")
        )
        with np.errstate(over="ignore"):
            ratio = float(np.divide(after, before))
        latency = {
            "champion_p95_ms": before,
            "challenger_p95_ms": after,
            "p95_ratio": ratio if np.isfinite(ratio) else None,
        }
        if not np.isfinite(ratio) or ratio > policy.max_p95_latency_ratio:
            failures.append("challenger p95 latency ratio exceeded")
    for name in sorted(set(groups) | set(policy.required_segments)):
        members = groups.get(name, [])
        required = name in policy.required_segments
        item: dict = {"count": len(members), "required": required, "metrics": None}
        if (
            len(members) < policy.min_segment_rows
            or len({r.label for r in members}) < 2
        ):
            if required:
                evidence.append(
                    f"segment {name!r} lacks minimum count or both label classes"
                )
        else:
            stats = paired_metrics(members, policy)
            item["metrics"] = asdict(stats)
            if required and stats.lower_95 < -policy.max_segment_regression:
                failures.append(
                    f"segment {name!r} Brier non-inferiority margin not met"
                )
        slices[name] = item
    verdict: Literal["hold", "block", "pass"] = (
        "block" if failures else "hold" if evidence else "pass"
    )
    return {
        "schema_version": 1,
        "verdict": verdict,
        "failures": failures,
        "insufficient_evidence": evidence,
        "overall": asdict(overall) if overall else None,
        "segments": slices,
        "latency": latency,
        "policy": policy.model_dump(),
    }
