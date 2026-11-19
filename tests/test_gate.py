import hashlib
import json

import pytest
from pydantic import ValidationError

from rollout_gate.cli import main
from rollout_gate.evaluation import Policy, Prediction, evaluate


def fixture(priority_wrong=False, slow=False):
    return [
        Prediction(
            sample_id=str(i),
            segment="priority" if i < 40 else "standard",
            label=i % 2,
            champion=0.5,
            challenger=float(1 - i % 2) if i < 40 and priority_wrong else float(i % 2),
            champion_ms=10.0,
            challenger_ms=15.0 if slow else 10.0,
        )
        for i in range(240)
    ]


def policy(*segments):
    return Policy(required_segments=list(segments), bootstrap_draws=250)


def test_paired_confidence_and_order_invariance():
    rows = fixture()
    result = evaluate(rows, policy("priority"))
    assert result == evaluate(rows[::-1], policy("priority"))
    assert result["verdict"] == "pass"
    assert result["overall"]["lower_95"] == pytest.approx(0.25)


def test_required_slice_can_block_when_global_improves():
    result = evaluate(fixture(priority_wrong=True), policy("priority"))
    assert result["overall"]["lower_95"] > 0
    assert result["verdict"] == "block"
    assert result["failures"] == [
        "segment 'priority' Brier non-inferiority margin not met"
    ]


def test_latency_budget_and_absent_slice():
    result = evaluate(fixture(slow=True), policy("absent"))
    assert result["verdict"] == "block"
    assert result["latency"]["p95_ratio"] == 1.5
    assert result["insufficient_evidence"]


def test_required_slice_missing():
    result = evaluate(fixture(), policy("absent"))
    assert result["verdict"] == "hold"
    assert result["segments"]["absent"]["count"] == 0


def test_one_class_is_insufficient():
    result = evaluate(fixture()[::2], policy())
    assert result["verdict"] == "hold" and result["overall"] is None


def test_duplicate_ids_rejected():
    rows = fixture()
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(rows + rows[:1], policy())


@pytest.mark.parametrize(
    "key,value",
    [
        ("label", True),
        ("champion", 1.2),
        ("challenger_ms", float("inf")),
        ("label", "1"),
    ],
)
def test_prediction_schema_rejects_invalid_values(key, value):
    with pytest.raises(ValidationError):
        Prediction.model_validate({**fixture()[0].model_dump(), key: value})


@pytest.mark.parametrize("mode,expected", [("pass", 0), ("block", 2), ("hold", 3)])
def test_cli_report_digest_and_exit(tmp_path, mode, expected):
    rows = fixture(slow=mode == "block") if mode != "hold" else []
    raw = "\n".join(row.model_dump_json() for row in rows).encode()
    source = tmp_path / "input.jsonl"
    source.write_bytes(raw)
    output = tmp_path / "report.json"
    args = [str(source), "--out", str(output)]
    assert main(args) == expected
    report = json.loads(output.read_text())
    assert report["input_sha256"] == hashlib.sha256(raw).hexdigest()
    assert report["verdict"] == mode
    assert main(args) == 1
    assert json.loads(output.read_text()) == report


def test_cli_rejects_invalid_json_without_creating_report(tmp_path):
    source, output = tmp_path / "input", tmp_path / "output"
    source.write_bytes(b"not json\n")
    assert main([str(source), "--out", str(output)]) == 1
    assert not output.exists()
