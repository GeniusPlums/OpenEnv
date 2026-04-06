"""Tests for the inference script — validates log format and deterministic scoring."""

from __future__ import annotations

import io
import os
import re
from contextlib import redirect_stdout
from unittest import mock


def _capture_inference_output() -> str:
    """Run inference in deterministic mode (no API key) and capture stdout."""
    env_vars = {
        "HF_TOKEN": "",
        "OPENAI_API_KEY": "",
        "LEADQUALENV_SEED": "0",
        "LEADQUALENV_MAX_STEPS": "10",
    }
    buf = io.StringIO()
    with mock.patch.dict(os.environ, env_vars, clear=False):
        import inference
        with redirect_stdout(buf):
            inference.main()
    return buf.getvalue()


def test_inference_log_format():
    """All lines must match [START], [STEP], or [END] format."""
    output = _capture_inference_output()
    lines = [line for line in output.strip().split("\n") if line.strip()]

    start_pattern = re.compile(r"^\[START\] task=\w+ env=\w+ model=.+$")
    step_pattern = re.compile(r"^\[STEP\] step=\d+ action=.+ reward=-?\d+\.\d+ done=(true|false) error=\S+.*$")
    buyer_pattern = re.compile(r"^\[BUYER\] step=\d+ response=.+$")
    end_pattern = re.compile(r"^\[END\] success=(true|false) steps=\d+ score=\d+\.\d+ rewards=.+$")

    for line in lines:
        line = line.strip()
        assert (
            start_pattern.match(line) or step_pattern.match(line) or buyer_pattern.match(line) or end_pattern.match(line)
        ), f"Line does not match any expected format: {line!r}"


def test_inference_produces_four_tasks():
    """Default run should produce exactly 4 [START] and 4 [END] lines."""
    output = _capture_inference_output()
    starts = [line for line in output.split("\n") if line.strip().startswith("[START]")]
    ends = [line for line in output.split("\n") if line.strip().startswith("[END]")]
    assert len(starts) == 4, f"Expected 4 [START] lines, got {len(starts)}"
    assert len(ends) == 4, f"Expected 4 [END] lines, got {len(ends)}"


def test_inference_scores_in_range():
    """All scores must be in [0.0, 1.0]."""
    output = _capture_inference_output()
    for line in output.split("\n"):
        if line.strip().startswith("[END]"):
            score_match = re.search(r"score=(\d+\.\d+)", line)
            assert score_match, f"No score found in END line: {line!r}"
            score = float(score_match.group(1))
            assert 0.0 <= score <= 1.0, f"Score {score} out of [0, 1] range"


def test_inference_all_tasks_succeed_deterministic():
    """Deterministic fallback should produce successes for all except hard mode."""
    output = _capture_inference_output()
    starts = [line for line in output.split("\n") if line.strip().startswith("[START]")]
    ends = [line for line in output.split("\n") if line.strip().startswith("[END]")]
    for start, end in zip(starts, ends):
        if "task=hard" not in start:
            assert "success=true" in end, f"Expected success=true for non-hard deterministic mode: {end!r}"


def test_inference_step_count_reasonable():
    """Each task should complete in 10 or fewer steps."""
    output = _capture_inference_output()
    for line in output.split("\n"):
        if line.strip().startswith("[END]"):
            steps_match = re.search(r"steps=(\d+)", line)
            assert steps_match
            steps = int(steps_match.group(1))
            assert 1 <= steps <= 10, f"Step count {steps} outside expected range"
