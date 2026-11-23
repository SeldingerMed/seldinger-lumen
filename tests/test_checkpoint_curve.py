import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.external_comparison import checkpoint_curve as cc


def _args(tmp_path: Path, **kw):
    base = dict(environment="lumen", task="phantom3_bca", seed=0, ckpt_dir=str(tmp_path / "ck"),
                final_model=str(tmp_path / "final.zip"), timesteps=100, episodes=10, max_steps=5,
                threshold=0.9, stop_at_first=False, out=str(tmp_path / "curve.json"))
    base.update(kw)
    return argparse.Namespace(**base)


def _setup(tmp_path: Path):
    ck = tmp_path / "ck"
    ck.mkdir()
    for s in (25, 50, 75, 100):            # the snapshot at the budget is never used
        (ck / f"ckpt_{s}_steps.zip").write_bytes(f"m{s}".encode())
    (ck / "._ckpt_50_steps.zip").write_bytes(b"appledouble")
    (tmp_path / "final.zip").write_bytes(b"final")


def _fake(rates, calls):
    """Evaluator honouring max_failures like train_sb3.evaluate_model: episodes in order, stop after too many fails."""
    def evaluate(eargs, model, *, training_seed, model_id, max_failures):
        calls.append(model_id)
        k = round(rates[model_id] * eargs.eval_episodes)
        eps = []
        for i in range(eargs.eval_episodes):
            if sum(not e.success for e in eps) > max_failures:
                break
            eps.append(SimpleNamespace(success=i < k, steps=3, crashed=False))
        return eps
    return evaluate


def test_curve_early_stop_first_reach_and_resume(tmp_path: Path):
    _setup(tmp_path)
    rates = {"ckpt_25_steps.zip": 0.0, "ckpt_50_steps.zip": 0.5, "ckpt_75_steps.zip": 0.9, "final.zip": 1.0}
    calls = []
    rec = cc.run(_args(tmp_path), evaluate=_fake(rates, calls), load=lambda p: p)
    pts = {p["steps"]: p for p in rec["points"]}
    assert sorted(pts) == [25, 50, 75, 100]
    assert pts[100]["model"] == "final.zip"                     # final model stands for the budget
    assert pts[25]["exact"] is False and pts[25]["episodes_evaluated"] == 2   # stopped once 90% unreachable
    assert pts[50]["exact"] is False and pts[50]["success_rate"] < 0.9
    assert pts[75]["exact"] is True and pts[75]["success_rate"] == pytest.approx(0.9)
    assert rec["first_reach_steps"] == 75
    # rerun skips everything already scored
    calls.clear()
    cc.run(_args(tmp_path), evaluate=_fake(rates, calls), load=lambda p: p)
    assert calls == []
    assert json.loads((tmp_path / "curve.json").read_text())["first_reach_steps"] == 75


def test_curve_not_reached_and_identity_guard(tmp_path: Path):
    _setup(tmp_path)
    rates = {"ckpt_25_steps.zip": 0.0, "ckpt_50_steps.zip": 0.3, "ckpt_75_steps.zip": 0.8, "final.zip": 0.85}
    rec = cc.run(_args(tmp_path), evaluate=_fake(rates, []), load=lambda p: p)
    assert rec["first_reach_steps"] is None
    with pytest.raises(ValueError, match="different evaluation"):
        cc.run(_args(tmp_path, seed=1), evaluate=_fake(rates, []), load=lambda p: p)


def test_stop_at_first(tmp_path: Path):
    _setup(tmp_path)
    rates = {"ckpt_25_steps.zip": 1.0, "ckpt_50_steps.zip": 1.0, "ckpt_75_steps.zip": 1.0, "final.zip": 1.0}
    calls = []
    rec = cc.run(_args(tmp_path, stop_at_first=True), evaluate=_fake(rates, calls), load=lambda p: p)
    assert calls == ["ckpt_25_steps.zip"] and rec["first_reach_steps"] == 25
