"""CPU contracts; synthetic features, not evidence of JSCAS accuracy."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from surgenet_dino.diagnostics import checkpoint_disagreement, probability_metrics
from surgenet_dino.gpu import CPUControl, Resources
from surgenet_dino.head import (EarlyStopping, RegularizedLinearHead,
                               evaluation_steps, head_from_checkpoint, make_head)
from surgenet_dino.head_options import HeadOptions, resolve_options
from surgenet_dino.head_training import fit
from surgenet_dino.training import LABELS, evaluate, head_step


@pytest.fixture(autouse=True)
def few_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("change", [
    {"epochs": 0}, {"epochs": True}, {"head_batch_size": -1}, {"evals_per_epoch": 0},
    {"patience": -1}, {"patience": 1.5}, {"learning_rate": 0}, {"learning_rate": float("nan")},
    {"weight_decay": -1}, {"dropout": 1}, {"dropout": -1}, {"label_smoothing": 1},
    {"min_delta": float("inf")},
    {"learning_rate": 100},
])
def test_invalid_options_rejected(change):
    with pytest.raises(ValueError):
        HeadOptions(**change)


def test_toml_cli_precedence(tmp_path):
    config = tmp_path / "head.toml"
    config.write_text('[training]\nlearning_rate = 0.02\ndropout = 0.2\n')
    args = SimpleNamespace(head_config=config, dropout=.3)
    options = resolve_options(args)
    assert options.learning_rate == .02 and options.dropout == .3
    assert options.weight_decay == HeadOptions().weight_decay
    config.write_text('[training]\nunknown = 1\n')
    with pytest.raises(ValueError, match="Unknown"):
        resolve_options(args)






def test_legacy_checkpoint_stays_usable_and_new_dimension_checked():
    head = torch.nn.Linear(5, 13).eval()
    x = torch.randn(7, 5)
    restored = head_from_checkpoint({"state_dict": head.state_dict()}, 5, 13, "cpu")
    torch.testing.assert_close(restored(x), head(x), rtol=0, atol=0)
    new = RegularizedLinearHead(5, 13, HeadOptions())
    with pytest.raises(ValueError, match="dimension"):
        head_from_checkpoint({"head_spec": new.specification(), "state_dict": new.state_dict()},
                             7, 13, "cpu")




def test_training_loss_smoothing():
    o = replace(HeadOptions(), dropout=0)
    h = RegularizedLinearHead(3, 13, o)
    logits = torch.randn(8, 13, requires_grad=True)
    y = torch.arange(8)
    expected = F.cross_entropy(logits, y, label_smoothing=.05)
    torch.testing.assert_close(h.training_loss(logits, y), expected)


def test_evaluation_is_ordinary_ce_and_dropout_off():
    h = make_head(torch.randn(26, 13), 13, "cpu", 42, replace(HeadOptions(), dropout=.9))
    x, y = torch.randn(26, 13), torch.arange(13).repeat(2)
    with CPUControl(torch.device("cpu"), Resources()) as guard:
        logits, loss = evaluate(h, x, y, guard, 7)
        assert not h.training
        assert loss == pytest.approx(F.cross_entropy(logits, y).item())
        other, _ = evaluate(h, x, y, guard, 7)
        torch.testing.assert_close(logits, other, rtol=0, atol=0)


@pytest.mark.parametrize("n,checks,wanted", [(1, 4, {1}), (3, 4, {1, 2, 3}),
                                            (872, 4, {218, 436, 654, 872}), (7, 3, {3, 5, 7})])
def test_evaluation_schedule(n, checks, wanted):
    assert evaluation_steps(n, checks) == wanted








def test_early_stopping_units_and_min_delta():
    stop = EarlyStopping(replace(HeadOptions(), patience=2, min_epochs=0, min_delta=.1))
    assert not stop.update(1., .25)
    assert not stop.update(.95, .5)  # a strict best, but not enough to reset patience
    assert stop.update(.94, .75)
    assert not stop.update(.8, 1.)
    no_stop = EarlyStopping(replace(HeadOptions(), patience=0))
    assert not any(no_stop.update(2., i) for i in range(30))


def tiny_features():
    y = torch.arange(13).repeat_interleave(4)
    gen = torch.Generator().manual_seed(18)
    x = F.one_hot(y, 13).float() + .03 * torch.randn(52, 13, generator=gen)
    vx = -x.clone()
    def rows(video):
        return [{"video_id": video, "filename": f"{video}_{i * 30:06d}.png",
                 "frame_index": i * 30, "phase_id": LABELS[t], "target": t}
                for i, t in enumerate(y.tolist())]
    return x, y, vx, y.clone(), rows("01"), rows("04")


def run_fit(tmp_path, options, step_fn=None, data=None):
    tmp_path.mkdir(exist_ok=False)
    data = tiny_features() if data is None else data
    with CPUControl(torch.device("cpu"), Resources()) as guard:
        return fit(*data, options, 42, guard, tmp_path, {"scope": "synthetic test"}, step_fn=step_fn)


def test_regularized_optimizer_parity(tmp_path):
    options = replace(HeadOptions(), epochs=3, head_batch_size=16, patience=0)
    data = tiny_features()
    run_fit(tmp_path / "new", options, data=data)
    x, y = data[:2]
    torch.manual_seed(42)
    original = torch.nn.Linear(13, 13)
    optim = torch.optim.AdamW([{ "params": [original.weight]},
                              {"params": [original.bias], "weight_decay": 0}], lr=.0003, weight_decay=.05)
    generator = torch.Generator().manual_seed(42)
    for _ in range(3):
        for ids in torch.randperm(len(x), generator=generator).split(16):
            optim.zero_grad(set_to_none=True)
            F.cross_entropy(original(F.dropout(x[ids], p=.1)), y[ids], label_smoothing=.05).backward()
            optim.step()
    saved = torch.load(tmp_path / "new/head_last.pt", weights_only=True)
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, saved["state_dict"]["linear." + key], rtol=0, atol=0)


def test_dropout_restored_after_each_subepoch_validation_and_cache_unchanged(tmp_path):
    data = tiny_features()
    originals = [v.clone() for v in data[:4]]
    seen = []
    def observe(head, optimizer, x, y, guard):
        assert head.training and head.dropout.training
        seen.append(len(x))
        return head_step(head, optimizer, x, y, guard)
    options = replace(HeadOptions(), epochs=2, head_batch_size=8, patience=0)
    result = run_fit(tmp_path / "dropout", options, step_fn=observe, data=data)
    assert len(result["history"]) == 9
    assert sum(seen) == 104
    assert result["reload_identical"]
    for old, new in zip(originals, data[:4]):
        torch.testing.assert_close(old, new, rtol=0, atol=0)


def test_subepoch_best_saved_and_early_stopping_does_not_save_requested_last_epoch(tmp_path):
    options = replace(HeadOptions(), epochs=20, head_batch_size=8, patience=2, min_epochs=0)
    result = run_fit(tmp_path / "early", options)
    assert result["stopped_early"] and result["last_epoch"] < 20
    expected = min(result["history"][1:], key=lambda r: r["validation_loss"])
    assert result["best_global_step"] == expected["global_step"]
    best = torch.load(tmp_path / "early/head_best.pt", weights_only=True)
    last = torch.load(tmp_path / "early/head_last.pt", weights_only=True)
    assert best["epoch"] == expected["epoch"]
    assert last["epoch"] == result["last_epoch"]
    assert best["epoch"] < 1
    assert len(list((tmp_path / "early").glob("*.partial"))) == 0
    assert (tmp_path / "early/validation_predictions_last.csv").is_file()




def test_bad_inputs_or_video_leakage_rejected(tmp_path):
    data = list(tiny_features())
    data[-1] = copy.deepcopy(data[-2])
    with pytest.raises(ValueError, match="leakage"):
        run_fit(tmp_path / "bad", HeadOptions(), data=data)


def test_probabilities_include_confidence_one_and_shared_wrong_cohort():
    logits = torch.tensor([[100., -100.], [100., -100.], [0., 0.]])
    targets = torch.tensor([0, 1, 1])
    m = probability_metrics(logits, targets)
    assert sum(b["count"] for b in m["ece_bins"]) == 3
    assert m["ece_bins"][-1]["count"] == 2
    assert m["wrong_count"] == 2 and m["nll"] > 60
    d = checkpoint_disagreement(logits, 2 * logits, targets)
    assert d["prediction_changes"] == 0
    assert d["shared_wrong_nll_last"] > d["shared_wrong_nll_best"]


def test_per_video_absent_class_metric_is_explicit():
    from surgenet_dino.training import classification_metrics
    logits = torch.zeros(4, 13)
    logits[:, 0] = 8
    metrics = classification_metrics(logits, torch.zeros(4, dtype=torch.long),
                                      [{"video_id": "01"}] * 4)
    assert metrics["macro_f1"] == pytest.approx(1 / 13)
    assert metrics["macro_f1_present"] == 1
    assert metrics["video_mean_macro_f1_present"] == 1
    assert metrics["per_video"]["01"]["nll"] > 0


def test_frozen_head_remains_linear_parameter_count():
    head = RegularizedLinearHead(768, 13, HeadOptions())
    assert sum(p.numel() for p in head.parameters()) == 9997


def test_best_checkpoint_survives_a_later_training_failure(tmp_path):
    calls = 0
    def crash(head, optimizer, x, y, guard):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated interruption")
        return head_step(head, optimizer, x, y, guard)
    with pytest.raises(RuntimeError, match="simulated"):
        run_fit(tmp_path / "interrupted", replace(HeadOptions(), head_batch_size=8), step_fn=crash)
    best = torch.load(tmp_path / "interrupted/head_best.pt", weights_only=True)
    assert best["global_step"] == 2
    head_from_checkpoint(best, 13, 13, "cpu")


def test_small_strict_improvements_saved_even_below_stopping_min_delta(tmp_path):
    data = list(tiny_features())
    data[2] = data[0].clone()
    options = replace(HeadOptions(), epochs=20, head_batch_size=8, patience=2, min_epochs=0, min_delta=10)
    result = run_fit(tmp_path / "min_delta", options, data=data)
    assert result["stopped_early"]
    assert result["best_global_step"] == result["total_steps"]
    assert result["history"][-1]["early_stop_bad_checks"] == 2


@pytest.mark.parametrize("legacy", [False, True])
def test_public_predict_head_respects_checkpoint_preprocessing(tmp_path, monkeypatch, legacy):
    from surgenet_dino import training
    from surgenet_dino.cli import make_parser
    from surgenet_dino.config import ModelConfig
    torch.manual_seed(4)
    x = 3 + 2 * torch.randn(32, 768)
    sample = 7 + torch.randn(1, 768)
    h = (torch.nn.Linear(768, 13) if legacy else
         make_head(x, 13, "cpu", 8, HeadOptions()))
    h.eval()
    identity = {"id": "mock_encoder_identity"}
    payload = {"labels": LABELS, "feature_identity": identity, "state_dict": h.state_dict()}
    if not legacy:
        payload["head_spec"] = h.specification()
    checkpoint = tmp_path / "head.pt"
    torch.save(payload, checkpoint)
    monkeypatch.setattr(training, "GPUControl", CPUControl)
    monkeypatch.setattr(training, "encoder", lambda *a: (torch.nn.Identity(), identity))
    monkeypatch.setattr(training, "extract_batch", lambda *a: sample.clone())
    output = tmp_path / "prediction"
    output.mkdir()
    args = make_parser().parse_args(["predict-head", "--head", str(checkpoint), "--image", "mock.png",
                                    "--output-dir", str(output), "--device", "cpu"])
    result = training.predict_head(args, ModelConfig(device="cpu"), output)
    actual = torch.tensor(list(result["predictions"][0]["scores"].values()))
    torch.testing.assert_close(actual, h(sample).softmax(1)[0], rtol=0, atol=0)


def test_plotter_accepts_head_only_directory_and_refuses_overwrite(tmp_path):
    pytest.importorskip("matplotlib")
    from scripts.plot_training_result import plot_run
    output = tmp_path / "plots_run"
    run_fit(output, replace(HeadOptions(), epochs=1, head_batch_size=8))
    plots = plot_run(output)
    assert {p.name for p in plots.glob("*.png")} == {
        "loss_curve.png", "accuracy_curve.png", "macro_f1_curve.png", "confidence_curve.png",
        "weight_norm_curve.png", "confusion_matrix.png"}
    with pytest.raises(FileExistsError):
        plot_run(output)


def test_historical_v2_prediction_and_unsupported_transform():
    from dataclasses import asdict
    h = RegularizedLinearHead(5, 13, HeadOptions()).eval()
    spec = {"schema_version": 2, "dimension": 5, "classes": 13,
            "options": {**asdict(HeadOptions()), "feature_norm": "none", "sampling": "frame",
                        "l2_strength": 0, "max_grad_norm": 0, "selection_metric": "loss",
                        "decay_bias": False, "temporal_bin_seconds": 5}}
    saved = {"head_spec": spec, "state_dict": {**h.state_dict(),
             "feature_mean": torch.zeros(5), "feature_scale": torch.ones(5)}}
    x = torch.randn(4, 5)
    restored = head_from_checkpoint(saved, 5, 13, "cpu")
    torch.testing.assert_close(restored(x), h(x), rtol=0, atol=0)
    spec["options"]["feature_norm"] = "rms"
    with pytest.raises(ValueError, match="Unsupported"):
        head_from_checkpoint(saved, 5, 13, "cpu")
    spec["options"]["feature_norm"] = "none"
    saved["state_dict"]["feature_scale"][0] = 2
    with pytest.raises(ValueError, match="buffers"):
        head_from_checkpoint(saved, 5, 13, "cpu")
