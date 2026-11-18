"""Unit-, інтеграційні та регресійні тести модуля класифікації MLP (`src/classification.py`)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.run_feature_window_study import build_features, load_csv as load_study_csv
from src.classification import (
    DEFAULT_MODEL_PATH,
    DEFAULT_WINDOW_SIZE,
    MLPClassifier,
    apply_normalizer,
    block_split_indices,
    classify_frames,
    compute_frame_values,
    extract_features_from_raw,
    extract_features_matrix,
    extract_window_features,
    fit_normalizer,
    init_mlp_params,
    load_model,
    macro_f1,
    mlp_forward,
    mlp_loss_and_grads,
    save_model,
    softmax,
    train_single_seed,
)
from src.telemetry import TelemetryFrame, load_session_bin

REPO_ROOT = Path(__file__).resolve().parent.parent
TEAM_DIR = REPO_ROOT / "team"


def _make_frame(alt: int, temp: int, vib: int, mode: int, critical: int = 0, ts: int = 0) -> TelemetryFrame:
    return TelemetryFrame(
        timestamp_ms=ts,
        altitude_dm=alt,
        temp_c10=temp,
        vibration=vib,
        mode=mode,
        critical=critical,
    )


def test_numerical_gradient():
    """1. Чисельна перевірка аналітичного градієнта (центральна різниця, float64)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 7)).astype(np.float64)
    y = rng.integers(0, 3, size=12, dtype=np.int64)
    params = init_mlp_params(in_dim=7, hidden=10, out_dim=3, seed=1)
    for k, arr in params.items():
        if k.startswith("b"):
            params[k] = rng.normal(size=arr.shape) * 0.1

    _, ana = mlp_loss_and_grads(params, X, y)
    eps = 1e-6
    worst_rel_err = 0.0
    for k, arr in params.items():
        for idx in np.ndindex(arr.shape):
            old = arr[idx]
            arr[idx] = old + eps
            lp, _ = mlp_loss_and_grads(params, X, y)
            arr[idx] = old - eps
            lm, _ = mlp_loss_and_grads(params, X, y)
            arr[idx] = old
            num = (lp - lm) / (2.0 * eps)
            rel_err = abs(num - ana[k][idx]) / max(1e-8, abs(num) + abs(ana[k][idx]))
            worst_rel_err = max(worst_rel_err, rel_err)

    assert worst_rel_err < 1e-5, f"Відносна похибка градієнта завелика: {worst_rel_err:.2e}"


def test_forward_manual_weights_and_softmax_stability():
    """2. Прямий прохід на ручних вагах та числова стійкість softmax."""
    params = {
        "W1": np.array([[1.0, -1.0], [2.0, 0.5]], dtype=np.float64),
        "b1": np.array([0.0, 1.0], dtype=np.float64),
        "W2": np.array([[1.0, 0.0, -1.0], [0.0, 2.0, 1.0]], dtype=np.float64),
        "b2": np.array([0.5, -0.5, 0.0], dtype=np.float64),
    }
    X = np.array([[1.0, 2.0]], dtype=np.float64)
    # z1 = [5.0, 1.0] -> h = [5.0, 1.0] -> logits = [5.5, 1.5, -4.0]
    h, logits, probs = mlp_forward(params, X)
    assert np.allclose(h, [[5.0, 1.0]])
    assert np.allclose(logits, [[5.5, 1.5, -4.0]])
    assert np.isclose(probs.sum(), 1.0)
    assert int(np.argmax(probs, axis=1)[0]) == 0

    extreme = softmax(np.array([[1000.0, 0.0, -1000.0], [-1000.0, -1000.0, -1000.0]]))
    assert np.isfinite(extreme).all()
    assert np.allclose(extreme.sum(axis=1), [1.0, 1.0])


def test_loss_decreases_and_no_severe_overfit():
    """3. Loss спадає на train-частині та немає сильного перенавчання."""
    train = load_study_csv(str(TEAM_DIR / "train.csv"))
    X = extract_features_from_raw(train["raw"], window_size=DEFAULT_WINDOW_SIZE)
    y = train["label"]
    tr_idx, va_idx = block_split_indices(len(y), window_size=DEFAULT_WINDOW_SIZE)

    res = train_single_seed(
        X[tr_idx],
        y[tr_idx],
        X[va_idx],
        y[va_idx],
        seed=0,
        epochs=15,
        patience=15,
    )
    assert res["final_loss"] < res["init_loss"]
    assert res["loss_history"][-1] < res["loss_history"][0]
    assert (res["train_f1"] - res["val_f1"]) < 0.08


def test_block_split_disjoint_and_no_window_leak():
    """4. train ∩ val = ∅ і вікна train та val не перекриваються."""
    w = DEFAULT_WINDOW_SIZE
    past = w - 1
    tr_idx, va_idx = block_split_indices(8000, window_size=w, train_ratio=0.8)
    assert set(tr_idx.tolist()).isdisjoint(set(va_idx.tolist()))
    # Максимальний кадр, що бачить train (idx), менший за мінімальний кадр вікна val (idx - past)
    assert int(tr_idx.max()) < int(va_idx.min()) - past


def test_macro_f1_edge_cases():
    """5. macro-F1 при P+R=0, ідеальному прогнозі та колапсі в один клас."""
    y = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    m_collapse, per_c = macro_f1(y, np.zeros(6, dtype=np.int64))
    assert np.isfinite(m_collapse)
    assert per_c[1] == 0.0 and per_c[2] == 0.0
    assert np.isclose(per_c[0], 0.5)
    assert np.isclose(m_collapse, 0.5 / 3.0)

    m_perf, per_perf = macro_f1(y, y)
    assert m_perf == 1.0 and per_perf == [1.0, 1.0, 1.0]


def test_window_features_known_and_causal():
    """6. Відомі значення ознак, еквівалентність extract_window_features та causal-властивість."""
    frames = [
        _make_frame(100, 10, 0, 1, ts=0),
        _make_frame(101, 10, 0, 1, ts=50),
        _make_frame(103, 12, 4, 2, ts=100),
        _make_frame(103, 12, -4, 2, ts=150),
    ]
    X = extract_features_matrix(frames, window_size=4)
    assert np.allclose(X[0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.isclose(X[3, 0], 0.0)
    assert np.isclose(X[3, 1], 0.0)
    assert np.isclose(X[3, 2], 4.0)
    assert np.isclose(X[3, 3], np.std([0, 0, 4, -4]))
    assert np.isclose(X[3, 4], np.std([100, 101, 103, 103]))
    assert np.isclose(X[3, 5], np.std([10, 10, 12, 12]))
    assert np.isclose(X[3, 6], 1.0 / 3.0)

    for idx in range(len(frames)):
        assert np.allclose(extract_window_features(frames, idx, window_size=4), X[idx])

    rng = np.random.default_rng(7)
    raw = rng.normal(size=(40, 5)) * 10.0
    raw[:, 3] = rng.integers(0, 8, size=40)
    A = extract_features_from_raw(raw, window_size=DEFAULT_WINDOW_SIZE)
    raw_corrupted = raw.copy()
    raw_corrupted[21:] += 999.0
    B = extract_features_from_raw(raw_corrupted, window_size=DEFAULT_WINDOW_SIZE)
    assert np.allclose(A[:21], B[:21])


def test_features_match_experiment_script():
    """7. Ознаки з src/classification.py на team/train.csv збігаються з run_feature_window_study.py."""
    train = load_study_csv(str(TEAM_DIR / "train.csv"))
    X_src = extract_features_from_raw(train["raw"], window_size=DEFAULT_WINDOW_SIZE)
    cfg = {"kind": "hand", "scales": [(DEFAULT_WINDOW_SIZE, "c")], "extras": (), "drop": ()}
    X_exp, _, _, _ = build_features(train["raw"], cfg, DEFAULT_WINDOW_SIZE)
    assert X_src.shape == X_exp.shape
    assert np.allclose(X_src, X_exp, atol=1e-12, rtol=1e-12)


def test_normalizer_train_only_and_constant_feature():
    """8. Параметри нормалізації рахуються по train; константна ознака не дає NaN/inf."""
    rng = np.random.default_rng(3)
    X_train = np.column_stack([rng.normal(10.0, 2.0, 100), np.full(100, 5.0)])
    X_other = np.column_stack([rng.normal(25.0, 2.0, 50), np.full(50, 5.0)])

    mean, std = fit_normalizer(X_train)
    assert std[1] == 0.0
    Zn_train = apply_normalizer(X_train, mean, std)
    Zn_other = apply_normalizer(X_other, mean, std)

    assert np.isfinite(Zn_train).all() and np.isfinite(Zn_other).all()
    assert np.isclose(Zn_train[:, 0].mean(), 0.0, atol=1e-9)
    assert np.isclose(Zn_train[:, 0].std(), 1.0, atol=1e-6)
    assert np.allclose(Zn_train[:, 1], 0.0)
    assert abs(Zn_other[:, 0].mean()) > 3.0


def test_save_and_load_roundtrip(tmp_path: Path):
    """9. Збереження та завантаження моделі дають побітово однакові прогнози на сеансі."""
    params = init_mlp_params(in_dim=7, hidden=8, out_dim=3, seed=42)
    mean = np.zeros(7, dtype=np.float64)
    std = np.ones(7, dtype=np.float64)
    model_file = tmp_path / "test_model.npz"

    save_model(str(model_file), params, mean, std, {"version": "test_v1"})
    bundle = load_model(str(model_file))

    frames = load_session_bin(str(TEAM_DIR / "session.bin"))[:120]
    preds_before = MLPClassifier(params, mean, std, window_size=DEFAULT_WINDOW_SIZE).predict_frames(frames)
    preds_after = classify_frames(frames, model_path=str(model_file))
    assert preds_before == preds_after
    assert bundle["metadata"]["version"] == "test_v1"


def test_reproducibility_and_valid_quality_gate():
    """10. Відтворюваність за seed, якість навченої моделі >= 0.90 (F1_min=0.58) та класифікація session.bin."""
    train = load_study_csv(str(TEAM_DIR / "train.csv"))
    valid = load_study_csv(str(TEAM_DIR / "valid.csv"))
    X_tr_all = extract_features_from_raw(train["raw"], window_size=DEFAULT_WINDOW_SIZE)
    tr_idx, va_idx = block_split_indices(len(train["label"]), window_size=DEFAULT_WINDOW_SIZE)

    r1 = train_single_seed(
        X_tr_all[tr_idx], train["label"][tr_idx],
        X_tr_all[va_idx], train["label"][va_idx],
        seed=2, epochs=5, patience=5,
    )
    r2 = train_single_seed(
        X_tr_all[tr_idx], train["label"][tr_idx],
        X_tr_all[va_idx], train["label"][va_idx],
        seed=2, epochs=5, patience=5,
    )
    assert r1["val_f1"] == r2["val_f1"]
    for k in r1["params"]:
        assert np.array_equal(r1["params"][k], r2["params"][k])

    bundle = load_model(DEFAULT_MODEL_PATH)
    X_valid = extract_features_from_raw(valid["raw"], window_size=bundle["window_size"])
    valid_preds = bundle["classifier"].predict(X_valid)
    valid_f1, _ = macro_f1(valid["label"], valid_preds)
    assert valid_f1 >= 0.58
    assert valid_f1 >= 0.90, f"Очікували macro-F1 >= 0.90 на valid.csv, отримано {valid_f1:.4f}"

    session_frames = load_session_bin(str(TEAM_DIR / "session.bin"))
    session_preds = classify_frames(session_frames)
    assert len(session_preds) == len(session_frames) == 3000
    assert set(session_preds).issubset({0, 1, 2})


def test_compute_frame_values_mapping_and_errors():
    """11. Точний мапінг пріоритетів із team/team_config.json та явна помилка при збоях."""
    with open(TEAM_DIR / "team_config.json", encoding="utf-8") as fh:
        cfg = json.load(fh)

    frames = [
        _make_frame(1000, 200, 0, 1, critical=0),
        _make_frame(1000, 200, 0, 1, critical=1),
        _make_frame(1000, 200, 400, 1, critical=0),
        _make_frame(1000, 200, 20, 2, critical=1),
    ]
    # норма=1, аномалія=5, перехідний_режим=3, critical_бонус=5
    values = compute_frame_values(frames, [0, 0, 1, 2], cfg)
    assert values == [1, 6, 5, 8]

    with pytest.raises(ValueError):
        compute_frame_values([frames[0]], [3], cfg)
    with pytest.raises(ValueError):
        compute_frame_values([frames[0]], [-1], cfg)
    with pytest.raises(ValueError):
        compute_frame_values([frames[0]], [True], cfg)  # type: ignore[list-item]
    with pytest.raises(ValueError):
        compute_frame_values(frames[:2], [0], cfg)

    bad_cfg = {"пріоритети_кадрів": {"норма": 1, "аномалія": 5, "critical_бонус": 5}}
    with pytest.raises(ValueError):
        compute_frame_values([frames[0]], [0], bad_cfg)
