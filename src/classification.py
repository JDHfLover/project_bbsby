"""Модуль гілки `classification`: 7 віконних ознак (w=5), самописний MLP та пріоритети кадрів."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.telemetry import TelemetryFrame

FEATURE_NAMES = ("dalt", "dtmp", "absvib", "std_vib", "std_alt", "std_temp", "mfreq")
DEFAULT_WINDOW_SIZE = 5
DEFAULT_MODEL_PATH = "models/classifier_v1.npz"
NUM_CLASSES = 3
CLASS_TO_KEY = {0: "норма", 1: "аномалія", 2: "перехідний_режим"}

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODEL_CACHE: dict[tuple[str, int, int], dict[str, Any]] = {}


def frames_to_raw_array(frames: list[TelemetryFrame]) -> np.ndarray:
    """Перетворює список TelemetryFrame у матрицю (N, 5) float64."""
    if not frames:
        return np.empty((0, 5), dtype=np.float64)
    return np.asarray(
        [
            [f.altitude_dm, f.temp_c10, f.vibration, f.mode, f.critical]
            for f in frames
        ],
        dtype=np.float64,
    )


def extract_features_from_raw(
    raw: np.ndarray, window_size: int = DEFAULT_WINDOW_SIZE
) -> np.ndarray:
    """Обчислює 7 causal-ознак по вікну `window_size` з матриці (N, 5)."""
    if window_size < 2:
        raise ValueError(f"window_size має бути >= 2, отримано {window_size}")
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 5:
        raise ValueError(f"Очікувалась матриця форми (N, 5), отримано {arr.shape}")
    if len(arr) == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("Вхідна телеметрія містить NaN або inf")

    past = window_size - 1
    pad = np.vstack([np.repeat(arr[:1], past, axis=0), arr])
    idx = np.arange(len(arr))[:, None] + np.arange(window_size)[None, :]
    win = pad[idx]

    alt = win[:, :, 0]
    tmp = win[:, :, 1]
    vib = win[:, :, 2]
    mod = win[:, :, 3]
    cur = past

    features = np.column_stack(
        [
            alt[:, cur] - alt[:, cur - 1],
            tmp[:, cur] - tmp[:, cur - 1],
            np.abs(vib[:, cur]),
            vib.std(axis=1, ddof=0),
            alt.std(axis=1, ddof=0),
            tmp.std(axis=1, ddof=0),
            (np.diff(mod, axis=1) != 0).mean(axis=1),
        ]
    ).astype(np.float64)

    if not np.isfinite(features).all():
        raise ValueError("Обчислені ознаки містять NaN або inf")
    return features


def extract_features_matrix(
    frames: list[TelemetryFrame], window_size: int = DEFAULT_WINDOW_SIZE
) -> np.ndarray:
    """Будує матрицю ознак (N, 7) для списку TelemetryFrame."""
    return extract_features_from_raw(frames_to_raw_array(frames), window_size)


def extract_window_features(
    frames: list[TelemetryFrame], idx: int, window_size: int = DEFAULT_WINDOW_SIZE
) -> list[float]:
    """Повертає 7 віконних ознак для кадру `idx` із доповненням краю першим кадром."""
    if not frames:
        raise ValueError("Порожній список кадрів")
    if not (0 <= idx < len(frames)):
        raise IndexError(f"Індекс {idx} поза межами [0, {len(frames)})")
    if window_size < 2:
        raise ValueError(f"window_size має бути >= 2, отримано {window_size}")

    start = max(0, idx - window_size + 1)
    pad_count = max(0, window_size - 1 - idx)
    sub = [frames[0]] * pad_count + frames[start : idx + 1]
    feats = extract_features_from_raw(frames_to_raw_array(sub), window_size)
    return [float(val) for val in feats[-1]]


def block_split_indices(
    n: int, window_size: int = DEFAULT_WINDOW_SIZE, train_ratio: float = 0.8
) -> tuple[np.ndarray, np.ndarray]:
    """Повертає (train_idx, val_idx) для блокового спліту з розривом `window_size - 1`."""
    if window_size < 2:
        raise ValueError(f"window_size має бути >= 2, отримано {window_size}")
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio має бути в (0, 1), отримано {train_ratio}")

    past = window_size - 1
    n_tr = int(n * train_ratio)
    if n_tr <= 2 * past or n_tr >= n:
        raise ValueError(f"Недостатньо рядків ({n}) для спліту з вікном {window_size}")

    tr_idx = np.arange(past, n_tr - past, dtype=np.int64)
    va_idx = np.arange(n_tr, n, dtype=np.int64)
    return tr_idx, va_idx


def fit_normalizer(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Обчислює (mean, std) по train-частині для z-score нормалізації."""
    arr = np.asarray(X_train, dtype=np.float64)
    if arr.ndim != 2 or len(arr) == 0:
        raise ValueError("X_train має бути непорожньою 2D-матрицею")
    return arr.mean(axis=0), arr.std(axis=0, ddof=0)


def apply_normalizer(
    X: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Застосовує z-score `(X - mean) / (std + 1e-8)`."""
    arr = np.asarray(X, dtype=np.float64)
    if len(arr) == 0:
        return arr.copy()
    return (arr - mean) / (std + 1e-8)


def macro_f1(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES
) -> tuple[float, list[float]]:
    """Обчислює macro-F1 і список F1 по класах (P+R=0 -> F1=0.0)."""
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    if yt.shape != yp.shape:
        raise ValueError(f"Невідповідні форми міток: {yt.shape} vs {yp.shape}")

    f1s: list[float] = []
    for cls in range(num_classes):
        tp = int(((yp == cls) & (yt == cls)).sum())
        fp = int(((yp == cls) & (yt != cls)).sum())
        fn = int(((yp != cls) & (yt == cls)).sum())
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append((2.0 * pr * rc / (pr + rc)) if (pr + rc) > 0 else 0.0)
    return float(np.mean(f1s)), f1s


def confusion_matrix_3x3(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES
) -> np.ndarray:
    """Будує матрицю помилок (num_classes, num_classes): рядок=істина, стовпець=прогноз."""
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    if yt.shape != yp.shape:
        raise ValueError(f"Невідповідні форми міток: {yt.shape} vs {yp.shape}")

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_c, pred_c in zip(yt, yp):
        if 0 <= true_c < num_classes and 0 <= pred_c < num_classes:
            cm[true_c, pred_c] += 1
    return cm


def softmax(z: np.ndarray) -> np.ndarray:
    """Чисельно стійкий softmax по рядках матриці логітів."""
    arr = np.asarray(z, dtype=np.float64)
    shifted = arr - arr.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def init_mlp_params(
    in_dim: int = len(FEATURE_NAMES),
    hidden: int = 32,
    out_dim: int = NUM_CLASSES,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """He-ініціалізація ваг і нульові зсуви для 1-прихованого шару ReLU."""
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0.0, np.sqrt(2.0 / in_dim), (in_dim, hidden)),
        "b1": np.zeros(hidden, dtype=np.float64),
        "W2": rng.normal(0.0, np.sqrt(2.0 / hidden), (hidden, out_dim)),
        "b2": np.zeros(out_dim, dtype=np.float64),
    }


def mlp_forward(
    params: dict[str, np.ndarray], X: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Прямий прохід `X -> (h, logits, probs)` для ReLU + softmax."""
    z1 = X @ params["W1"] + params["b1"]
    h = np.maximum(0.0, z1)
    logits = h @ params["W2"] + params["b2"]
    probs = softmax(logits)
    return h, logits, probs


def mlp_loss_and_grads(
    params: dict[str, np.ndarray], X: np.ndarray, y: np.ndarray
) -> tuple[float, dict[str, np.ndarray]]:
    """Обчислює cross-entropy loss та аналітичні градієнти (backprop)."""
    n = len(y)
    if n == 0:
        raise ValueError("Неможливо обчислити loss на порожньому батчі")

    z1 = X @ params["W1"] + params["b1"]
    h = np.maximum(0.0, z1)
    logits = h @ params["W2"] + params["b2"]
    probs = softmax(logits)

    loss = float(-np.log(probs[np.arange(n), y] + 1e-12).mean())

    g = probs.copy()
    g[np.arange(n), y] -= 1.0
    g /= n

    dW2 = h.T @ g
    db2 = g.sum(axis=0)
    dh = g @ params["W2"].T
    dz1 = dh * (z1 > 0)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)

    return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def train_single_seed(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    seed: int = 0,
    hidden: int = 32,
    lr: float = 1e-3,
    batch_size: int = 64,
    epochs: int = 100,
    patience: int = 15,
) -> dict[str, Any]:
    """Навчає один екземпляр MLP через Adam з ранньою зупинкою за val macro-F1."""
    if epochs < 1:
        raise ValueError(f"epochs має бути >= 1, отримано {epochs}")
    if len(X_tr) == 0 or len(X_va) == 0:
        raise ValueError("train і val вибірки мають бути непорожніми")

    mean, std = fit_normalizer(X_tr)
    Xt = apply_normalizer(X_tr, mean, std)
    Xv = apply_normalizer(X_va, mean, std)
    yt = np.asarray(y_tr, dtype=np.int64)
    yv = np.asarray(y_va, dtype=np.int64)

    params = init_mlp_params(Xt.shape[1], hidden, NUM_CLASSES, seed=seed)
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v2 = {k: np.zeros_like(v) for k, v in params.items()}
    rng = np.random.default_rng(10_000 + seed)

    init_loss, _ = mlp_loss_and_grads(params, Xt, yt)
    step = 0
    best_val_f1 = -1.0
    best_params: dict[str, np.ndarray] | None = None
    best_epoch = 0
    bad_epochs = 0
    loss_history: list[float] = []
    epochs_run = 0

    for ep in range(epochs):
        epochs_run = ep + 1
        perm = rng.permutation(len(Xt))
        for start in range(0, len(Xt), batch_size):
            batch_idx = perm[start : start + batch_size]
            _, grads = mlp_loss_and_grads(params, Xt[batch_idx], yt[batch_idx])
            step += 1
            b1_corr = 1.0 - 0.9**step
            b2_corr = 1.0 - 0.999**step
            for k in params:
                m[k] = 0.9 * m[k] + 0.1 * grads[k]
                v2[k] = 0.999 * v2[k] + 0.001 * (grads[k] ** 2)
                params[k] -= lr * (m[k] / b1_corr) / (np.sqrt(v2[k] / b2_corr) + 1e-8)

        ep_loss, _ = mlp_loss_and_grads(params, Xt, yt)
        loss_history.append(ep_loss)

        _, va_logits, _ = mlp_forward(params, Xv)
        val_f1, _ = macro_f1(yv, np.argmax(va_logits, axis=1))
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = ep
            bad_epochs = 0
            best_params = {k: arr.copy() for k, arr in params.items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    assert best_params is not None
    _, tr_logits, _ = mlp_forward(best_params, Xt)
    train_f1, _ = macro_f1(yt, np.argmax(tr_logits, axis=1))
    final_loss, _ = mlp_loss_and_grads(best_params, Xt, yt)

    return {
        "seed": seed,
        "params": best_params,
        "mean": mean,
        "std": std,
        "val_f1": best_val_f1,
        "train_f1": train_f1,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "init_loss": init_loss,
        "final_loss": final_loss,
        "loss_history": loss_history,
    }


class MLPClassifier:
    """Обгортка навченого 1-прихованого MLP з інтегрованим z-score нормалізатором."""

    def __init__(
        self,
        params: dict[str, np.ndarray],
        mean: np.ndarray,
        std: np.ndarray,
        window_size: int = DEFAULT_WINDOW_SIZE,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.params = {k: np.asarray(v, dtype=np.float64) for k, v in params.items()}
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.window_size = int(window_size)
        self.metadata = dict(metadata) if metadata is not None else {}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Повертає ймовірності класів (N, 3) для матриці ознак X."""
        if len(X) == 0:
            return np.empty((0, NUM_CLASSES), dtype=np.float64)
        Xn = apply_normalizer(X, self.mean, self.std)
        _, _, probs = mlp_forward(self.params, Xn)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Повертає вектор передбачених класів (N,) типу int64."""
        if len(X) == 0:
            return np.empty(0, dtype=np.int64)
        Xn = apply_normalizer(X, self.mean, self.std)
        _, logits, _ = mlp_forward(self.params, Xn)
        return np.argmax(logits, axis=1).astype(np.int64)

    def predict_frames(self, frames: list[TelemetryFrame]) -> list[int]:
        """Класифікує список TelemetryFrame одним матричним проходом."""
        if not frames:
            return []
        X = extract_features_matrix(frames, self.window_size)
        return self.predict(X).tolist()


def _meta_path_for(path_npz: str) -> str:
    stem, _ = os.path.splitext(path_npz)
    return f"{stem}.json"


def save_model(
    path_npz: str,
    params: dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Зберігає ваги та параметри нормалізації у `.npz` і метадані у `.json`."""
    if not path_npz.endswith(".npz"):
        raise ValueError(f"Шлях моделі має закінчуватися на '.npz': {path_npz}")

    parent = os.path.dirname(path_npz)
    if parent:
        os.makedirs(parent, exist_ok=True)

    np.savez(
        path_npz,
        W1=params["W1"],
        b1=params["b1"],
        W2=params["W2"],
        b2=params["b2"],
        mean=mean,
        std=std,
    )
    meta_out = {
        "window_size": DEFAULT_WINDOW_SIZE,
        "feature_names": list(FEATURE_NAMES),
        "num_classes": NUM_CLASSES,
        "class_names": {str(k): v for k, v in CLASS_TO_KEY.items()},
        **metadata,
    }
    with open(_meta_path_for(path_npz), "w", encoding="utf-8") as fh:
        json.dump(meta_out, fh, ensure_ascii=False, indent=2)


def _resolve_model_path(path_npz: str) -> str:
    if os.path.exists(path_npz):
        return os.path.abspath(path_npz)
    if not os.path.isabs(path_npz):
        candidate = _REPO_ROOT / path_npz
        if candidate.exists():
            return str(candidate.resolve())
    raise FileNotFoundError(f"Файл моделі не знайдено: {path_npz}")


def load_model(path_npz: str = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Завантажує ваги `.npz` і метадані `.json` із валідацією та кешем."""
    resolved_npz = _resolve_model_path(path_npz)
    resolved_json = _meta_path_for(resolved_npz)
    if not os.path.exists(resolved_json):
        raise FileNotFoundError(f"Файл метаданих моделі не знайдено: {resolved_json}")

    cache_key = (
        resolved_npz,
        os.stat(resolved_npz).st_mtime_ns,
        os.stat(resolved_json).st_mtime_ns,
    )
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    with open(resolved_json, encoding="utf-8") as fh:
        metadata = json.load(fh)

    if list(metadata.get("feature_names", [])) != list(FEATURE_NAMES):
        raise ValueError(
            f"Невідповідні ознаки моделі: {metadata.get('feature_names')} != {list(FEATURE_NAMES)}"
        )
    window_size = int(metadata.get("window_size", 0))
    if window_size < 2:
        raise ValueError(f"Некоректний window_size у метаданих: {window_size}")

    required_arrays = ("W1", "b1", "W2", "b2", "mean", "std")
    with np.load(resolved_npz) as data:
        missing = [k for k in required_arrays if k not in data]
        if missing:
            raise ValueError(f"У {resolved_npz} відсутні масиви: {missing}")
        params = {k: data[k].astype(np.float64) for k in ("W1", "b1", "W2", "b2")}
        mean = data["mean"].astype(np.float64)
        std = data["std"].astype(np.float64)

    if params["W1"].shape[0] != len(FEATURE_NAMES) or params["W2"].shape[1] != NUM_CLASSES:
        raise ValueError("Некоректні розмірності матриць ваг у моделі")

    clf = MLPClassifier(params, mean, std, window_size=window_size, metadata=metadata)
    bundle = {
        "params": params,
        "mean": mean,
        "std": std,
        "metadata": metadata,
        "window_size": window_size,
        "classifier": clf,
    }
    _MODEL_CACHE[cache_key] = bundle
    return bundle


def classify_frames(
    frames: list[TelemetryFrame], model_path: str = DEFAULT_MODEL_PATH
) -> list[int]:
    """Повертає клас (0/1/2) для кожного кадру за допомогою навченого MLP."""
    if not frames:
        return []
    bundle = load_model(model_path)
    return bundle["classifier"].predict_frames(frames)


def compute_frame_values(
    frames: list[TelemetryFrame], classes: list[int], config: dict
) -> list[int]:
    """
    Обчислює цінність кадру за Розділом 3.5 FORMAT_SPEC_1.md і team/team_config.json:
    v_i = пріоритет(клас_i) + critical_бонус * [critical_i == 1].
    """
    if not isinstance(config, dict) or "пріоритети_кадрів" not in config:
        raise ValueError("У конфігурації відсутня секція 'пріоритети_кадрів'")
    prio_cfg = config["пріоритети_кадрів"]
    if not isinstance(prio_cfg, dict):
        raise ValueError("Секція 'пріоритети_кадрів' має бути словником")

    required_keys = (*CLASS_TO_KEY.values(), "critical_бонус")
    for key in required_keys:
        if key not in prio_cfg:
            raise ValueError(f"У 'пріоритети_кадрів' відсутній обов'язковий ключ: {key!r}")

    if len(frames) != len(classes):
        raise ValueError(
            f"Кількість кадрів ({len(frames)}) не дорівнює кількості класів ({len(classes)})"
        )

    crit_bonus = int(prio_cfg["critical_бонус"])
    values: list[int] = []
    for frame, cls in zip(frames, classes):
        if isinstance(cls, bool) or not isinstance(cls, (int, np.integer)):
            raise ValueError(f"Некоректний тип класу: {cls!r}")
        cls_int = int(cls)
        if cls_int not in CLASS_TO_KEY:
            raise ValueError(f"Невідомий клас кадру: {cls_int} (очікується 0, 1 або 2)")
        if frame.critical not in (0, 1):
            raise ValueError(f"Некоректне значення critical: {frame.critical}")

        base_prio = int(prio_cfg[CLASS_TO_KEY[cls_int]])
        values.append(base_prio + (crit_bonus if frame.critical == 1 else 0))
    return values
