#!/usr/bin/env python3
"""
experiments/run_feature_window_study.py

Великий, чесніший за "швидкий" експеримент для MLP-класифікатора кадрів (TLM1):
  - розмір вікна (одиночні та багатомасштабні),
  - набори ознак (віконні / сирі / сирі+min,max,розмах / біти),
  - ablation (викидаємо по одній ознаці),
  - вікно з кадрами "вперед" (потребує дозволу, див. нижче),
  - ваги класів у loss, розмір прихованого шару (1 шар) та глибина мережі (1 vs 2 приховані шари),
  - нормалізація входів: none / z-score / min-max / robust / robust з fallback / log+z-score,
  - функції активації прихованих шарів (relu, leaky, elu, silu, gelu, softplus,
    tanh, sigmoid, linear) та вплив ініціалізації (He / Xavier).

Протокол (щоб не було завищених чисел):
  1. Ознаки рахуються по всьому train.csv, потім БЛОКОВИЙ спліт 80/20 з розривом,
     щоб вікна train і val-частин не перекривались.
  2. Для кожної ініціалізації найкраща епоха обирається за macro-F1 на val-частині.
  3. Звітне число -- macro-F1 на valid.csv, який НЕ бере участі в жодному виборі.
  4. Рядки idx < (довжина минулого вікна) не беруться в навчання.

Потрібні лише numpy та стандартна бібліотека. Запуск з кореня репозиторію:
    python experiments/run_feature_window_study.py --data-dir team
    python experiments/run_feature_window_study.py --data-dir team --quick
    python experiments/run_feature_window_study.py --data-dir team --seeds 20 --groups windows,multi
    python experiments/run_feature_window_study.py --data-dir team --base-window 5

    python experiments/run_feature_window_study.py --data-dir team --groups norm --base-window 5
    python experiments/run_feature_window_study.py --data-dir team --norm robust   # нормалізація для всіх груп
    python experiments/run_feature_window_study.py --data-dir team --cw none --base-window 5   # без ваг класів

=== ПЕРЕВІРКА ДВОШАРОВОЇ МЕРЕЖІ (група depth або --hidden 32,16) ===
    # 1) швидка перевірка 1-шарових vs 2-шарових архітектур (3 ініціалізації):
    python experiments/run_feature_window_study.py --data-dir team --base-window 5 --cw none \
        --groups depth --quick --tag depth_quick
    # 2) основний прогін порівняння глибини (10 ініціалізацій):
    python experiments/run_feature_window_study.py --data-dir team --base-window 5 --cw none \
        --groups depth --tag depth
    # 3) прогін будь-якої іншої групи (напр. act чи lr) з фіксованою двошаровою мережею 32x16:
    python experiments/run_feature_window_study.py --data-dir team --base-window 5 --cw none \
        --hidden 32,16 --groups act --tag act_2layer

=== АКТИВАЦІЇ (рекомендований прогін; конфіг із дизайн-документа: w=5, без ваг, 32 нейрони) ===
    # 1) швидка перевірка, що все працює (~1-2 хв):
    python experiments/run_feature_window_study.py --data-dir team --base-window 5 --cw none \
        --groups act,act_init --quick --tag act_quick
    # 2) основний прогін (10 ініціалізацій):
    python experiments/run_feature_window_study.py --data-dir team --base-window 5 --cw none \
        --groups act,act_init --tag activations
    # 3) якщо різниці в межах шуму -- більше ініціалізацій:
    python experiments/run_feature_window_study.py --data-dir team --base-window 5 --cw none \
        --groups act,act_init --seeds 30 --tag activations_30
    # 4) за бажання: інший lr для всіх активацій
    python experiments/run_feature_window_study.py --data-dir team --base-window 5 --cw none \
        --groups act --lr 3e-3 --tag act_lr3e-3

  Групи:  act       -- кожна активація з "природною" ініціалізацією (He для сімейства ReLU, Xavier для tanh/sigmoid/linear);
          act_init  -- кожна активація з ПРОТИЛЕЖНОЮ ініціалізацією (відокремлює ефект активації від ефекту ініціалізації).
  Запускай act та act_init разом: порівняння "Δ vs relu" береться з групи act.

=== ПЕРЕВІРКА lr І ЗБІЖНОСТІ (група act_lr) ===
  act_lr: ReLU, SiLU, tanh, sigmoid x lr {3e-4, 1e-3, 3e-3, 1e-2}. Для кожної активації найкращий lr
  обирається за val-частиною train.csv (не за valid.csv), далі порівнюється з найкращим ReLU.
    python experiments/run_feature_window_study.py --data-dir team --base-window 5 --cw none \
        --groups act_lr --tag act_lr
  Скільки епох потрібно: у підсумку act/act_init/act_lr друкується середня найкраща епоха та частка запусків,
  що дійшли до ліміту --epochs без ранньої зупинки (якщо вона висока, збільш --epochs, напр. 300).

Групи (--groups): windows, multi, features, ablation, lookahead, weights, hidden, depth, norm, act, act_init, act_lr
Результати: results/feature_window_study{_TAG}.csv (кожен запуск), ..._summary.csv, ..._meta.json,
            results/activation_compare{_TAG}.csv (лише для груп act/act_init),
            results/depth_compare{_TAG}.csv (лише для групи depth)

ЗАХИСТ РЕЗУЛЬТАТІВ: якщо файл результатів уже існує, а --tag не задано, скрипт зупиниться
(щоб не затерти попередні прогони). Задай --tag або --overwrite.

УВАГА про lookahead: вікно з кадрами "вперед" чесне лише якщо приймач/викладач
подає holdout.csv суцільним рядом і дозволяє дивитися вперед. Спершу уточни це.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

EXPECTED_HEADER = [
    "timestamp_ms",
    "altitude_dm",
    "temp_c10",
    "vibration",
    "mode",
    "critical",
    "label",
]
HAND_NAMES = ["dalt", "dtmp", "absvib", "std_vib", "std_alt", "std_temp", "mfreq"]


# ----------------------------------------------------------------------------- дані
def load_csv(path):
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    if header != EXPECTED_HEADER:
        raise ValueError(
            f"{path}: очікував заголовок {EXPECTED_HEADER}, отримав {header}"
        )
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64)
    return {
        "ts": data[:, 0],
        "raw": data[:, 1:6].astype(
            np.float64
        ),  # altitude, temp, vibration, mode, critical
        "label": data[:, 6].astype(np.int64),
    }


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------------- ознаки
def scale_pf(size, mode):
    """(скільки кадрів ДО поточного, скільки ПІСЛЯ). 'c' -- causal, 'm' -- центроване."""
    if size < 2:
        raise ValueError("розмір вікна має бути >= 2")
    if mode == "c":
        return size - 1, 0
    return size // 2, size - 1 - size // 2


def window_tensor(raw, past, future):
    """(n, past+1+future, 5); поточний кадр на позиції `past`; краї доповнюються крайнім кадром."""
    n = len(raw)
    pad = np.vstack(
        [np.repeat(raw[:1], past, axis=0), raw, np.repeat(raw[-1:], future, axis=0)]
    )
    idx = np.arange(n)[:, None] + np.arange(past + future + 1)[None, :]
    return pad[idx]


def build_features(raw, cfg, base_w):
    """Повертає (X, імена, P, F), де P/F -- скільки кадрів минулого/майбутнього бачить вікно."""
    kind = cfg["kind"]
    scales = cfg["scales"] if kind == "hand" else [(base_w, "c")]
    pfs = [scale_pf(s, m) for s, m in scales]
    P = max(p for p, _ in pfs)
    F = max(f for _, f in pfs)
    W = window_tensor(raw, P, F)
    cur = P

    def sl(s, m):
        p, f = scale_pf(s, m)
        return slice(cur - p, cur + f + 1)

    b = sl(*scales[0])
    extras = {}
    if "range" in cfg["extras"]:
        for k, nm in ((0, "alt"), (1, "temp"), (2, "vib")):
            extras[f"range_{nm}"] = W[:, b, k].max(1) - W[:, b, k].min(1)
    if "minmax" in cfg["extras"]:
        for k, nm in ((0, "alt"), (1, "temp"), (2, "vib")):
            extras[f"min_{nm}"] = W[:, b, k].min(1)
            extras[f"max_{nm}"] = W[:, b, k].max(1)

    if kind == "hand":
        alt, tmp, vib, mod = W[:, :, 0], W[:, :, 1], W[:, :, 2], W[:, :, 3]
        cols = {
            "dalt": alt[:, cur] - alt[:, cur - 1],
            "dtmp": tmp[:, cur] - tmp[:, cur - 1],
            "absvib": np.abs(vib[:, cur]),
        }
        for s, m in scales:
            s_ = sl(s, m)
            tag = f"@{s}{m}"
            cols["std_vib" + tag] = vib[:, s_].std(1)
            cols["std_alt" + tag] = alt[:, s_].std(1)
            cols["std_temp" + tag] = tmp[:, s_].std(1)
            cols["mfreq" + tag] = (np.diff(mod[:, s_], axis=1) != 0).mean(1)
        cols = {k: v for k, v in cols.items() if k.split("@")[0] not in cfg["drop"]}
        cols.update(extras)
    elif kind == "raw":
        cols = {"raw": W[:, b, :].reshape(len(W), -1)}
        cols.update(extras)
    elif kind == "bits":
        parts = []
        for k, nb in ((0, 16), (1, 16), (2, 16), (3, 3), (4, 1)):
            v = W[:, b, k].astype(np.int64) & ((1 << nb) - 1)
            parts.append(
                ((v[:, :, None] >> np.arange(nb)[None, None, :]) & 1).reshape(
                    len(W), -1
                )
            )
        cols = {"bits": np.hstack(parts).astype(np.float64)}
    else:
        raise ValueError(f"невідомий kind: {kind}")
    if not cols:
        raise ValueError("після drop не лишилось ознак")
    blocks = [v if v.ndim == 2 else v[:, None] for v in cols.values()]
    return np.hstack(blocks), list(cols.keys()), P, F


# ----------------------------------------------------------------------------- метрика
def macro_f1(y, p):
    f1s = []
    for c in range(3):
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(
            2 * pr * rc / (pr + rc) if pr + rc else 0.0
        )  # P+R=0 -> F1=0, без ділення на нуль
    return float(np.mean(f1s)), f1s


# ----------------------------------------------------------------------------- функції активації
# Кожна активація: (f(z), df(z, h), "природна" ініціалізація), де h = f(z).
# df повертає похідну f за z; для деяких активацій зручніше брати її через h (tanh, sigmoid, elu).
def _sigmoid(z):
    # стабільна сигмоїда без переповнення exp: sigma(z) = 0.5 * (1 + tanh(z/2))
    return 0.5 * (1.0 + np.tanh(0.5 * z))


_GELU_C = float(np.sqrt(2.0 / np.pi))
_GELU_A = 0.044715


def _relu(z):
    return np.maximum(0.0, z)


def _relu_d(z, h):
    return (z > 0).astype(z.dtype)


def _leaky(z):
    return np.where(z > 0, z, 0.01 * z)


def _leaky_d(z, h):
    return np.where(z > 0, 1.0, 0.01)


def _elu(z):
    return np.where(z > 0, z, np.expm1(np.minimum(z, 0.0)))  # alpha = 1


def _elu_d(z, h):
    return np.where(z > 0, 1.0, h + 1.0)  # для z<=0: exp(z) = h + 1


def _silu(z):
    return z * _sigmoid(z)


def _silu_d(z, h):
    s = _sigmoid(z)
    return s * (1.0 + z * (1.0 - s))


def _gelu(z):
    return 0.5 * z * (1.0 + np.tanh(_GELU_C * (z + _GELU_A * z**3)))


def _gelu_d(z, h):
    t = np.tanh(_GELU_C * (z + _GELU_A * z**3))
    return 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t * t) * _GELU_C * (
        1.0 + 3.0 * _GELU_A * z * z
    )


def _softplus(z):
    return np.logaddexp(0.0, z)  # ln(1 + e^z), стабільно


def _softplus_d(z, h):
    return _sigmoid(z)


def _tanh_d(z, h):
    return 1.0 - h * h


def _sigmoid_d(z, h):
    return h * (1.0 - h)


def _linear(z):
    return z


def _linear_d(z, h):
    return np.ones_like(z)


ACTIVATIONS = {
    "relu": (_relu, _relu_d, "he"),  # еталон (поточний вибір дизайн-документа)
    "leaky": (_leaky, _leaky_d, "he"),  # alpha = 0.01
    "elu": (_elu, _elu_d, "he"),  # alpha = 1
    "silu": (_silu, _silu_d, "he"),  # swish
    "gelu": (_gelu, _gelu_d, "he"),  # наближення через tanh
    "softplus": (_softplus, _softplus_d, "he"),
    "tanh": (np.tanh, _tanh_d, "xavier"),
    "sigmoid": (_sigmoid, _sigmoid_d, "xavier"),
    "linear": (_linear, _linear_d, "xavier"),  # контроль: мережа без нелінійності
}
ACT_ORDER = list(ACTIVATIONS)
INITS = ("auto", "he", "xavier")


def resolve_init(act, init):
    if init == "auto":
        return ACTIVATIONS[act][2]
    if init not in ("he", "xavier"):
        raise ValueError(f"невідома ініціалізація: {init}")
    return init


def parse_hidden(hidden):
    """Нормалізує опис прихованих шарів (int, tuple/list або рядок '32,16' / '32x16') у кортеж int."""
    if isinstance(hidden, int):
        sizes = (hidden,)
    elif isinstance(hidden, str):
        parts = [p.strip() for p in hidden.replace("x", ",").replace("X", ",").split(",") if p.strip()]
        sizes = tuple(int(p) for p in parts)
    elif isinstance(hidden, (tuple, list)):
        sizes = tuple(int(x) for x in hidden)
    else:
        raise ValueError(f"некоректний опис прихованих шарів: {hidden!r}")
    if not sizes or any(s < 1 for s in sizes):
        raise ValueError(f"розміри прихованих шарів мають бути >= 1, отримано {sizes}")
    return sizes


def format_hidden(hidden):
    sizes = parse_hidden(hidden)
    return str(sizes[0]) if len(sizes) == 1 else "x".join(map(str, sizes))


def count_params(params):
    return int(sum(v.size for v in params.values()))


# ----------------------------------------------------------------------------- MLP (numpy, 1 або кілька прихованих шарів)
def init_params(d, hidden, seed, init="he"):
    """He: N(0, sqrt(2/n_in)); Xavier (LeCun-варіант, як у дизайні): N(0, sqrt(1/n_in)). Зсуви нулі.
    Підтримує як один прихований шар (hidden=32), так і кілька (hidden=(32, 16))."""
    if init not in ("he", "xavier"):
        raise ValueError(f"невідома ініціалізація: {init}")
    h_sizes = parse_hidden(hidden)
    dims = [d, *h_sizes, 3]
    k = 2.0 if init == "he" else 1.0
    rng = np.random.default_rng(seed)
    params = {}
    for idx in range(len(dims) - 1):
        n_in, n_out = dims[idx], dims[idx + 1]
        params[f"W{idx + 1}"] = rng.normal(0, np.sqrt(k / n_in), (n_in, n_out))
        params[f"b{idx + 1}"] = np.zeros(n_out)
    return params


def softmax(z):
    z = z - z.max(1, keepdims=True)  # стійкість: віднімаємо max
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def forward(params, X, act="relu"):
    f = ACTIVATIONS[act][0]
    n_layers = len(params) // 2
    a = X
    hiddens = []
    for layer in range(1, n_layers):
        a = f(a @ params[f"W{layer}"] + params[f"b{layer}"])
        hiddens.append(a)
    logits = a @ params[f"W{n_layers}"] + params[f"b{n_layers}"]
    h_all = hiddens[0] if len(hiddens) == 1 else np.hstack(hiddens)
    return h_all, logits


def loss_and_grads(params, X, y, cw, act="relu"):
    f, df, _ = ACTIVATIONS[act]
    n = len(y)
    n_layers = len(params) // 2
    acts = [X]
    zs = [None]
    for layer in range(1, n_layers):
        z_l = acts[-1] @ params[f"W{layer}"] + params[f"b{layer}"]
        h_l = f(z_l)
        zs.append(z_l)
        acts.append(h_l)
    z_out = acts[-1] @ params[f"W{n_layers}"] + params[f"b{n_layers}"]
    p = softmax(z_out)
    w = cw[y]
    loss = float(-(w * np.log(p[np.arange(n), y] + 1e-12)).sum() / n)
    g = p.copy()
    g[np.arange(n), y] -= 1.0
    g *= (w / n)[:, None]
    grads = {}
    for layer in range(n_layers, 0, -1):
        a_prev = acts[layer - 1]
        grads[f"W{layer}"] = a_prev.T @ g
        grads[f"b{layer}"] = g.sum(0)
        if layer > 1:
            g = (g @ params[f"W{layer}"].T) * df(zs[layer - 1], acts[layer - 1])
    return loss, grads


def class_weights(y, mode):
    full = len(y) / (3.0 * np.maximum(np.bincount(y, minlength=3), 1))
    if mode == "none":
        return np.ones(3)
    if mode == "sqrt":
        return np.sqrt(full)
    if mode == "full":
        return full
    raise ValueError(f"невідомий режим ваг: {mode}")


NORMS = ("none", "zscore", "minmax", "robust", "robust_fb", "logz")


def fit_normalizer(X, mode):
    """Параметри нормалізації рахуються ЛИШЕ на X (train-частина). Повертає функцію apply(A).
    robust: (x - медіана) / IQR; якщо IQR=0 (ознака майже завжди 0), scale=1, як у sklearn.
    robust_fb: те саме, але при IQR=0 ділимо на std.
    logz: sign(x)*log1p(|x|), потім z-score."""

    def pre(A):
        return np.sign(A) * np.log1p(np.abs(A)) if mode == "logz" else A

    Z = pre(X)
    if mode == "none":
        shift, scale = np.zeros(X.shape[1]), np.ones(X.shape[1])
    elif mode in ("zscore", "logz"):
        shift, scale = Z.mean(0), Z.std(0) + 1e-8
    elif mode == "minmax":
        shift, scale = Z.min(0), np.maximum(Z.max(0) - Z.min(0), 1e-8)
    elif mode in ("robust", "robust_fb"):
        q1, med, q3 = np.percentile(Z, [25, 50, 75], axis=0)
        iqr = q3 - q1
        fallback = 1.0 if mode == "robust" else Z.std(0) + 1e-8
        shift, scale = med, np.where(iqr > 1e-12, iqr, fallback)
    else:
        raise ValueError(f"невідома нормалізація: {mode}")
    return lambda A: (pre(A) - shift) / scale


def fit_eval(
    Xtr,
    ytr,
    Xva,
    yva,
    Xext,
    yext,
    seed,
    hidden,
    cw_mode,
    epochs,
    patience,
    norm="zscore",
    bs=64,
    lr=1e-3,
    act="relu",
    init="auto",
):
    apply_norm = fit_normalizer(
        Xtr, norm
    )  # параметри нормалізації -- лише з train-частини
    Xt, Xv, Xe = apply_norm(Xtr), apply_norm(Xva), apply_norm(Xext)
    params = init_params(Xt.shape[1], hidden, seed, resolve_init(act, init))
    n_params = count_params(params)
    cw = class_weights(ytr, cw_mode)
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v2 = {k: np.zeros_like(v) for k, v in params.items()}
    rng = np.random.default_rng(10_000 + seed)
    t, best_f, best_params, best_ep, bad = 0, -1.0, None, 0, 0
    for ep in range(epochs):
        perm = rng.permutation(len(Xt))
        for i in range(0, len(Xt), bs):
            ix = perm[i : i + bs]
            _, grads = loss_and_grads(params, Xt[ix], ytr[ix], cw, act)
            t += 1
            for k in params:
                m[k] = 0.9 * m[k] + 0.1 * grads[k]
                v2[k] = 0.999 * v2[k] + 0.001 * grads[k] ** 2
                params[k] -= (
                    lr
                    * (m[k] / (1 - 0.9**t))
                    / (np.sqrt(v2[k] / (1 - 0.999**t)) + 1e-8)
                )
        f, _ = macro_f1(yva, np.argmax(forward(params, Xv, act)[1], 1))
        if f > best_f:
            best_f, best_ep, bad = f, ep, 0
            best_params = {k: a.copy() for k, a in params.items()}
        else:
            bad += 1
            if bad >= patience:
                break
    pe = np.argmax(forward(best_params, Xe, act)[1], 1)
    ext, per = macro_f1(yext, pe)
    h_tr, z_tr = forward(best_params, Xt, act)
    train_f1, _ = macro_f1(ytr, np.argmax(z_tr, 1))
    return {
        "val_f1": best_f,
        "ext_f1": ext,
        "ext_per_class": per,
        "train_f1": train_f1,  # macro-F1 на train-частині (різниця з val_f1 -- ознака перенавчання)
        "const_units": float(
            (h_tr.std(0) < 1e-6).mean()
        ),  # частка "мертвих" (константних) нейронів по всіх прихованих шарах
        "n_params": n_params,
        "best_epoch": best_ep,
        "epochs_run": ep + 1,
        "max_abs_input": float(np.abs(Xt).max()),
    }


# ----------------------------------------------------------------------------- самоперевірки
def _grad_check(act, seed=0, hidden=7):
    """Найгірша відносна похибка між аналітичним і чисельним градієнтом (float64) для активації act."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(12, 5))
    y = rng.integers(0, 3, 12)
    cw = np.array([0.5, 2.0, 3.0])
    params = init_params(5, hidden, 1, ACTIVATIONS[act][2])
    for k, arr in params.items():
        if k.startswith("b"):
            params[k] = rng.normal(size=arr.shape) * 0.1
    _, ana = loss_and_grads(params, X, y, cw, act)
    eps, worst = 1e-6, 0.0
    for k, arr in params.items():
        for idx in np.ndindex(arr.shape):
            old = arr[idx]
            arr[idx] = old + eps
            lp, _ = loss_and_grads(params, X, y, cw, act)
            arr[idx] = old - eps
            lm, _ = loss_and_grads(params, X, y, cw, act)
            arr[idx] = old
            num = (lp - lm) / (2 * eps)
            worst = max(
                worst,
                abs(num - ana[k][idx]) / max(1e-8, abs(num) + abs(ana[k][idx])),
            )
    return worst


def run_selftests(train, valid):
    results = []

    def check(name, fn):
        try:
            fn()
            results.append((name, True, ""))
        except AssertionError as e:
            results.append((name, False, str(e)))

    def t_data():
        for nm, d in (("train", train), ("valid", valid)):
            assert (np.diff(d["ts"]) >= 0).all(), f"{nm}: timestamp_ms спадає"
            assert set(np.unique(d["label"])) <= {0, 1, 2}, f"{nm}: невідомі мітки"
            assert np.isfinite(d["raw"]).all(), f"{nm}: NaN/inf"
            r = d["raw"]
            assert ((r[:, 0] >= -32768) & (r[:, 0] <= 32767)).all(), (
                f"{nm}: altitude поза i16"
            )
            assert ((r[:, 3] >= 0) & (r[:, 3] <= 7)).all(), f"{nm}: mode поза 0..7"
            assert set(np.unique(r[:, 4])) <= {0.0, 1.0}, f"{nm}: critical не 0/1"

    def t_features_known():
        raw = np.array(
            [
                [100, 10, 0, 1, 0],
                [101, 10, 0, 1, 0],
                [103, 12, 4, 2, 0],
                [103, 12, -4, 2, 1],
            ],
            float,
        )
        cfg = {"kind": "hand", "scales": [(4, "c")], "extras": (), "drop": ()}
        X, names, P, F = build_features(raw, cfg, 8)
        assert (P, F) == (3, 0) and names[:3] == ["dalt", "dtmp", "absvib"]
        assert np.allclose(X[0], [0, 0, 0, 0, 0, 0, 0]), (
            f"перший кадр має дати нулі, а дав {X[0]}"
        )
        last = dict(zip(names, X[3]))
        assert last["dalt"] == 0 and last["dtmp"] == 0 and last["absvib"] == 4
        assert np.isclose(last["std_vib@4c"], np.std([0, 0, 4, -4])), "std вібрації"
        assert np.isclose(last["std_alt@4c"], np.std([100, 101, 103, 103])), (
            "std висоти"
        )
        assert np.isclose(last["mfreq@4c"], 1 / 3), (
            "частота зміни mode: [1,1,2,2] -> 1/3"
        )

    def t_causal():
        rng = np.random.default_rng(1)
        raw = rng.normal(size=(60, 5)) * 10
        raw[:, 3] = rng.integers(0, 8, 60)
        i = 30
        for sc, must_equal in (([(8, "c")], True), ([(8, "m")], False)):
            cfg = {"kind": "hand", "scales": sc, "extras": (), "drop": ()}
            A, *_ = build_features(raw, cfg, 8)
            raw2 = raw.copy()
            raw2[i + 1 :] += 999.0  # псуємо лише майбутнє
            B, *_ = build_features(raw2, cfg, 8)
            same = np.allclose(A[i], B[i])
            assert same == must_equal, (
                f"{sc}: causal-властивість порушена (same={same})"
            )

    def t_split_disjoint():
        for sc in ([(8, "c")], [(4, "c"), (32, "c")], [(16, "m")]):
            P = max(scale_pf(s, m)[0] for s, m in sc)
            F = max(scale_pf(s, m)[1] for s, m in sc)
            n, n_tr = 8000, 6400
            tr_idx = np.arange(P, n_tr - (P + F))
            va_idx = np.arange(n_tr, n)
            assert max(tr_idx) + F < min(va_idx) - P, (
                f"вікна train і val перекриваються для {sc}"
            )
            assert not set(tr_idx) & set(va_idx), "train ∩ val ≠ ∅"

    def t_f1():
        y = np.array([0, 0, 1, 1, 2, 2])
        m, per = macro_f1(y, np.zeros(6, int))
        assert np.isfinite(m) and per[1] == 0.0 and per[2] == 0.0, (
            "P+R=0 має давати F1=0"
        )
        assert np.isclose(per[0], 0.5) and np.isclose(m, 0.5 / 3)
        assert macro_f1(y, y)[0] == 1.0

    def t_softmax():
        p = softmax(np.array([[1000.0, 0.0, -1000.0]]))
        assert np.isfinite(p).all() and np.isclose(p.sum(), 1.0)

    def t_weights():
        y = np.array([0] * 70 + [1] * 10 + [2] * 20)
        w = class_weights(y, "full")
        assert np.isclose((np.bincount(y) * w).sum(), len(y)), (
            "зважені лічильники мають давати N"
        )
        assert np.allclose(class_weights(y, "none"), 1.0)

    def t_norm():
        rng = np.random.default_rng(2)
        X = np.c_[rng.normal(5, 3, 500), rng.exponential(2, 500), np.zeros(500)]
        X[:50, 2] = 1.0  # третя ознака: 90% нулів -> IQR=0
        for mode in NORMS:
            Z = fit_normalizer(X, mode)(X)
            assert np.isfinite(Z).all(), f"{mode}: NaN/inf (ділення на нуль?)"
        Z = fit_normalizer(X, "zscore")(X)
        assert np.allclose(Z.mean(0)[:2], 0, atol=1e-9) and np.allclose(
            Z.std(0)[:2], 1, atol=1e-6
        ), "z-score"
        Z = fit_normalizer(X, "minmax")(X)
        assert np.allclose(Z.min(0), 0) and np.allclose(Z.max(0), 1), (
            "min-max має давати [0,1] на train"
        )
        Z = fit_normalizer(X, "robust")(X)
        assert np.allclose(np.median(Z[:, :2], axis=0), 0, atol=1e-9), (
            "robust: медіана 0"
        )
        q1, q3 = np.percentile(Z[:, :2], [25, 75], axis=0)
        assert np.allclose(q3 - q1, 1, atol=1e-9), "robust: IQR 1"
        f = fit_normalizer(X[:400], "zscore")
        assert not np.allclose(f(X[400:]).mean(0), 0, atol=1e-3), (
            "параметри мають братися з train, не з нових даних"
        )

    def t_gradient():
        worst = _grad_check("relu")
        assert worst < 1e-5, (
            f"градієнт не збігається з чисельним: відн. похибка {worst:.2e}"
        )

    def t_act_derivatives():
        # (1) df збігається з чисельною похідною f (поза зламом ReLU/Leaky у нулі)
        rng = np.random.default_rng(4)
        z = rng.normal(size=400) * 3.0
        z = z[np.abs(z) > 1e-3]
        eps = 1e-6
        for name, (f, df, init) in ACTIVATIONS.items():
            num = (f(z + eps) - f(z - eps)) / (2 * eps)
            ana = df(z, f(z))
            err = float(np.max(np.abs(num - ana)))
            assert err < 1e-6, f"{name}: похідна не збігається з чисельною ({err:.2e})"
            assert init in ("he", "xavier"), f"{name}: невідома ініціалізація"
        # (2) немає переповнень/NaN на великих |z|
        big = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])
        for name, (f, df, _) in ACTIVATIONS.items():
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    h = f(big)
                    d = df(big, h)
            except FloatingPointError as e:
                raise AssertionError(f"{name}: переповнення на великих |z| ({e})")
            assert np.isfinite(h).all() and np.isfinite(d).all(), (
                f"{name}: NaN/inf на великих |z|"
            )

    def t_act_gradients():
        # повна чисельна перевірка backprop для КОЖНОЇ активації
        for name in ACT_ORDER:
            worst = _grad_check(name)
            assert worst < 1e-5, (
                f"{name}: градієнт не збігається з чисельним (відн. похибка {worst:.2e})"
            )

    def t_two_layer_mlp():
        # перевірка градієнта і спадання loss для двошарової мережі (hidden=(6, 4) та (16, 8))
        for act in ("relu", "silu", "tanh"):
            worst = _grad_check(act, seed=7, hidden=(6, 4))
            assert worst < 1e-5, (
                f"2-шарова ({act}): градієнт не збігається з чисельним ({worst:.2e})"
            )
        cfg = {"kind": "hand", "scales": [(5, "c")], "extras": (), "drop": ()}
        X, _, P, _ = build_features(train["raw"], cfg, 5)
        sl = slice(P, 2000)
        Xt, yt = X[sl], train["label"][sl]
        Xn = (Xt - Xt.mean(0)) / (Xt.std(0) + 1e-8)
        cw = class_weights(yt, "none")
        params = init_params(Xn.shape[1], (16, 8), 3, "he")
        assert set(params.keys()) == {"W1", "b1", "W2", "b2", "W3", "b3"}
        l0, _ = loss_and_grads(params, Xn, yt, cw, "relu")
        for _ in range(60):
            _, g = loss_and_grads(params, Xn, yt, cw, "relu")
            for k in params:
                params[k] -= 0.05 * g[k]
        l1, _ = loss_and_grads(params, Xn, yt, cw, "relu")
        assert l1 < l0, f"2-шарова: loss не спадає: {l0:.4f} -> {l1:.4f}"

    def t_init_scale():
        for init, k in (("he", 2.0), ("xavier", 1.0)):
            p = init_params(400, 600, 0, init)
            assert abs(p["W1"].std() / np.sqrt(k / 400) - 1) < 0.05, (
                f"{init}: неправильний масштаб W1"
            )
            assert abs(p["W2"].std() / np.sqrt(k / 600) - 1) < 0.05, (
                f"{init}: неправильний масштаб W2"
            )
        assert resolve_init("relu", "auto") == "he"
        assert resolve_init("tanh", "auto") == "xavier"
        assert resolve_init("tanh", "he") == "he"

    def t_loss_down_and_determinism():
        cfg = {"kind": "hand", "scales": [(8, "c")], "extras": (), "drop": ()}
        X, _, P, F = build_features(train["raw"], cfg, 8)
        Xe, *_ = build_features(valid["raw"], cfg, 8)
        sl = slice(P, 2000)
        Xt, yt = X[sl], train["label"][sl]
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-8
        Xn = (Xt - mu) / sd
        cw = class_weights(yt, "full")
        for act in ("relu", "tanh"):
            params = init_params(Xn.shape[1], 16, 3, ACTIVATIONS[act][2])
            l0, _ = loss_and_grads(params, Xn, yt, cw, act)
            for _ in range(60):
                _, g = loss_and_grads(params, Xn, yt, cw, act)
                for k in params:
                    params[k] -= 0.05 * g[k]
            l1, _ = loss_and_grads(params, Xn, yt, cw, act)
            assert l1 < l0, f"{act}: loss не спадає: {l0:.4f} -> {l1:.4f}"
        for act in ("relu", "silu"):
            kw = dict(act=act)
            a = fit_eval(
                X[sl],
                yt,
                X[2000:2600],
                train["label"][2000:2600],
                Xe,
                valid["label"],
                5,
                16,
                "full",
                3,
                3,
                **kw,
            )
            b = fit_eval(
                X[sl],
                yt,
                X[2000:2600],
                train["label"][2000:2600],
                Xe,
                valid["label"],
                5,
                16,
                "full",
                3,
                3,
                **kw,
            )
            assert a["ext_f1"] == b["ext_f1"], (
                f"{act}: один і той самий seed дає різні результати"
            )

    check("дані: час не спадає, мітки 0..2, діапазони полів", t_data)
    check("ознаки: відомі значення, перший кадр дає нулі", t_features_known)
    check(
        "ознаки: causal-вікно не залежить від майбутнього (centered -- залежить)",
        t_causal,
    )
    check(
        "спліт: вікна train і val не перекриваються, train ∩ val = ∅", t_split_disjoint
    )
    check("macro-F1: P+R=0 -> F1=0, ідеал=1", t_f1)
    check("softmax стійкий до великих логітів", t_softmax)
    check("ваги класів: зважені лічильники дають N", t_weights)
    check("нормалізації: z-score/min-max/robust коректні, IQR=0 не дає NaN", t_norm)
    check("backprop (ReLU): чисельна перевірка градієнта (float64)", t_gradient)
    check(
        "активації: похідні = чисельні, без переповнень на великих |z|",
        t_act_derivatives,
    )
    check(f"backprop для КОЖНОЇ активації ({', '.join(ACT_ORDER)})", t_act_gradients)
    check("двошарова мережа: градієнт (relu/silu/tanh) і спадання loss", t_two_layer_mlp)
    check("ініціалізації He/Xavier: правильний масштаб, auto-вибір", t_init_scale)
    check(
        "loss спадає; фіксований seed -> однаковий результат (relu, tanh/silu)",
        t_loss_down_and_determinism,
    )
    return results


# ----------------------------------------------------------------------------- конфіги
ACT_GROUPS = ("act", "act_init")
DEPTH_GROUP = "depth"
DEPTH_ARCHS = ((32,), (64,), (16, 16), (32, 16), (32, 32), (64, 32))

LR_GROUP = "act_lr"
LR_GRID = (3e-4, 1e-3, 3e-3, 1e-2)
LR_ACTS = ("relu", "silu", "tanh", "sigmoid")


def make_configs(
    groups,
    bw,
    norm="zscore",
    cw_default="full",
    act_default="relu",
    init_default="auto",
    lr_default=1e-3,
    hidden_default=(32,),
):
    C = []
    h_def = parse_hidden(hidden_default)

    def add(
        group,
        name,
        kind="hand",
        scales=None,
        extras=(),
        drop=(),
        hidden=None,
        cw=None,
        nm=None,
        act=None,
        init=None,
        lr=None,
    ):
        a = act or act_default
        i = init or init_default
        lr_ = lr or lr_default
        h_ = parse_hidden(hidden) if hidden is not None else h_def
        # якщо глобально змінено архітектуру/активацію/ініціалізацію/lr, позначаємо це в назві конфіга,
        # щоб не сплутати з прогонами за замовчуванням
        if group not in ("hidden", DEPTH_GROUP) and h_ != (32,):
            name = f"{name} [h={format_hidden(h_)}]"
        if (
            group not in ACT_GROUPS
            and group != LR_GROUP
            and (a != "relu" or i != "auto")
        ):
            name = f"{name} [act={a} init={i}]"
        if group not in ACT_GROUPS and group != LR_GROUP and lr_ != 1e-3:
            name = f"{name} [lr={lr_:g}]"
        C.append(
            {
                "group": group,
                "name": name,
                "kind": kind,
                "scales": scales or [(bw, "c")],
                "extras": tuple(extras),
                "drop": tuple(drop),
                "hidden": h_,
                "cw": cw or cw_default,
                "norm": nm or norm,
                "act": a,
                "init": i,
                "lr": lr_,
            }
        )

    if "windows" in groups:
        for s in (2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48):
            add("windows", f"hand causal w={s}", scales=[(s, "c")])
    if "multi" in groups:
        for sc in ((4, 8, 16), (8, 32), (4, 8, 16, 32), (3, 6, 12, 24)):
            add(
                "multi",
                "hand multi " + "+".join(map(str, sc)),
                scales=[(s, "c") for s in sc],
            )
    if "features" in groups:
        add("features", f"hand w={bw}")
        add("features", f"hand w={bw} + range", extras=("range",))
        add("features", f"hand w={bw} + min,max,range", extras=("minmax", "range"))
        add("features", f"raw 5x{bw}", kind="raw")
        add(
            "features",
            f"raw 5x{bw} + min,max,range",
            kind="raw",
            extras=("minmax", "range"),
        )
        add("features", f"BITS of window w={bw}", kind="bits")
    if "ablation" in groups:
        for d in HAND_NAMES:
            add("ablation", f"hand w={bw} minus {d}", drop=(d,))
    if "lookahead" in groups:
        for s in (4, 8, 16):
            add(
                "lookahead",
                f"hand CENTERED w={s} (дивиться вперед!)",
                scales=[(s, "m")],
            )
    if "weights" in groups:
        for cw in ("none", "sqrt", "full"):
            add("weights", f"hand w={bw} class-weights={cw}", cw=cw)
    if "hidden" in groups:
        for h in (8, 16, 32, 64):
            add("hidden", f"hand w={bw} hidden={h}", hidden=(h,))
    if DEPTH_GROUP in groups:
        for h in DEPTH_ARCHS:
            lbl = format_hidden(h)
            tag = "1-шар" if len(h) == 1 else "2-шари"
            add(DEPTH_GROUP, f"hand w={bw} arch={lbl} ({tag})", hidden=h)
    if "norm" in groups:
        for nm in NORMS:
            add("norm", f"hand w={bw} norm={nm}", nm=nm)
    if "act" in groups:
        # кожна активація зі своєю "природною" ініціалізацією
        for a in ACT_ORDER:
            add("act", f"hand w={bw} act={a}", act=a, init="auto")
    if "act_init" in groups:
        # та сама активація, але з ПРОТИЛЕЖНОЮ ініціалізацією: чи вирішує ініціалізація, а не активація?
        for a in ACT_ORDER:
            other = "xavier" if ACTIVATIONS[a][2] == "he" else "he"
            add("act_init", f"hand w={bw} act={a} init={other}", act=a, init=other)
    if LR_GROUP in groups:
        for a in LR_ACTS:
            for lr in LR_GRID:
                add(
                    LR_GROUP,
                    f"hand w={bw} act={a} lr={lr:g}",
                    act=a,
                    init="auto",
                    lr=lr,
                )
    return C


def print_lr_table(summary_rows, ext_by_name, path):
    """Таблиця активація x lr. Найкращий lr для кожної активації обирається за val-частиною (val_mean),
    а не за valid.csv; потім порівнюємо ext macro-F1 цього вибору з вибраним для ReLU."""
    rows = [r for r in summary_rows if r["group"] == LR_GROUP]
    if not rows:
        return
    acts = list(dict.fromkeys(r["act"] for r in rows))
    lrs = sorted({r["lr"] for r in rows})
    picked = {}
    for a in acts:
        ra = [r for r in rows if r["act"] == a]
        picked[a] = max(ra, key=lambda r: r["val_mean"])
    print(
        "\n== lr x АКТИВАЦІЯ: macro-F1 на valid.csv; * = lr, обраний за val-частиною =="
    )
    print("   " + " " * 9 + "".join(f"{('lr=' + format(lr, 'g')):>16s}" for lr in lrs))
    for a in acts:
        line = f"   {a:9s}"
        for lr in lrs:
            r = next(x for x in rows if x["act"] == a and x["lr"] == lr)
            mark = "*" if r is picked[a] else " "
            line += f"  {r['ext_mean']:.3f}±{r['ext_std']:.3f}{mark}"
        print(line)
    print(
        "\n   епоха найкращої ваги / частка запусків, що дійшли до ліміту епох (без ранньої зупинки):"
    )
    for a in acts:
        line = f"   {a:9s}"
        for lr in lrs:
            r = next(x for x in rows if x["act"] == a and x["lr"] == lr)
            line += f"  {r['best_epoch_mean']:5.0f} / {r['capped_frac']:4.0%} "
        print(line)
    base = "relu"
    out = []
    print("\n   Δ відносно ReLU (обидва з lr, обраним за val-частиною):")
    for a in acts:
        x = ext_by_name[picked[a]["name"]]
        b = ext_by_name[picked[base]["name"]]
        delta = float(x.mean() - b.mean())
        if len(x) >= 5 and len(b) >= 5:
            se = float(np.sqrt(x.var(ddof=1) / len(x) + b.var(ddof=1) / len(b)))
            z = delta / se if se > 0 else float("nan")
            verdict = (
                "в межах шуму"
                if not np.isfinite(z) or abs(z) < 2
                else ("імовірно " if abs(z) < 3 else "значуще ")
                + ("краще" if delta > 0 else "гірше")
            )
        else:
            se = z = float("nan")
            verdict = "замало --seeds (<5) для висновку"
        zs = f"{z:6.1f}" if np.isfinite(z) else "   n/a"
        print(
            f"   {a:9s} lr={picked[a]['lr']:g}  ext={float(x.mean()):.3f}  Δ={delta:+.4f}  Δ/SE={zs}  {verdict}"
        )
        out.append(
            {
                "act": a,
                "lr_picked_by_val": picked[a]["lr"],
                "ext_mean": round(float(x.mean()), 5),
                "ext_std": round(float(x.std()), 5),
                "delta_vs_relu": round(delta, 5),
                "z": round(z, 2) if np.isfinite(z) else "",
                "verdict": verdict,
            }
        )
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        wr.writeheader()
        wr.writerows(out)
    print(
        "   Якщо найкращий lr лежить на краю сітки, розшир її; якщо багато запусків дійшли до ліміту епох, збільш --epochs."
    )


def depth_compare(summary_rows, ext_by_name, base_name, path):
    """Порівняння 1-шарових і 2-шарових архітектур відносно базового 1-шарового еталона (arch=32)."""
    if base_name not in ext_by_name:
        return
    base = ext_by_name[base_name]
    rows = []
    for r in summary_rows:
        if r["group"] != DEPTH_GROUP:
            continue
        x = ext_by_name[r["name"]]
        delta = float(x.mean() - base.mean())
        if len(x) > 1 and len(base) > 1:
            se = float(np.sqrt(x.var(ddof=1) / len(x) + base.var(ddof=1) / len(base)))
            z = delta / se if se > 0 else float("nan")
        else:
            se = z = float("nan")
        if len(x) < 5 or len(base) < 5:
            verdict = "замало --seeds (<5) для висновку"
        elif not np.isfinite(z) or abs(z) < 2:
            verdict = "в межах шуму"
        elif abs(z) < 3:
            verdict = "імовірно " + ("краще" if delta > 0 else "гірше")
        else:
            verdict = "значуще " + ("краще" if delta > 0 else "гірше")
        rows.append(
            {
                "name": r["name"],
                "arch": r["arch"],
                "n_params": r["n_params"],
                "ext_mean": round(float(x.mean()), 5),
                "ext_std": round(float(x.std()), 5),
                "delta_vs_32": round(delta, 5),
                "se": round(se, 5) if np.isfinite(se) else "",
                "z": round(z, 2) if np.isfinite(z) else "",
                "verdict": verdict,
            }
        )
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n== ГЛИБИНА МЕРЕЖІ: Δ macro-F1 відносно '{base_name}' ==")
    print(f"   {'конфіг':46s} {'params':>6s} {'mean':>6s}  {'Δ':>7s}  {'Δ/SE':>6s}  висновок")
    for r in rows:
        zs = f"{r['z']:6.1f}" if r["z"] != "" else "   n/a"
        print(
            f"   {r['name']:46s} {r['n_params']:6d} {r['ext_mean']:.3f}  {r['delta_vs_32']:+.4f}  {zs}  {r['verdict']}"
        )


def activation_compare(summary_rows, ext_by_name, base_name, path):
    """Порівняння кожної активації з ReLU: Δ середнього macro-F1 і оцінка Δ/SE (лише розкид між ініціалізалізаціями)."""
    if base_name not in ext_by_name:
        print(
            f"\n(порівняння з еталоном пропущено: у цьому прогоні немає '{base_name}'; "
            "запускай групу act разом з act_init)"
        )
        return
    base = ext_by_name[base_name]
    rows = []
    for r in summary_rows:
        if r["group"] not in ACT_GROUPS:
            continue
        x = ext_by_name[r["name"]]
        delta = float(x.mean() - base.mean())
        if len(x) > 1 and len(base) > 1:
            se = float(np.sqrt(x.var(ddof=1) / len(x) + base.var(ddof=1) / len(base)))
            z = delta / se if se > 0 else float("nan")
        else:
            se = z = float("nan")
        if len(x) < 5 or len(base) < 5:
            verdict = "замало --seeds (<5) для висновку"
        elif not np.isfinite(z) or abs(z) < 2:
            verdict = "в межах шуму"
        elif abs(z) < 3:
            verdict = "імовірно " + ("краще" if delta > 0 else "гірше")
        else:
            verdict = "значуще " + ("краще" if delta > 0 else "гірше")
        rows.append(
            {
                "name": r["name"],
                "group": r["group"],
                "act": r["act"],
                "init": r["init"],
                "ext_mean": round(float(x.mean()), 5),
                "ext_std": round(float(x.std()), 5),
                "delta_vs_relu": round(delta, 5),
                "se": round(se, 5) if np.isfinite(se) else "",
                "z": round(z, 2) if np.isfinite(z) else "",
                "verdict": verdict,
            }
        )
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n== АКТИВАЦІЇ: Δ macro-F1 відносно '{base_name}' ==")
    print(f"   {'конфіг':46s} {'mean':>6s}  {'Δ':>7s}  {'Δ/SE':>6s}  висновок")
    for r in rows:
        zs = f"{r['z']:6.1f}" if r["z"] != "" else "   n/a"
        print(
            f"   {r['name']:46s} {r['ext_mean']:.3f}  {r['delta_vs_relu']:+.4f}  {zs}  {r['verdict']}"
        )
    print(
        "   |Δ/SE| < 2 -- різниця не відрізняється від шуму ініціалізацій. Конфігів багато, тож\n"
        "   для висновку вимагай |Δ/SE| >= 3 і відтворюваності на більшій кількості --seeds.\n"
        "   SE враховує лише розкид між ініціалізаціями, а не вибірку даних."
    )


# ----------------------------------------------------------------------------- головна частина
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", default="team", help="папка з train.csv і valid.csv")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument(
        "--seeds", type=int, default=10, help="кількість ініціалізацій на конфіг"
    )
    ap.add_argument("--quick", action="store_true", help="3 ініціалізації, 60 епох")
    ap.add_argument("--base-window", type=int, default=8)
    ap.add_argument(
        "--groups",
        default="windows,multi,features,ablation,lookahead,weights,hidden,norm",
        help="за замовчуванням БЕЗ груп act/act_init/act_lr/depth; додай їх явно: --groups depth або --groups act,act_init",
    )
    ap.add_argument(
        "--hidden",
        default="32",
        help="розмір прихованих шарів за замовчуванням: '32' (1 шар) або '32,16' / '32x16' (2 шари)",
    )
    ap.add_argument(
        "--norm",
        default="zscore",
        choices=NORMS,
        help="нормалізація для всіх груп, крім norm",
    )
    ap.add_argument(
        "--cw",
        default="full",
        choices=("none", "sqrt", "full"),
        help="ваги класів для всіх груп, крім weights (за замовчуванням full, як у попередніх прогонах)",
    )
    ap.add_argument(
        "--act",
        default="relu",
        choices=ACT_ORDER,
        help="активація прихованих шарів для всіх груп, крім act/act_init",
    )
    ap.add_argument(
        "--init",
        default="auto",
        choices=INITS,
        help="ініціалізація для всіх груп, крім act/act_init (auto: He для ReLU-сімейства, Xavier для tanh/sigmoid/linear)",
    )
    ap.add_argument("--lr", type=float, default=1e-3, help="learning rate Adam")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument(
        "--tag",
        default="",
        help="суфікс імен файлів результатів, щоб не затирати попередні прогони",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="дозволити перезапис feature_window_study.csv без --tag",
    )
    ap.add_argument("--skip-selftest", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.epochs, args.patience = 3, 60, 10
    hidden_default = parse_hidden(args.hidden)

    tag = f"_{args.tag}" if args.tag else ""
    os.makedirs(args.out_dir, exist_ok=True)
    runs_path = os.path.join(args.out_dir, f"feature_window_study{tag}.csv")
    sum_path = os.path.join(args.out_dir, f"feature_window_study_summary{tag}.csv")
    meta_path = os.path.join(args.out_dir, f"feature_window_study_meta{tag}.json")
    cmp_path = os.path.join(args.out_dir, f"activation_compare{tag}.csv")
    lrtab_path = os.path.join(args.out_dir, f"activation_lr_compare{tag}.csv")
    depth_path = os.path.join(args.out_dir, f"depth_compare{tag}.csv")
    if not args.tag and not args.overwrite and os.path.exists(runs_path):
        print(
            f"{runs_path} вже існує. Щоб не затерти попередні результати, задай --tag <ім'я> "
            "(наприклад --tag depth) або --overwrite."
        )
        return 1

    train = load_csv(os.path.join(args.data_dir, "train.csv"))
    valid = load_csv(os.path.join(args.data_dir, "valid.csv"))
    print(f"train: {len(train['label'])} рядків, valid: {len(valid['label'])} рядків")
    print(
        "частки класів у train:",
        np.round(np.bincount(train["label"]) / len(train["label"]), 3).tolist(),
    )

    if not args.skip_selftest:
        print("\n== САМОПЕРЕВІРКИ ==")
        res = run_selftests(train, valid)
        for name, ok, msg in res:
            print(
                ("  PASS  " if ok else "  FAIL  ")
                + name
                + (f"  -> {msg}" if msg else "")
            )
        if not all(ok for _, ok, _ in res):
            print(
                "\nСамоперевірки не пройшли, експеримент зупинено (виправ дані/код або --skip-selftest)."
            )
            return 1

    groups = {g.strip() for g in args.groups.split(",") if g.strip()}
    known = {
        "windows",
        "multi",
        "features",
        "ablation",
        "lookahead",
        "weights",
        "hidden",
        DEPTH_GROUP,
        "norm",
        *ACT_GROUPS,
        LR_GROUP,
    }
    unknown = groups - known
    if unknown:
        print(f"Невідомі групи: {sorted(unknown)}. Доступні: {sorted(known)}")
        return 1
    configs = make_configs(
        groups,
        args.base_window,
        args.norm,
        args.cw,
        args.act,
        args.init,
        args.lr,
        hidden_default=hidden_default,
    )
    fields = [
        "group",
        "name",
        "arch",
        "n_params",
        "act",
        "init",
        "lr",
        "seed",
        "dim",
        "val_f1",
        "train_f1",
        "ext_f1",
        "ext_f1_c0",
        "ext_f1_c1",
        "ext_f1_c2",
        "const_units",
        "best_epoch",
        "epochs_run",
        "max_abs_input",
        "seconds",
    ]
    cache, summary_rows, ext_by_name = {}, [], {}
    n_train_all = len(train["label"])
    n_tr = int(0.8 * n_train_all)

    if "norm" in groups:
        dcfg = {
            "kind": "hand",
            "scales": [(args.base_window, "c")],
            "extras": (),
            "drop": (),
        }
        Xd, nd, Pd, Fd = build_features(train["raw"], dcfg, args.base_window)
        Xd = Xd[np.arange(Pd, n_tr - (Pd + Fd))]
        q1, q3 = np.percentile(Xd, [25, 75], axis=0)
        print(
            "\n== ДІАГНОСТИКА для robust: частка нулів і IQR кожної ознаки (train-частина) =="
        )
        for j, nm_ in enumerate(nd):
            tag_ = (
                "  <-- IQR=0, robust не масштабує цю ознаку"
                if q3[j] - q1[j] < 1e-12
                else ""
            )
            print(
                f"   {nm_:16s} нулі: {np.mean(Xd[:, j] == 0):5.1%}   IQR={q3[j] - q1[j]:9.3f}   "
                f"std={Xd[:, j].std():9.3f}   max={Xd[:, j].max():9.1f}{tag_}"
            )

    print(f"\n== ЕКСПЕРИМЕНТ: {len(configs)} конфігів x {args.seeds} ініціалізацій ==")
    t_start = time.time()
    with open(runs_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for ci, cfg in enumerate(configs, 1):
            init_used = resolve_init(cfg["act"], cfg["init"])
            arch_str = format_hidden(cfg["hidden"])
            key = (
                cfg["kind"],
                tuple(cfg["scales"]),
                cfg["extras"],
                cfg["drop"],
                cfg["hidden"],
                cfg["cw"],
                cfg["norm"],
                cfg["act"],
                init_used,
                cfg["lr"],
                args.base_window if cfg["kind"] != "hand" else 0,
            )
            if key in cache:
                runs = cache[key]
                note = " (повтор конфігу, взято з кешу)"
            else:
                Xtr_all, names, P, F = build_features(
                    train["raw"], cfg, args.base_window
                )
                Xext, _, _, _ = build_features(valid["raw"], cfg, args.base_window)
                y_all, yext = train["label"], valid["label"]
                tr_idx = np.arange(P, n_tr - (P + F))  # warm-up і розрив між частинами
                va_idx = np.arange(n_tr, n_train_all)
                runs = []
                for s in range(args.seeds):
                    t0 = time.time()
                    r = fit_eval(
                        Xtr_all[tr_idx],
                        y_all[tr_idx],
                        Xtr_all[va_idx],
                        y_all[va_idx],
                        Xext,
                        yext,
                        s,
                        cfg["hidden"],
                        cfg["cw"],
                        args.epochs,
                        args.patience,
                        cfg["norm"],
                        lr=cfg["lr"],
                        act=cfg["act"],
                        init=cfg["init"],
                    )
                    r["seed"], r["dim"], r["seconds"] = (
                        s,
                        Xtr_all.shape[1],
                        time.time() - t0,
                    )
                    runs.append(r)
                cache[key] = runs
                note = ""
            for r in runs:
                wr.writerow(
                    {
                        "group": cfg["group"],
                        "name": cfg["name"],
                        "arch": arch_str,
                        "n_params": r["n_params"],
                        "act": cfg["act"],
                        "init": init_used,
                        "lr": cfg["lr"],
                        "seed": r["seed"],
                        "dim": r["dim"],
                        "val_f1": round(r["val_f1"], 5),
                        "train_f1": round(r["train_f1"], 5),
                        "ext_f1": round(r["ext_f1"], 5),
                        "ext_f1_c0": round(r["ext_per_class"][0], 5),
                        "ext_f1_c1": round(r["ext_per_class"][1], 5),
                        "ext_f1_c2": round(r["ext_per_class"][2], 5),
                        "const_units": round(r["const_units"], 4),
                        "best_epoch": r["best_epoch"],
                        "epochs_run": r["epochs_run"],
                        "max_abs_input": round(r["max_abs_input"], 2),
                        "seconds": round(r["seconds"], 2),
                    }
                )
            fh.flush()
            ext = np.array([r["ext_f1"] for r in runs])
            val = np.array([r["val_f1"] for r in runs])
            pick = runs[int(np.argmax(val))]
            per = np.mean([r["ext_per_class"] for r in runs], axis=0)
            ext_by_name[cfg["name"]] = ext
            row = {
                "group": cfg["group"],
                "name": cfg["name"],
                "arch": arch_str,
                "n_params": runs[0]["n_params"],
                "act": cfg["act"],
                "init": init_used,
                "lr": cfg["lr"],
                "dim": runs[0]["dim"],
                "val_mean": val.mean(),
                "train_f1": float(np.mean([r["train_f1"] for r in runs])),
                "ext_mean": ext.mean(),
                "ext_std": ext.std(),
                "ext_min": ext.min(),
                "ext_max": ext.max(),
                "ext_at_best_val": pick["ext_f1"],
                "const_units": float(np.mean([r["const_units"] for r in runs])),
                "best_epoch_mean": float(np.mean([r["best_epoch"] for r in runs])),
                "capped_frac": float(
                    np.mean([r["epochs_run"] >= args.epochs for r in runs])
                ),
                "max_abs_input": runs[0]["max_abs_input"],
                "f1_c0": per[0],
                "f1_c1": per[1],
                "f1_c2": per[2],
            }
            summary_rows.append(row)
            print(
                f"[{ci:2d}/{len(configs)}] {cfg['name']:46s} dim={row['dim']:3d}  "
                f"ext={row['ext_mean']:.3f}±{row['ext_std']:.3f}  [{row['ext_min']:.3f}..{row['ext_max']:.3f}]{note}",
                flush=True,
            )

    with open(sum_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        wr.writeheader()
        for r in summary_rows:
            wr.writerow(
                {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}
            )

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        commit = ""
    meta = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "commit": commit,
        "args": vars(args),
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "train_sha256_16": sha256_of(os.path.join(args.data_dir, "train.csv")),
        "valid_sha256_16": sha256_of(os.path.join(args.data_dir, "valid.csv")),
        "protocol": "ознаки по всьому файлу; блоковий спліт 80/20 з розривом P+F; вибір епохи за val-частиною; "
        "звіт на valid.csv",
        "total_seconds": round(time.time() - t_start, 1),
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print("\n== ПІДСУМОК (macro-F1 на valid.csv; найкращі в групі позначено *) ==")
    for g in (
        "windows",
        "multi",
        "features",
        "ablation",
        "lookahead",
        "weights",
        "hidden",
        DEPTH_GROUP,
        "norm",
        "act",
        "act_init",
        LR_GROUP,
    ):
        rows = [r for r in summary_rows if r["group"] == g]
        if not rows:
            continue
        top = max(r["ext_mean"] for r in rows)
        print(f"\n-- {g} --")
        for r in rows:
            star = "*" if r["ext_mean"] == top else " "
            extra = ""
            if g == "norm":
                extra = f"   max|x'|={r['max_abs_input']:.1f}"
            elif g in (DEPTH_GROUP, "hidden"):
                extra = (
                    f"   params={r['n_params']}  train={r['train_f1']:.3f}"
                    f"  мертвих={r['const_units']:.0%}  епоха={r['best_epoch_mean']:.0f}"
                )
            elif g in ACT_GROUPS or g == LR_GROUP:
                extra = (
                    f"   train={r['train_f1']:.3f}  мертвих={r['const_units']:.0%}"
                    f"  епоха={r['best_epoch_mean']:.0f}  до ліміту={r['capped_frac']:.0%}"
                )
            print(
                f" {star} {r['name']:46s} {r['ext_mean']:.3f}±{r['ext_std']:.3f}   "
                f"F1 класів 0/1/2: {r['f1_c0']:.2f}/{r['f1_c1']:.2f}/{r['f1_c2']:.2f}{extra}"
            )

    if DEPTH_GROUP in groups:
        depth_compare(
            summary_rows,
            ext_by_name,
            f"hand w={args.base_window} arch=32 (1-шар)",
            depth_path,
        )

    if LR_GROUP in groups:
        print_lr_table(summary_rows, ext_by_name, lrtab_path)

    if groups & set(ACT_GROUPS):
        base_act = f"hand w={args.base_window} act=relu"
        if hidden_default != (32,):
            base_act = f"{base_act} [h={format_hidden(hidden_default)}]"
        activation_compare(
            summary_rows, ext_by_name, base_act, cmp_path
        )

    print(f"\nГотово за {time.time() - t_start:.0f} с. Файли: {runs_path}, {sum_path}")
    print(
        "±  це розкид лише між ініціалізаціями. Різниці менші за ±std не інтерпретуй."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
