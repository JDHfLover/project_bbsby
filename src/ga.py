"""Модуль гілки `ga`: генетичний алгоритм, Random Search, межі генів, журнал еволюції та адаптер конвеєра."""
from __future__ import annotations

import csv
import itertools
import json
import os
import random
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

GENE_BOUNDS: dict[str, tuple[float | int, float | int, str]] = {
    "block_size": (16, 64, "int"),
    "compression_threshold": (0.0, 1.0, "float"),
    "priority_weight": (0.0, 10.0, "float"),
}
OPTIONAL_PACKET_GENE_BOUNDS: dict[str, tuple[float | int, float | int, str]] = {
    "max_packet_bytes": (192, 512, "int"),
}

FORBIDDEN_CHANNEL_SEED = 425243
DEFAULT_TRAIN_SEEDS: tuple[int, ...] = (11, 12)
DEFAULT_VAL_SEEDS: tuple[int, ...] = (21, 22, 23, 24, 25)
DEFAULT_TRAIN_SCENARIOS: tuple[str, ...] = ("робочий", "аварійний")
DEFAULT_VAL_SCENARIOS: tuple[str, ...] = ("спокійний", "робочий", "аварійний")
DEFAULT_GRID_BLOCK_SIZES: tuple[int, ...] = (16, 26, 36, 48, 64)
DEFAULT_GRID_COMP_THRESHOLDS: tuple[float, ...] = (0.0, 0.35, 0.50, 0.90)
DEFAULT_GRID_PRIORITY_WEIGHTS: tuple[float, ...] = (0.0, 5.0, 10.0)

CROSSOVER_KINDS = ("uniform", "single_point", "arithmetic")

_REPO_ROOT = Path(__file__).resolve().parent.parent


def get_default_hyperparams(config: dict | None = None) -> dict[str, int | float]:
    """Повертає базові параметри системи для вимірювання Q_baseline (§3.1–3.2 design/ga.md)."""
    pack_cfg = config.get("пакування", config) if isinstance(config, dict) else {}
    prio_cfg = config.get("пріоритети_кадрів", {}) if isinstance(config, dict) else {}
    return {
        "block_size": int(pack_cfg.get("candidate_window_frames", 64)),
        "compression_threshold": float(pack_cfg.get("compression_threshold", 0.0)),
        "priority_weight": float(prio_cfg.get("critical_бонус", 5.0)),
    }


def parse_bounds_spec(
    spec: str | None,
) -> dict[str, tuple[float | int, float | int, str]]:
    """
    Розбирає рядок меж генів із CLI `--bounds`, наприклад:
    `block_size=16:64:int,compression_threshold=0:1:float,priority_weight=0:10:float`.
    """
    if spec is None or not spec.strip():
        return dict(GENE_BOUNDS)

    parsed: dict[str, tuple[float | int, float | int, str]] = {}
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Некоректний формат межі {item!r}; очікується name=lo:hi[:int|float]")
        name, rhs = item.split("=", 1)
        name = name.strip()
        parts = [p.strip() for p in rhs.split(":")]
        if len(parts) not in (2, 3):
            raise ValueError(f"Некоректний діапазон для {name!r}: {rhs!r}")
        kind = parts[2] if len(parts) == 3 else (GENE_BOUNDS[name][2] if name in GENE_BOUNDS else "float")
        if kind not in ("int", "float"):
            raise ValueError(f"Невідомий тип гена {name!r}: {kind!r}")
        lo = int(round(float(parts[0]))) if kind == "int" else float(parts[0])
        hi = int(round(float(parts[1]))) if kind == "int" else float(parts[1])
        parsed[name] = (lo, hi, kind)

    if not parsed:
        return dict(GENE_BOUNDS)
    if len(parsed) < len(GENE_BOUNDS) and all(k in GENE_BOUNDS for k in parsed):
        merged = dict(GENE_BOUNDS)
        merged.update(parsed)
        return _resolve_bounds(merged)
    return _resolve_bounds(parsed)


def validate_seed_split(
    train_seeds: Sequence[int] = DEFAULT_TRAIN_SEEDS,
    val_seeds: Sequence[int] = DEFAULT_VAL_SEEDS,
) -> None:
    """Перевіряє, що навчальні та валідаційні seed не порожні, не перетинаються і не містять 425243."""
    if not train_seeds or not val_seeds:
        raise ValueError("Списки train_seeds і val_seeds не можуть бути порожніми")
    train_set = set(train_seeds)
    val_set = set(val_seeds)
    if FORBIDDEN_CHANNEL_SEED in train_set or FORBIDDEN_CHANNEL_SEED in val_set:
        raise ValueError(
            f"Відкритий налагоджувальний seed {FORBIDDEN_CHANNEL_SEED} заборонено використовувати у ГА"
        )
    overlap = train_set & val_set
    if overlap:
        raise ValueError(f"train_seeds і val_seeds не повинні перетинатися: {sorted(overlap)}")


def _resolve_bounds(
    bounds: Mapping[str, tuple[float | int, float | int, str]] | None,
) -> dict[str, tuple[float | int, float | int, str]]:
    eff = dict(GENE_BOUNDS if bounds is None else bounds)
    if not eff:
        raise ValueError("Словник меж генів не може бути порожнім")
    for name, (lo, hi, kind) in eff.items():
        if kind not in ("int", "float"):
            raise ValueError(f"Невідомий тип гена {name!r}: {kind!r}")
        if float(lo) > float(hi):
            raise ValueError(f"Некоректні межі для {name!r}: ({lo}, {hi})")
    return eff


def clip_and_repair_chromosome(
    genes: Mapping[str, float | int],
    bounds: Mapping[str, tuple[float | int, float | int, str]] = GENE_BOUNDS,
) -> dict[str, int | float]:
    """Кліпує гени у задані межі та відновлює точний тип (`int` для цілих, `float` для дійсних)."""
    repaired: dict[str, int | float] = {}
    for name, (lo, hi, kind) in bounds.items():
        if name not in genes:
            raise KeyError(f"У хромосомі відсутній ген {name!r}")
        raw_val = float(genes[name])
        if kind == "int":
            lo_i, hi_i = int(lo), int(hi)
            val_i = int(round(raw_val))
            repaired[name] = max(lo_i, min(hi_i, val_i))
        else:
            lo_f, hi_f = float(lo), float(hi)
            repaired[name] = float(max(lo_f, min(hi_f, raw_val)))
    return repaired


def sample_random_chromosome(
    rng: random.Random,
    bounds: Mapping[str, tuple[float | int, float | int, str]] = GENE_BOUNDS,
) -> dict[str, int | float]:
    """Рівномірно генерує одну хромосому у просторі `bounds`."""
    genes: dict[str, int | float] = {}
    for name, (lo, hi, kind) in bounds.items():
        if kind == "int":
            genes[name] = rng.randint(int(lo), int(hi))
        else:
            genes[name] = rng.uniform(float(lo), float(hi))
    return genes


def tournament_select(
    population: Sequence[Mapping[str, int | float]],
    fitnesses: Sequence[float],
    rng: random.Random,
    k: int = 3,
) -> dict[str, int | float]:
    """Турнірна селекція розміру `k` (при рівності повертає першого з обраних кандидатів)."""
    if not population or len(population) != len(fitnesses):
        raise ValueError("Некоректні розміри population та fitnesses")
    if k < 1:
        raise ValueError(f"Розмір турніру k має бути >= 1, отримано {k}")

    eff_k = min(k, len(population))
    candidates = rng.sample(range(len(population)), eff_k)
    best_idx = max(candidates, key=lambda idx: fitnesses[idx])
    return dict(population[best_idx])


def crossover_chromosomes(
    parent1: Mapping[str, int | float],
    parent2: Mapping[str, int | float],
    rng: random.Random,
    crossover_rate: float = 0.8,
    kind: str = "uniform",
    bounds: Mapping[str, tuple[float | int, float | int, str]] = GENE_BOUNDS,
) -> dict[str, int | float]:
    """Виконує кросовер (`uniform`, `single_point` або `arithmetic`) і ремонт меж."""
    if kind not in CROSSOVER_KINDS:
        raise ValueError(f"Невідомий тип кросоверу {kind!r}; допустимо: {CROSSOVER_KINDS}")

    keys = list(bounds.keys())
    if rng.random() >= crossover_rate:
        return clip_and_repair_chromosome(parent1, bounds=bounds)

    child: dict[str, int | float] = {}
    if kind == "uniform":
        for name in keys:
            child[name] = parent1[name] if rng.random() < 0.5 else parent2[name]
    elif kind == "single_point":
        if len(keys) <= 1:
            child = dict(parent1)
        else:
            cut = rng.randint(1, len(keys) - 1)
            for idx, name in enumerate(keys):
                child[name] = parent1[name] if idx < cut else parent2[name]
    else:  # arithmetic
        alpha = rng.random()
        for name in keys:
            child[name] = alpha * float(parent1[name]) + (1.0 - alpha) * float(parent2[name])

    return clip_and_repair_chromosome(child, bounds=bounds)


def mutate_chromosome(
    genes: Mapping[str, int | float],
    rng: random.Random,
    mutation_rate: float = 0.2,
    sigma_scale: float = 0.1,
    bounds: Mapping[str, tuple[float | int, float | int, str]] = GENE_BOUNDS,
) -> dict[str, int | float]:
    """
    Гаусова мутація (`sigma = sigma_scale * (hi - lo)`) із гарантованим кроком `±1`
    для цілих генів при нульовому округленому кроці та кліпінгом у `bounds`.
    """
    out: dict[str, int | float] = dict(genes)
    for name, (lo, hi, kind) in bounds.items():
        if rng.random() >= mutation_rate:
            continue
        span = float(hi) - float(lo)
        if span <= 0.0:
            continue

        step = rng.gauss(0.0, sigma_scale * span)
        if kind == "int":
            lo_i, hi_i = int(lo), int(hi)
            cur = max(lo_i, min(hi_i, int(round(float(out[name])))))
            int_step = int(round(step))
            if int_step == 0:
                int_step = 1 if rng.random() < 0.5 else -1
            cand = cur + int_step
            if cand < lo_i:
                cand = (lo_i + 1) if (cur == lo_i and lo_i < hi_i) else lo_i
            elif cand > hi_i:
                cand = (hi_i - 1) if (cur == hi_i and lo_i < hi_i) else hi_i
            out[name] = cand
        else:
            out[name] = float(out[name]) + step

    return clip_and_repair_chromosome(out, bounds=bounds)


def save_ga_history_csv(
    history_rows: Sequence[Mapping[str, object]],
    path: str | os.PathLike[str],
    gene_names: Sequence[str] | None = None,
) -> None:
    """Зберігає журнал еволюції ГА (`run_seed, generation, best_fitness, mean_fitness, evals_total, ...`)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eff_genes = list(GENE_BOUNDS.keys() if gene_names is None else gene_names)
    fieldnames = ["run_seed", "generation", "best_fitness", "mean_fitness", "evals_total", *eff_genes]

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in history_rows:
            writer.writerow({col: row.get(col, "") for col in fieldnames})


def save_best_params_json(
    best_genes: Mapping[str, object],
    best_fitness: float,
    path: str | os.PathLike[str],
    seed: int = 0,
    extra: Mapping[str, object] | None = None,
) -> None:
    """Зберігає знайдену найкращу хромосому й метадані у JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "best_genes": dict(best_genes),
        "best_fitness": float(best_fitness),
        "seed": int(seed),
    }
    if extra:
        payload.update(dict(extra))
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def optimize_with_ga(
    fitness_fn: Callable[[dict[str, int | float]], float],
    config: dict | None = None,
    budget: int = 200,
    pop_size: int = 20,
    elitism: int = 2,
    tournament_k: int = 3,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.2,
    sigma_scale: float = 0.1,
    crossover_kind: str = "uniform",
    reevaluate_elite: bool = False,
    eval_repeats: int = 1,
    seed: int = 0,
    bounds: Mapping[str, tuple[float | int, float | int, str]] | None = None,
    return_history: bool = False,
    history_csv_path: str | os.PathLike[str] | None = None,
) -> tuple[dict[str, int | float], float] | tuple[dict[str, int | float], float, list[dict[str, object]]]:
    """
    Генетичний алгоритм з елітизмом, турнірною селекцією (k=3), кросовером і мутацією.
    Гарантує використання рівно `budget` обчислень пристосованості (якщо `budget >= 1`).
    """
    if budget < 1:
        raise ValueError(f"Бюджет обчислень має бути >= 1, отримано {budget}")
    if pop_size < 1:
        raise ValueError(f"Розмір популяції має бути >= 1, отримано {pop_size}")
    if elitism < 0 or elitism >= pop_size:
        raise ValueError(f"Еліта має бути в межах 0 <= elitism < pop_size ({pop_size}), отримано {elitism}")
    if tournament_k < 1:
        raise ValueError(f"tournament_k має бути >= 1, отримано {tournament_k}")
    if eval_repeats < 1:
        raise ValueError(f"eval_repeats має бути >= 1, отримано {eval_repeats}")

    eff_bounds = _resolve_bounds(bounds)
    rng = random.Random(seed)
    evals_used = 0

    best_genes: dict[str, int | float] | None = None
    best_fitness = float("-inf")
    history: list[dict[str, object]] = []

    def evaluate(ind: Mapping[str, int | float]) -> float:
        nonlocal evals_used, best_genes, best_fitness
        n_calls = min(eval_repeats, budget - evals_used)
        if n_calls <= 0:
            raise RuntimeError("Спроба обчислення після вичерпання бюджету")
        total = 0.0
        ind_copy = dict(ind)
        for _ in range(n_calls):
            total += float(fitness_fn(dict(ind_copy)))
            evals_used += 1
        fit = total / n_calls
        if best_genes is None or fit > best_fitness:
            best_fitness = fit
            best_genes = dict(ind_copy)
        return fit

    def record_generation(gen: int, pop: list[dict[str, int | float]], fits: list[float]) -> None:
        top_idx = max(range(len(pop)), key=lambda idx: fits[idx])
        row: dict[str, object] = {
            "run_seed": seed,
            "generation": gen,
            "best_fitness": float(fits[top_idx]),
            "mean_fitness": float(sum(fits) / len(fits)),
            "evals_total": evals_used,
            **{k: pop[top_idx][k] for k in eff_bounds},
        }
        history.append(row)

    # Покоління 0
    population: list[dict[str, int | float]] = []
    fitnesses: list[float] = []
    while len(population) < pop_size and evals_used < budget:
        cand = sample_random_chromosome(rng, bounds=eff_bounds)
        fit = evaluate(cand)
        population.append(cand)
        fitnesses.append(fit)

    record_generation(0, population, fitnesses)

    gen = 0
    while evals_used < budget:
        gen += 1
        order = sorted(range(len(population)), key=lambda idx: fitnesses[idx], reverse=True)
        next_pop: list[dict[str, int | float]] = []
        next_fit: list[float] = []

        # Перенесення еліти
        for idx in order[:elitism]:
            elite = dict(population[idx])
            if reevaluate_elite:
                if evals_used >= budget:
                    break
                next_pop.append(elite)
                next_fit.append(evaluate(elite))
            else:
                next_pop.append(elite)
                next_fit.append(fitnesses[idx])

        # Генерація нащадків до заповнення популяції або вичерпання бюджету
        while len(next_pop) < pop_size and evals_used < budget:
            p1 = tournament_select(population, fitnesses, rng, k=tournament_k)
            p2 = tournament_select(population, fitnesses, rng, k=tournament_k)
            child = crossover_chromosomes(
                p1,
                p2,
                rng,
                crossover_rate=crossover_rate,
                kind=crossover_kind,
                bounds=eff_bounds,
            )
            child = mutate_chromosome(
                child,
                rng,
                mutation_rate=mutation_rate,
                sigma_scale=sigma_scale,
                bounds=eff_bounds,
            )
            child_fit = evaluate(child)
            next_pop.append(child)
            next_fit.append(child_fit)

        # Якщо бюджет закінчився посеред покоління — добираємо найкращих із попереднього покоління
        if len(next_pop) < pop_size:
            for idx in order[elitism:]:
                if len(next_pop) >= pop_size:
                    break
                next_pop.append(dict(population[idx]))
                next_fit.append(fitnesses[idx])

        population = next_pop
        fitnesses = next_fit
        record_generation(gen, population, fitnesses)

    if history_csv_path is not None:
        save_ga_history_csv(history, history_csv_path, gene_names=list(eff_bounds.keys()))

    # При стандартному елітизмі найкраща особина фінального покоління збігається з best_genes;
    # повертаємо best_genes/best_fitness по поколіннях або знайдений оптимум.
    final_top_idx = max(range(len(population)), key=lambda idx: fitnesses[idx])
    final_best_genes = dict(population[final_top_idx])
    final_best_fit = float(fitnesses[final_top_idx])
    if not reevaluate_elite and elitism > 0:
        out_genes, out_fit = final_best_genes, final_best_fit
    else:
        assert best_genes is not None
        out_genes, out_fit = dict(best_genes), float(best_fitness)

    if return_history:
        return out_genes, out_fit, history
    return out_genes, out_fit


def optimize_with_random_search(
    fitness_fn: Callable[[dict[str, int | float]], float],
    config: dict | None = None,
    budget: int = 200,
    seed: int = 0,
    bounds: Mapping[str, tuple[float | int, float | int, str]] | None = None,
    eval_repeats: int = 1,
    return_history: bool = False,
) -> tuple[dict[str, int | float], float] | tuple[dict[str, int | float], float, list[dict[str, object]]]:
    """Базовий пошук випадковими хромосомами за однакового бюджету обчислень `budget`."""
    if budget < 1:
        raise ValueError(f"Бюджет обчислень має бути >= 1, отримано {budget}")
    if eval_repeats < 1:
        raise ValueError(f"eval_repeats має бути >= 1, отримано {eval_repeats}")

    eff_bounds = _resolve_bounds(bounds)
    rng = random.Random(seed)
    evals_used = 0
    best_genes: dict[str, int | float] | None = None
    best_fitness = float("-inf")
    history: list[dict[str, object]] = []

    step = 0
    while evals_used < budget:
        cand = sample_random_chromosome(rng, bounds=eff_bounds)
        n_calls = min(eval_repeats, budget - evals_used)
        total = 0.0
        for _ in range(n_calls):
            total += float(fitness_fn(dict(cand)))
            evals_used += 1
        fit = total / n_calls
        if best_genes is None or fit > best_fitness:
            best_fitness = fit
            best_genes = dict(cand)
        history.append(
            {
                "run_seed": seed,
                "eval_step": step,
                "evals_total": evals_used,
                "candidate_fitness": float(fit),
                "best_fitness": float(best_fitness),
                **{k: best_genes[k] for k in eff_bounds},
            }
        )
        step += 1

    assert best_genes is not None
    if return_history:
        return dict(best_genes), float(best_fitness), history
    return dict(best_genes), float(best_fitness)


def synthetic_fitness(
    genes: Mapping[str, float | int],
    noise_std: float = 0.0,
    rng: random.Random | None = None,
) -> float:
    """
    Автономний гладкий ландшафт пристосованості у [0, 1] з відомим максимумом
    (block_size=36, compression_threshold=0.28, priority_weight=6.5, Q=0.85)
    для незалежного тестування ГА окремо від конвеєра.
    """
    b = float(genes["block_size"])
    c = float(genes["compression_threshold"])
    p = float(genes["priority_weight"])

    d_b = (b - 36.0) / 24.0
    d_c = (c - 0.28) / 0.35
    d_p = (p - 6.5) / 4.0

    q = 0.85 - 0.12 * (d_b * d_b) - 0.10 * (d_c * d_c) - 0.11 * (d_p * d_p) - 0.03 * d_b * d_c
    if noise_std > 0.0:
        eff_rng = rng if rng is not None else random.Random(0)
        q += eff_rng.gauss(0.0, noise_std)
    return float(max(0.0, min(1.0, q)))


def synthetic_scenario_seed_q(
    genes: Mapping[str, float | int],
    scenario: str = "робочий",
    seed: int = 11,
    fixed_compute_time_s: float | None = 0.15,
    timing_noise_std: float = 0.0,
    rng: random.Random | None = None,
) -> float:
    """
    Автономна модель Q = 0.5*S + 0.35*R + 0.15*T за сценаріями й seed каналу (§3.1, §3.3, §3.4):
    - у сценарії 'спокійний' завжди R = 1.0 незалежно від priority_weight;
    - при fixed_compute_time_s != None час обчислень фіксований (детермінований T).
    """
    if scenario not in ("спокійний", "робочий", "аварійний"):
        raise ValueError(f"Невідомий сценарій каналу: {scenario!r}")
    if seed == FORBIDDEN_CHANNEL_SEED:
        raise ValueError(f"Відкритий seed {FORBIDDEN_CHANNEL_SEED} заборонено використовувати")

    b = float(genes["block_size"])
    c = float(genes["compression_threshold"])
    p = float(genes["priority_weight"])

    s = (
        0.62
        - 0.14 * ((b - 40.0) / 24.0) ** 2
        - 0.16 * (max(0.0, 0.25 - c) ** 2)
        - 0.12 * (max(0.0, c - 0.35) ** 2)
        - 0.03 * ((b - 40.0) / 24.0) * (c - 0.30)
    )
    s = max(0.0, min(1.0, s))

    seed_shift = ((int(seed) * 37) % 11 - 5) * 0.004
    if scenario == "спокійний":
        r = 1.0
    elif scenario == "робочий":
        r = 0.96 - 0.04 * ((p - 6.0) / 6.0) ** 2 - 0.02 * ((b - 36.0) / 28.0) ** 2 + 0.5 * seed_shift
    else:  # аварійний
        r = 0.88 - 0.22 * ((p - 6.5) / 6.5) ** 2 - 0.08 * ((b - 32.0) / 32.0) ** 2 + seed_shift
    r = max(0.0, min(1.0, r))
    if r == 0.0:
        return 0.0

    t_ref = 20.0
    if fixed_compute_time_s is not None:
        t_exec = max(0.0, float(fixed_compute_time_s))
    else:
        eff_rng = rng if rng is not None else random.Random()
        t_exec = max(0.001, 0.15 + eff_rng.gauss(0.0, timing_noise_std))
    t_comp = t_ref / (t_ref + t_exec)

    return float(0.5 * s + 0.35 * r + 0.15 * t_comp)


@dataclass
class _PackedBundle:
    packets: list[bytes]
    decoded_results: list[object]
    rx_times: list[float]
    tx_time_s: float


class PipelineSessionContext:
    """
    Кеш незмінного контексту сеансу (конфіг, кадри, класифікатор MLP) для швидкої оцінки
    хромосом у реальному конвеєрі без кешування результатів різних викликів fitness.
    """

    def __init__(
        self,
        repo_root: Path | None = None,
        config_path: str | os.PathLike[str] | None = None,
        session_path: str | os.PathLike[str] | None = None,
        salt: bytes | None = None,
        session_nonce_prefix: bytes = b"SESSGA01",
        device_id: int = 1,
    ) -> None:
        self.repo_root = Path(repo_root) if repo_root is not None else _REPO_ROOT
        self.config_path = Path(config_path) if config_path is not None else (self.repo_root / "team" / "team_config.json")
        self.session_path = Path(session_path) if session_path is not None else (self.repo_root / "team" / "session.bin")
        self.salt = salt if salt is not None else bytes.fromhex("000102030405060708090A0B0C0D0E0F")
        self.session_nonce_prefix = session_nonce_prefix
        self.device_id = device_id
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        from src.classification import CLASS_TO_KEY, classify_frames
        from src.telemetry import load_session_bin

        with open(self.config_path, encoding="utf-8") as fh:
            self.config: dict = json.load(fh)
        self.password: str = str(self.config.get("пароль_сеансу", "algo2026"))
        self.q_weights: dict = dict(self.config["метрика_Q"])
        self.scenarios_cfg: dict = dict(self.config["канал"]["сценарії"])
        self.prio_cfg: dict = dict(self.config["пріоритети_кадрів"])
        pack_cfg = self.config.get("пакування", {})
        self.max_packet_bytes: int = int(pack_cfg.get("max_packet_bytes", 512))

        self.frames = load_session_bin(str(self.session_path))
        self.classes = classify_frames(self.frames)
        self.base_priorities: list[int] = [
            int(self.prio_cfg[CLASS_TO_KEY[cls]]) for cls in self.classes
        ]
        self._loaded = True

    def pack_and_decode(self, genes: Mapping[str, float | int]) -> _PackedBundle:
        """Виконує TX-пакування й одноразове RX-декодування пакетів для заданої хромосоми."""
        from src.dp import pack_telemetry_stream
        from src.receiver import receive_packet_ex

        self._ensure_loaded()
        block_size = max(1, int(round(float(genes["block_size"]))))
        compression_threshold = float(genes["compression_threshold"])
        priority_weight = float(genes["priority_weight"])
        max_pkt_bytes = (
            int(round(float(genes["max_packet_bytes"])))
            if "max_packet_bytes" in genes
            else self.max_packet_bytes
        )

        t_tx0 = time.perf_counter()
        values = [
            max(0, int(round(base_p + (priority_weight if f.critical == 1 else 0.0))))
            for f, base_p in zip(self.frames, self.base_priorities)
        ]
        packets = pack_telemetry_stream(
            self.frames,
            values=values,
            config=self.config,
            max_packet_bytes=max_pkt_bytes,
            candidate_window_frames=block_size,
            compression_threshold=compression_threshold,
            priority_weight=1.0,
            password=self.password,
            salt=self.salt,
            session_nonce_prefix=self.session_nonce_prefix,
            device_id=self.device_id,
            schedule=True,
            seal=True,
        )
        tx_time_s = time.perf_counter() - t_tx0

        decoded_results: list[object] = []
        rx_times: list[float] = []
        for pkt in packets:
            t_rx0 = time.perf_counter()
            res = receive_packet_ex(pkt, self.password)
            rx_times.append(time.perf_counter() - t_rx0)
            decoded_results.append(res)

        return _PackedBundle(
            packets=packets,
            decoded_results=decoded_results,
            rx_times=rx_times,
            tx_time_s=tx_time_s,
        )

    def evaluate_packed_on_channel(
        self,
        packed_bundle: _PackedBundle,
        scenario: str,
        seed: int,
        fixed_compute_time_s: float | None = None,
        timing_noise_std: float = 0.0,
        rng: random.Random | None = None,
    ) -> dict[str, float]:
        """Проганяє вже упакований потік через `Channel(scenario, seed)` і рахує метрики S, R, T, Q."""
        from team.channel import Channel
        from src.channel_runner import evaluate_metrics

        self._ensure_loaded()
        if scenario not in self.scenarios_cfg:
            raise ValueError(f"Невідомий сценарій каналу: {scenario!r}")
        if seed == FORBIDDEN_CHANNEL_SEED:
            raise ValueError(f"Відкритий seed {FORBIDDEN_CHANNEL_SEED} заборонено використовувати")

        ch = Channel(self.scenarios_cfg[scenario], seed=seed)
        bytes_sent = 0
        recovered = []
        seen: set[tuple[int, bytes]] = set()
        rx_time_s = 0.0
        crashed = False

        for pkt, res, pkt_rx_t in zip(
            packed_bundle.packets, packed_bundle.decoded_results, packed_bundle.rx_times
        ):
            prev_sent = ch.sent
            delivered, _ = ch.send(pkt)
            if ch.sent > prev_sent:
                bytes_sent += len(pkt)
            if delivered:
                rx_time_s += pkt_rx_t
                if res.error is not None:  # type: ignore[attr-defined]
                    crashed = True
                else:
                    for frame in res.frames:  # type: ignore[attr-defined]
                        key = (frame.timestamp_ms, frame.to_bytes())
                        if key not in seen:
                            seen.add(key)
                            recovered.append(frame)

        recovered.sort(key=lambda f: f.timestamp_ms)

        raw_compute_time_s = (
            float(fixed_compute_time_s)
            if fixed_compute_time_s is not None
            else (packed_bundle.tx_time_s + rx_time_s)
        )
        if timing_noise_std > 0.0:
            eff_rng = rng if rng is not None else random.Random()
            raw_compute_time_s = max(0.001, raw_compute_time_s + eff_rng.gauss(0.0, timing_noise_std))

        metrics = evaluate_metrics(
            self.frames,
            recovered,
            bytes_sent,
            raw_compute_time_s,
            self.q_weights,
            crashed=crashed,
        )
        return {
            "S": float(metrics["S"]),
            "R": float(metrics["R"]),
            "T": float(metrics["T"]),
            "Q": float(metrics["Q"]),
            "crit_loss_rate": float(metrics["crit_loss_rate"]),
            "tx_time_s": float(packed_bundle.tx_time_s),
            "rx_time_s": float(rx_time_s),
            "compute_time_s": float(metrics["compute_time_s"]),
            "bytes_sent": float(metrics["bytes_sent"]),
        }


_DEFAULT_PIPELINE_CONTEXT: PipelineSessionContext | None = None


def get_default_pipeline_context() -> PipelineSessionContext:
    """Повертає спільний лінивий контекст конвеєра для `call_pipeline`."""
    global _DEFAULT_PIPELINE_CONTEXT
    if _DEFAULT_PIPELINE_CONTEXT is None:
        _DEFAULT_PIPELINE_CONTEXT = PipelineSessionContext()
    return _DEFAULT_PIPELINE_CONTEXT


def call_pipeline(
    genes: Mapping[str, float | int],
    scenario: str = "робочий",
    seed: int = 11,
    fixed_compute_time_s: float | None = None,
    timing_noise_std: float = 0.0,
    rng: random.Random | None = None,
    context: PipelineSessionContext | None = None,
) -> float:
    """Адаптер реального конвеєра: за хромосомою, сценарієм і seed каналу повертає Q."""
    ctx = context if context is not None else get_default_pipeline_context()
    bundle = ctx.pack_and_decode(genes)
    metrics = ctx.evaluate_packed_on_channel(
        bundle,
        scenario=scenario,
        seed=seed,
        fixed_compute_time_s=fixed_compute_time_s,
        timing_noise_std=timing_noise_std,
        rng=rng,
    )
    return float(metrics["Q"])


def _bracketing_weights(axis: Sequence[float], value: float) -> tuple[int, int, float]:
    if len(axis) <= 1 or value <= axis[0]:
        return 0, 0, 0.0
    if value >= axis[-1]:
        last = len(axis) - 1
        return last, last, 0.0
    for idx in range(len(axis) - 1):
        lo = axis[idx]
        hi = axis[idx + 1]
        if lo <= value <= hi:
            frac = 0.0 if hi == lo else (value - lo) / (hi - lo)
            return idx, idx + 1, frac
    last = len(axis) - 1
    return last, last, 0.0


class PipelineGridTable:
    """
    Попередньо обчислена на реальному конвеєрі 3D-сітка (block_size × compression_threshold × priority_weight)
    із трилінійною інтерполяцією (S, R, tx_time_s, rx_time_s) для швидких масових досліджень ГА.
    """

    def __init__(
        self,
        block_sizes: Sequence[int],
        compression_thresholds: Sequence[float],
        priority_weights: Sequence[float],
        scenarios: Sequence[str],
        seeds: Sequence[int],
        q_weights: Mapping[str, float],
        records: list[dict[str, dict[str, dict[str, float]]]],
        measured_noise_std: float = 0.0,
        measured_timing_noise_std: float = 0.0,
        median_compute_time_s: float = 0.15,
        single_eval_time_s: float = 0.0,
        build_total_s: float = 0.0,
        measured_compute_times_s: Sequence[float] = (),
    ) -> None:
        self.block_sizes = [int(x) for x in block_sizes]
        self.compression_thresholds = [float(x) for x in compression_thresholds]
        self.priority_weights = [float(x) for x in priority_weights]
        self.scenarios = list(scenarios)
        self.seeds = [int(s) for s in seeds]
        self.q_weights = dict(q_weights)
        self.records = records
        self.measured_noise_std = float(measured_noise_std)
        self.measured_timing_noise_std = float(measured_timing_noise_std)
        self.median_compute_time_s = float(median_compute_time_s)
        self.single_eval_time_s = float(single_eval_time_s)
        self.build_total_s = float(build_total_s)
        self.measured_compute_times_s = [float(t) for t in measured_compute_times_s]

        expected = (
            len(self.block_sizes)
            * len(self.compression_thresholds)
            * len(self.priority_weights)
        )
        if len(self.records) != expected:
            raise ValueError(f"Очікувалось {expected} вузлів сітки, отримано {len(self.records)}")

    def _node(self, ib: int, ic: int, ip: int) -> dict[str, dict[str, dict[str, float]]]:
        idx = (ib * len(self.compression_thresholds) + ic) * len(self.priority_weights) + ip
        return self.records[idx]

    def interpolate_metrics(
        self, genes: Mapping[str, float | int], scenario: str, seed: int
    ) -> tuple[float, float, float, float]:
        if scenario not in self.scenarios:
            raise ValueError(f"Сценарій {scenario!r} відсутній у таблиці")
        if seed == FORBIDDEN_CHANNEL_SEED:
            raise ValueError(f"Відкритий seed {FORBIDDEN_CHANNEL_SEED} заборонено використовувати")
        seed_key = str(int(seed))
        if seed not in self.seeds:
            raise ValueError(f"Seed {seed} відсутній у таблиці ({self.seeds})")

        ib0, ib1, tb = _bracketing_weights([float(v) for v in self.block_sizes], float(genes["block_size"]))
        ic0, ic1, tc = _bracketing_weights(self.compression_thresholds, float(genes["compression_threshold"]))
        ip0, ip1, tp = _bracketing_weights(self.priority_weights, float(genes["priority_weight"]))

        s_acc = 0.0
        r_acc = 0.0
        tx_acc = 0.0
        rx_acc = 0.0
        for ib, wb in ((ib0, 1.0 - tb), (ib1, tb)):
            for ic, wc in ((ic0, 1.0 - tc), (ic1, tc)):
                for ip, wp in ((ip0, 1.0 - tp), (ip1, tp)):
                    weight = wb * wc * wp
                    if weight == 0.0:
                        continue
                    cell = self._node(ib, ic, ip)[scenario][seed_key]
                    s_acc += weight * float(cell["S"])
                    r_acc += weight * float(cell["R"])
                    tx_acc += weight * float(cell["tx"])
                    rx_acc += weight * float(cell["rx"])

        return s_acc, r_acc, tx_acc, rx_acc

    def scenario_seed_q(
        self,
        genes: Mapping[str, float | int],
        scenario: str = "робочий",
        seed: int = 11,
        fixed_compute_time_s: float | None = None,
        timing_noise_std: float = 0.0,
        rng: random.Random | None = None,
    ) -> float:
        """Обчислює інтерпольоване `Q` за таблицею реального конвеєра."""
        s_val, r_val, tx_val, rx_val = self.interpolate_metrics(genes, scenario=scenario, seed=seed)
        if r_val <= 0.0:
            return 0.0

        t_exec = float(fixed_compute_time_s) if fixed_compute_time_s is not None else (tx_val + rx_val)
        if timing_noise_std > 0.0:
            eff_rng = rng if rng is not None else random.Random()
            t_exec = max(0.001, t_exec + eff_rng.gauss(0.0, timing_noise_std))

        t_ref = float(self.q_weights.get("t_ref_секунд", self.q_weights.get("t_ref", 20.0)))
        alpha = float(self.q_weights.get("alpha", 0.5))
        beta = float(self.q_weights.get("beta", 0.35))
        gamma = float(self.q_weights.get("gamma", 0.15))

        t_comp = (t_ref / (t_ref + max(0.0, t_exec))) if t_ref > 0 else 0.0
        q = alpha * s_val + beta * r_val + gamma * t_comp
        return float(max(0.0, min(1.0, q)))

    def to_dict(self) -> dict[str, object]:
        return {
            "block_sizes": self.block_sizes,
            "compression_thresholds": self.compression_thresholds,
            "priority_weights": self.priority_weights,
            "scenarios": self.scenarios,
            "seeds": self.seeds,
            "q_weights": self.q_weights,
            "measured_noise_std": self.measured_noise_std,
            "measured_timing_noise_std": self.measured_timing_noise_std,
            "median_compute_time_s": self.median_compute_time_s,
            "single_eval_time_s": self.single_eval_time_s,
            "build_total_s": self.build_total_s,
            "measured_compute_times_s": self.measured_compute_times_s,
            "records": self.records,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> PipelineGridTable:
        return cls(
            block_sizes=list(data["block_sizes"]),  # type: ignore[arg-type]
            compression_thresholds=list(data["compression_thresholds"]),  # type: ignore[arg-type]
            priority_weights=list(data["priority_weights"]),  # type: ignore[arg-type]
            scenarios=list(data["scenarios"]),  # type: ignore[arg-type]
            seeds=list(data["seeds"]),  # type: ignore[arg-type]
            q_weights=dict(data["q_weights"]),  # type: ignore[arg-type]
            records=list(data["records"]),  # type: ignore[arg-type]
            measured_noise_std=float(data.get("measured_noise_std", 0.0)),  # type: ignore[arg-type]
            measured_timing_noise_std=float(data.get("measured_timing_noise_std", 0.0)),  # type: ignore[arg-type]
            median_compute_time_s=float(data.get("median_compute_time_s", 0.15)),  # type: ignore[arg-type]
            single_eval_time_s=float(data.get("single_eval_time_s", 0.0)),  # type: ignore[arg-type]
            build_total_s=float(data.get("build_total_s", 0.0)),  # type: ignore[arg-type]
            measured_compute_times_s=list(data.get("measured_compute_times_s", ())),  # type: ignore[arg-type]
        )


def build_pipeline_grid_table(
    block_sizes: Sequence[int] = DEFAULT_GRID_BLOCK_SIZES,
    compression_thresholds: Sequence[float] = DEFAULT_GRID_COMP_THRESHOLDS,
    priority_weights: Sequence[float] = DEFAULT_GRID_PRIORITY_WEIGHTS,
    scenarios: Sequence[str] = DEFAULT_VAL_SCENARIOS,
    seeds: Sequence[int] = (*DEFAULT_TRAIN_SEEDS, *DEFAULT_VAL_SEEDS),
    noise_samples: int = 15,
    context: PipelineSessionContext | None = None,
) -> PipelineGridTable:
    """Будує 3D-сітку значень S, R і часу виконання на реальному конвеєрі."""
    if FORBIDDEN_CHANNEL_SEED in set(seeds):
        raise ValueError(f"Відкритий seed {FORBIDDEN_CHANNEL_SEED} заборонено використовувати")

    ctx = context if context is not None else get_default_pipeline_context()
    ctx._ensure_loaded()

    sorted_b = sorted({int(x) for x in block_sizes})
    sorted_c = sorted({float(x) for x in compression_thresholds})
    sorted_p = sorted({float(x) for x in priority_weights})
    eff_scenarios = list(scenarios)
    eff_seeds = [int(s) for s in seeds]

    t_build0 = time.perf_counter()
    records: list[dict[str, dict[str, dict[str, float]]]] = []
    train_eval_times: list[float] = []

    for b, c, p in itertools.product(sorted_b, sorted_c, sorted_p):
        genes = {"block_size": b, "compression_threshold": c, "priority_weight": p}
        t_node0 = time.perf_counter()
        bundle = ctx.pack_and_decode(genes)
        node_entry: dict[str, dict[str, dict[str, float]]] = {}
        train_pair_elapsed = 0.0
        t_pack_elapsed = time.perf_counter() - t_node0

        for sc in eff_scenarios:
            sc_map: dict[str, dict[str, float]] = {}
            for sd in eff_seeds:
                t_ch0 = time.perf_counter()
                m = ctx.evaluate_packed_on_channel(bundle, scenario=sc, seed=sd)
                if sc in DEFAULT_TRAIN_SCENARIOS and sd in DEFAULT_TRAIN_SEEDS:
                    train_pair_elapsed += time.perf_counter() - t_ch0
                sc_map[str(sd)] = {
                    "S": round(m["S"], 8),
                    "R": round(m["R"], 8),
                    "tx": round(m["tx_time_s"], 6),
                    "rx": round(m["rx_time_s"], 6),
                }
            node_entry[sc] = sc_map

        train_eval_times.append(t_pack_elapsed + train_pair_elapsed)
        records.append(node_entry)

    # Вимір шуму від фактичного часу виконання T на базовій хромосомі (§3.4–3.5 design/ga.md)
    default_genes = get_default_hyperparams(ctx.config)
    noise_scenario = "аварійний" if "аварійний" in eff_scenarios else eff_scenarios[-1]
    noise_seed = DEFAULT_VAL_SEEDS[0] if DEFAULT_VAL_SEEDS[0] in eff_seeds else eff_seeds[0]
    compute_times: list[float] = []
    q_samples: list[float] = []
    for _ in range(max(0, noise_samples)):
        bundle = ctx.pack_and_decode(default_genes)
        m = ctx.evaluate_packed_on_channel(bundle, scenario=noise_scenario, seed=noise_seed)
        compute_times.append(float(m["compute_time_s"]))
        q_samples.append(float(m["Q"]))

    build_total_s = time.perf_counter() - t_build0
    if len(compute_times) >= 2:
        measured_timing_noise_std = float(statistics.stdev(compute_times))
        measured_noise_std = float(statistics.stdev(q_samples))
        median_compute_time_s = float(statistics.median(compute_times))
    elif compute_times:
        measured_timing_noise_std = 0.0
        measured_noise_std = 0.0
        median_compute_time_s = float(compute_times[0])
    else:
        measured_timing_noise_std = 0.0
        measured_noise_std = 0.0
        median_compute_time_s = float(statistics.median(train_eval_times)) if train_eval_times else 0.15

    single_eval_time_s = float(statistics.mean(train_eval_times)) if train_eval_times else 0.0

    return PipelineGridTable(
        block_sizes=sorted_b,
        compression_thresholds=sorted_c,
        priority_weights=sorted_p,
        scenarios=eff_scenarios,
        seeds=eff_seeds,
        q_weights=ctx.q_weights,
        records=records,
        measured_noise_std=measured_noise_std,
        measured_timing_noise_std=measured_timing_noise_std,
        median_compute_time_s=median_compute_time_s,
        single_eval_time_s=single_eval_time_s,
        build_total_s=build_total_s,
        measured_compute_times_s=compute_times,
    )


def save_pipeline_grid_table(table: PipelineGridTable, path: str | os.PathLike[str]) -> None:
    """Зберігає попередньо обчислену таблицю конвеєра у JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(table.to_dict(), fh, ensure_ascii=False, indent=2)


def load_pipeline_grid_table(path: str | os.PathLike[str]) -> PipelineGridTable:
    """Завантажує попередньо обчислену таблицю конвеєра з JSON."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return PipelineGridTable.from_dict(data)


def build_fitness_evaluator(
    scenario_seed_fn: Callable[..., float] = synthetic_scenario_seed_q,
    scenarios: Sequence[str] = DEFAULT_TRAIN_SCENARIOS,
    seeds: Sequence[int] = DEFAULT_TRAIN_SEEDS,
    fixed_compute_time_s: float | None = 0.15,
    timing_noise_std: float = 0.0,
    rng: random.Random | None = None,
) -> Callable[[dict[str, int | float]], float]:
    """Створює функцію пристосованості як середнє Q по заданих сценаріях і seed каналу."""
    if not scenarios or not seeds:
        raise ValueError("Списки сценаріїв і seed не можуть бути порожніми")
    if FORBIDDEN_CHANNEL_SEED in set(seeds):
        raise ValueError(f"Відкритий seed {FORBIDDEN_CHANNEL_SEED} заборонено у фітнес-оцінці")

    eff_scenarios = tuple(scenarios)
    eff_seeds = tuple(seeds)
    for sc in eff_scenarios:
        if sc not in ("спокійний", "робочий", "аварійний"):
            raise ValueError(f"Невідомий сценарій каналу: {sc!r}")

    def _fitness(genes: dict[str, int | float]) -> float:
        total = 0.0
        if scenario_seed_fn is call_pipeline:
            ctx = get_default_pipeline_context()
            bundle = ctx.pack_and_decode(genes)
            for sc in eff_scenarios:
                for sd in eff_seeds:
                    m = ctx.evaluate_packed_on_channel(
                        bundle,
                        scenario=sc,
                        seed=sd,
                        fixed_compute_time_s=fixed_compute_time_s,
                        timing_noise_std=timing_noise_std,
                        rng=rng,
                    )
                    total += float(m["Q"])
        else:
            for sc in eff_scenarios:
                for sd in eff_seeds:
                    total += float(
                        scenario_seed_fn(
                            genes,
                            scenario=sc,
                            seed=sd,
                            fixed_compute_time_s=fixed_compute_time_s,
                            timing_noise_std=timing_noise_std,
                            rng=rng,
                        )
                    )
        return total / (len(eff_scenarios) * len(eff_seeds))

    return _fitness


def evaluate_train_and_val(
    genes: Mapping[str, int | float],
    scenario_seed_fn: Callable[..., float] = synthetic_scenario_seed_q,
    train_scenarios: Sequence[str] = DEFAULT_TRAIN_SCENARIOS,
    train_seeds: Sequence[int] = DEFAULT_TRAIN_SEEDS,
    val_scenarios: Sequence[str] = DEFAULT_VAL_SCENARIOS,
    val_seeds: Sequence[int] = DEFAULT_VAL_SEEDS,
    fixed_compute_time_s: float | None = 0.15,
) -> dict[str, float]:
    """Оцінює хромосому на train і val (усі 3 сценарії), рахуючи `gap_train_minus_val`."""
    validate_seed_split(train_seeds, val_seeds)
    genes_copy = dict(genes)

    if scenario_seed_fn is call_pipeline:
        ctx = get_default_pipeline_context()
        t_start = time.perf_counter()
        bundle = ctx.pack_and_decode(genes_copy)
        train_scores = [
            float(
                ctx.evaluate_packed_on_channel(
                    bundle, scenario=sc, seed=sd, fixed_compute_time_s=fixed_compute_time_s
                )["Q"]
            )
            for sc in train_scenarios
            for sd in train_seeds
        ]
        single_eval_time_s = time.perf_counter() - t_start
        train_q = float(sum(train_scores) / len(train_scores))

        result: dict[str, float] = {
            "train_q": train_q,
            "single_eval_time_s": single_eval_time_s,
        }
        val_scores: list[float] = []
        for sc in val_scenarios:
            sc_scores = [
                float(
                    ctx.evaluate_packed_on_channel(
                        bundle, scenario=sc, seed=sd, fixed_compute_time_s=fixed_compute_time_s
                    )["Q"]
                )
                for sd in val_seeds
            ]
            sc_q = float(sum(sc_scores) / len(sc_scores))
            result[f"val_q_{sc}"] = sc_q
            val_scores.append(sc_q)

        val_q = float(sum(val_scores) / len(val_scores))
        result["val_q"] = val_q
        result["gap_train_minus_val"] = train_q - val_q
        return result

    train_eval = build_fitness_evaluator(
        scenario_seed_fn=scenario_seed_fn,
        scenarios=train_scenarios,
        seeds=train_seeds,
        fixed_compute_time_s=fixed_compute_time_s,
        timing_noise_std=0.0,
    )
    train_q = float(train_eval(genes_copy))

    result = {"train_q": train_q}
    val_scores = []
    for sc in val_scenarios:
        sc_eval = build_fitness_evaluator(
            scenario_seed_fn=scenario_seed_fn,
            scenarios=(sc,),
            seeds=val_seeds,
            fixed_compute_time_s=fixed_compute_time_s,
            timing_noise_std=0.0,
        )
        sc_q = float(sc_eval(genes_copy))
        result[f"val_q_{sc}"] = sc_q
        val_scores.append(sc_q)

    val_q = float(sum(val_scores) / len(val_scores))
    result["val_q"] = val_q
    result["gap_train_minus_val"] = train_q - val_q
    return result


def extract_best_so_far_curve(
    history: Sequence[Mapping[str, object]],
) -> list[tuple[int, float]]:
    """Витягує монотонну криву `(evals_total, best_so_far)` із журналу ГА або Random Search."""
    curve: list[tuple[int, float]] = []
    running_best = float("-inf")
    for row in history:
        fit = float(row["best_fitness"])
        if fit > running_best:
            running_best = fit
        curve.append((int(row["evals_total"]), running_best))
    return curve
