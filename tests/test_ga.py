"""Модульні тести генетичного алгоритму та Random Search (`src/ga.py`, §5 design/ga.md)."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import pytest

from src.ga import (
    CROSSOVER_KINDS,
    DEFAULT_TRAIN_SEEDS,
    DEFAULT_VAL_SEEDS,
    FORBIDDEN_CHANNEL_SEED,
    GENE_BOUNDS,
    PipelineGridTable,
    build_fitness_evaluator,
    build_pipeline_grid_table,
    call_pipeline,
    crossover_chromosomes,
    evaluate_train_and_val,
    extract_best_so_far_curve,
    get_default_hyperparams,
    load_pipeline_grid_table,
    mutate_chromosome,
    optimize_with_ga,
    optimize_with_random_search,
    parse_bounds_spec,
    sample_random_chromosome,
    save_best_params_json,
    save_ga_history_csv,
    save_pipeline_grid_table,
    synthetic_fitness,
    synthetic_scenario_seed_q,
    tournament_select,
    validate_seed_split,
)


def _assert_in_bounds(genes: dict) -> None:
    for name, (lo, hi, kind) in GENE_BOUNDS.items():
        val = genes[name]
        if kind == "int":
            assert type(val) is int, f"{name} має бути int, отримано {type(val)}"
            assert int(lo) <= val <= int(hi)
        else:
            assert isinstance(val, float)
            assert float(lo) <= val <= float(hi)


def test_operators_stay_within_gene_bounds_and_types():
    """1. Усі 3 види кросоверу + мутація утримують гени в GENE_BOUNDS, block_size є int (300 пар)."""
    rng = random.Random(101)
    for _ in range(300):
        p1 = sample_random_chromosome(rng)
        p2 = sample_random_chromosome(rng)
        _assert_in_bounds(p1)
        _assert_in_bounds(p2)

        for kind in CROSSOVER_KINDS:
            child = crossover_chromosomes(p1, p2, rng, crossover_rate=1.0, kind=kind)
            _assert_in_bounds(child)
            mutated = mutate_chromosome(child, rng, mutation_rate=1.0, sigma_scale=0.35)
            _assert_in_bounds(mutated)
            # Гарантована зміна цілого гена при mutation_rate=1.0 (крок 0 -> ±1 з урахуванням меж)
            assert mutated["block_size"] != child["block_size"]


def test_elitism_monotonicity_d1_case16_and_no_elitism_control():
    """2. Кейс D1 №16: best_final >= best_initial (з елітою) + контрольний тест без еліти."""
    for seed in range(10):
        _, best_fit, history = optimize_with_ga(
            synthetic_fitness,
            budget=200,
            pop_size=20,
            elitism=2,
            seed=seed,
            return_history=True,
        )
        gen_bests = [float(row["best_fitness"]) for row in history]
        assert gen_bests[-1] >= gen_bests[0]
        assert best_fit == gen_bests[-1]
        for prev, curr in zip(gen_bests, gen_bests[1:]):
            assert curr >= prev

    # Контрольний тест: без еліти (elitism=0) і з сильною мутацією найкраще значення може падати
    saw_drop = False
    for seed in range(15):
        _, _, no_elite_hist = optimize_with_ga(
            synthetic_fitness,
            budget=200,
            pop_size=20,
            elitism=0,
            mutation_rate=0.95,
            sigma_scale=0.6,
            seed=seed,
            return_history=True,
        )
        gen_bests = [float(row["best_fitness"]) for row in no_elite_hist]
        if any(curr < prev for prev, curr in zip(gen_bests, gen_bests[1:])):
            saw_drop = True
            break
    assert saw_drop is True


def test_exact_evaluation_budget_across_modes():
    """3. Кількість викликів fitness_fn строго == budget при N = 1, 7, 10, 33, 200 у всіх режимах."""
    for n_budget in (1, 7, 10, 33, 200):
        # Стандартний ГА
        calls = 0

        def counting_fn(genes):
            nonlocal calls
            calls += 1
            return synthetic_fitness(genes)

        optimize_with_ga(counting_fn, budget=n_budget, pop_size=20, elitism=2, seed=4)
        assert calls == n_budget

        # ГА з переоцінкою еліти
        calls = 0
        optimize_with_ga(
            counting_fn, budget=n_budget, pop_size=20, elitism=2, reevaluate_elite=True, seed=4
        )
        assert calls == n_budget

        # ГА з усередненням eval_repeats=2
        calls = 0
        optimize_with_ga(
            counting_fn, budget=n_budget, pop_size=20, elitism=2, eval_repeats=2, seed=4
        )
        assert calls == n_budget

        # Random Search
        calls = 0
        optimize_with_random_search(counting_fn, budget=n_budget, seed=4)
        assert calls == n_budget


def test_reproducibility_with_fixed_seed():
    """4. Однаковий seed дає ідентичні результати та історію (для ГА і Random Search)."""
    ga_a = optimize_with_ga(synthetic_fitness, budget=200, seed=7, return_history=True)
    ga_b = optimize_with_ga(synthetic_fitness, budget=200, seed=7, return_history=True)
    assert ga_a == ga_b

    rs_a = optimize_with_random_search(synthetic_fitness, budget=200, seed=7, return_history=True)
    rs_b = optimize_with_random_search(synthetic_fitness, budget=200, seed=7, return_history=True)
    assert rs_a == rs_b


def test_edge_cases_zero_constant_fitness_small_budget_and_invalid_elitism():
    """5. Нульова/константна пристосованість, турнір при рівних значеннях, budget < pop, elitism >= pop."""
    # Константна нульова пристосованість -> без збоїв, повертає валідну хромосому з 0.0
    zero_genes, zero_fit, zero_hist = optimize_with_ga(
        lambda g: 0.0, budget=60, pop_size=20, elitism=2, seed=1, return_history=True
    )
    _assert_in_bounds(zero_genes)
    assert zero_fit == 0.0
    assert len(zero_hist) == 4  # 20 + 18 + 18 + 4

    # Турнір при однакових пристосованостях не падає й повертає одного з кандидатів
    rng = random.Random(3)
    pop = [sample_random_chromosome(rng) for _ in range(5)]
    picked = tournament_select(pop, [0.0] * 5, rng, k=3)
    assert picked in pop

    # Бюджет менший за популяцію -> оцінено рівно budget особин у поколінні 0
    g_small, f_small, h_small = optimize_with_ga(
        synthetic_fitness, budget=5, pop_size=20, elitism=2, seed=2, return_history=True
    )
    _assert_in_bounds(g_small)
    assert f_small > 0.0
    assert len(h_small) == 1 and h_small[0]["evals_total"] == 5

    # Некоректні elitism / budget -> ValueError
    with pytest.raises(ValueError):
        optimize_with_ga(synthetic_fitness, budget=20, pop_size=5, elitism=5)
    with pytest.raises(ValueError):
        optimize_with_ga(synthetic_fitness, budget=20, pop_size=5, elitism=6)
    with pytest.raises(ValueError):
        optimize_with_ga(synthetic_fitness, budget=0, pop_size=5, elitism=1)
    with pytest.raises(ValueError):
        optimize_with_random_search(synthetic_fitness, budget=0)


def test_seed_split_validation_rejects_overlap_and_425243():
    """6. Навчальні й валідаційні seed не перетинаються і не містять відкритий 425243."""
    validate_seed_split(DEFAULT_TRAIN_SEEDS, DEFAULT_VAL_SEEDS)

    with pytest.raises(ValueError):
        validate_seed_split((FORBIDDEN_CHANNEL_SEED, 12), (21, 22))
    with pytest.raises(ValueError):
        validate_seed_split((11, 12), (21, FORBIDDEN_CHANNEL_SEED))
    with pytest.raises(ValueError):
        validate_seed_split((11, 21), (21, 22))
    with pytest.raises(ValueError):
        validate_seed_split((), (21, 22))


def test_ga_converges_on_synthetic_landscape_and_saves_artifacts(tmp_path: Path):
    """7. ГА покращує пристосованість відносно початкової популяції та зберігає CSV/JSON."""
    defaults = get_default_hyperparams({"пакування": {"candidate_window_frames": 64}})
    assert defaults["block_size"] == 64

    improvements = []
    all_rows = []
    best_overall_genes = None
    best_overall_fit = float("-inf")
    best_seed = 0

    for seed in range(10):
        genes, fit, hist = optimize_with_ga(
            synthetic_fitness, budget=200, pop_size=20, elitism=2, seed=seed, return_history=True
        )
        improvements.append(float(hist[-1]["best_fitness"]) - float(hist[0]["best_fitness"]))
        all_rows.extend(hist)
        if fit > best_overall_fit:
            best_overall_fit = fit
            best_overall_genes = genes
            best_seed = seed

    assert min(improvements) > 0.0
    assert best_overall_fit > 0.84

    hist_csv = tmp_path / "ga_history.csv"
    best_json = tmp_path / "best_params.json"
    save_ga_history_csv(all_rows, hist_csv)
    save_best_params_json(best_overall_genes or {}, best_overall_fit, best_json, seed=best_seed)

    with open(hist_csv, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 10 * 11
    assert {"run_seed", "generation", "best_fitness", "mean_fitness", "evals_total"}.issubset(rows[0].keys())

    loaded_best = json.loads(best_json.read_text(encoding="utf-8"))
    assert loaded_best["best_fitness"] == pytest.approx(best_overall_fit)
    _assert_in_bounds(loaded_best["best_genes"])

    curve_ga = extract_best_so_far_curve(hist)
    assert curve_ga and curve_ga[0][0] == 20 and curve_ga[-1][0] == 200
    assert all(b >= a for (_, a), (_, b) in zip(curve_ga, curve_ga[1:]))


def test_scenario_evaluator_fixed_t_and_calm_invariance():
    """Перевіряє build_fitness_evaluator, evaluate_train_and_val та інваріант спокійного сценарію (§3.1, §3.3)."""
    genes_low_p = {"block_size": 36, "compression_threshold": 0.28, "priority_weight": 0.0}
    genes_high_p = {"block_size": 36, "compression_threshold": 0.28, "priority_weight": 6.5}

    # У спокійному сценарії R = 1 незалежно від priority_weight; у робочому/аварійному priority_weight впливає на Q
    q_calm_low = synthetic_scenario_seed_q(genes_low_p, scenario="спокійний", seed=21, fixed_compute_time_s=0.15)
    q_calm_high = synthetic_scenario_seed_q(genes_high_p, scenario="спокійний", seed=21, fixed_compute_time_s=0.15)
    assert q_calm_low == pytest.approx(q_calm_high)

    q_emerg_low = synthetic_scenario_seed_q(genes_low_p, scenario="аварійний", seed=21, fixed_compute_time_s=0.15)
    q_emerg_high = synthetic_scenario_seed_q(genes_high_p, scenario="аварійний", seed=21, fixed_compute_time_s=0.15)
    assert q_emerg_high > q_emerg_low

    # Заборонений seed 425243 відхиляється навіть у build_fitness_evaluator
    with pytest.raises(ValueError):
        build_fitness_evaluator(seeds=(11, FORBIDDEN_CHANNEL_SEED))

    train_eval = build_fitness_evaluator(fixed_compute_time_s=0.15)
    best_genes, _, hist = optimize_with_ga(train_eval, budget=200, seed=0, return_history=True)
    summary = evaluate_train_and_val(best_genes, fixed_compute_time_s=0.15)
    assert {"train_q", "val_q", "gap_train_minus_val", "val_q_спокійний", "val_q_робочий", "val_q_аварійний"} <= summary.keys()
    assert summary["val_q"] > 0.70
    assert extract_best_so_far_curve(hist)[-1][0] == 200


def test_real_pipeline_adapter_and_grid_table_integration(tmp_path: Path):
    """8. Інтеграція call_pipeline, parse_bounds_spec і 3D-сітки реального конвеєра (§3.1–3.5)."""
    assert parse_bounds_spec(None) == GENE_BOUNDS
    partial = parse_bounds_spec("block_size=20:48:int")
    assert partial["block_size"] == (20, 48, "int")
    assert partial["compression_threshold"] == GENE_BOUNDS["compression_threshold"]
    with pytest.raises(ValueError):
        parse_bounds_spec("bad_format")

    defaults = get_default_hyperparams(
        {"пакування": {"candidate_window_frames": 64}, "пріоритети_кадрів": {"critical_бонус": 5}}
    )
    assert defaults == {"block_size": 64, "compression_threshold": 0.0, "priority_weight": 5.0}

    # Живий конвеєр працює на всіх 3 сценаріях і забороняє відкритий seed 425243
    for sc in ("спокійний", "робочий", "аварійний"):
        q_val = call_pipeline(defaults, scenario=sc, seed=21, fixed_compute_time_s=0.15)
        assert 0.0 < q_val <= 1.0
    with pytest.raises(ValueError):
        call_pipeline(defaults, scenario="аварійний", seed=FORBIDDEN_CHANNEL_SEED)

    # Перевірка чутливості на реальному конвеєрі: при виправленій черзі block_size=64 концентрує
    # критичні кадри у 47 пакетах і суттєво перевершує block_size=36 в аварійному сценарії
    q_b36 = call_pipeline(
        {"block_size": 36, "compression_threshold": 0.0, "priority_weight": 5.0},
        scenario="аварійний",
        seed=11,
        fixed_compute_time_s=0.15,
    )
    q_b64 = call_pipeline(defaults, scenario="аварійний", seed=11, fixed_compute_time_s=0.15)
    assert q_b64 > q_b36
    assert abs(q_b64 - q_b36) > 0.05

    # Однократна пакетна оптимізація в build_fitness_evaluator(call_pipeline)
    pipe_train_eval = build_fitness_evaluator(
        scenario_seed_fn=call_pipeline,
        scenarios=("робочий", "аварійний"),
        seeds=(11, 12),
        fixed_compute_time_s=None,
    )
    assert 0.0 < pipe_train_eval(defaults) <= 1.0

    # Міні-сітка реального конвеєра: збереження/завантаження та ГА vs Random з рівним бюджетом
    grid = build_pipeline_grid_table(
        block_sizes=(24, 48),
        compression_thresholds=(0.0, 0.5),
        priority_weights=(0.0, 5.0),
        scenarios=("спокійний", "робочий", "аварійний"),
        seeds=(11, 12, 21, 22, 23, 24, 25),
        noise_samples=0,
    )
    table_path = tmp_path / "mini_grid.json"
    save_pipeline_grid_table(grid, table_path)
    loaded = load_pipeline_grid_table(table_path)
    assert isinstance(loaded, PipelineGridTable)

    table_eval = build_fitness_evaluator(
        scenario_seed_fn=loaded.scenario_seed_q,
        scenarios=("робочий", "аварійний"),
        seeds=(11, 12),
        fixed_compute_time_s=None,
    )
    ga_calls = 0

    def ga_counting_fn(genes):
        nonlocal ga_calls
        ga_calls += 1
        return table_eval(genes)

    ga_genes, ga_fit, ga_hist = optimize_with_ga(
        ga_counting_fn, budget=40, pop_size=10, elitism=2, seed=0, return_history=True
    )
    assert ga_calls == 40
    assert ga_hist[-1]["best_fitness"] >= ga_hist[0]["best_fitness"]
    assert ga_fit == ga_hist[-1]["best_fitness"]
    _assert_in_bounds(ga_genes)

    rs_calls = 0

    def rs_counting_fn(genes):
        nonlocal rs_calls
        rs_calls += 1
        return table_eval(genes)

    rs_genes, _, _ = optimize_with_random_search(rs_counting_fn, budget=40, seed=0, return_history=True)
    assert rs_calls == 40
    _assert_in_bounds(rs_genes)
