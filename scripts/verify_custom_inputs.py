#!/usr/bin/env python3
"""
Генератор і перевірник розширеного набору власних вхідних даних (`sample_inputs/`)
для тестування конвеєра TLM1 на різних масштабах (від 0 до 6000 кадрів) і крайових випадках.

Запуск:
- Перевірити наявні файли (усереднення за 5 seeds 21..25):
    python scripts/verify_custom_inputs.py
- Перегенерувати всі файли (можна задати свій розмір великої сесії `--frames 5000`):
    python scripts/verify_custom_inputs.py --generate --frames 3000 --seeds 21,22,23,24,25
"""
from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.channel_runner import (
    evaluate_metrics,
    load_team_config,
    run_all_scenarios,
    transmit_packets,
)
from src.classification import classify_frames, compute_frame_values, macro_f1
from src.crypto import resolve_session_password
from src.dp import pack_telemetry_stream
from src.receiver import receive_stream
from src.telemetry import (
    ProtocolError,
    TelemetryFrame,
    load_frames_csv,
    load_session_bin,
    save_session_bin,
)

DEFAULT_DIR = REPO_ROOT / "sample_inputs"
FIXED_SALT = bytes.fromhex("000102030405060708090A0B0C0D0E0F")
FIXED_PREFIX = b"CUST0001"


def _save_csv(path: Path, frames: list[TelemetryFrame], labels: list[int] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        if labels is not None:
            writer.writerow(
                ["timestamp_ms", "altitude_dm", "temp_c10", "vibration", "mode", "critical", "label"]
            )
            for f, lbl in zip(frames, labels):
                writer.writerow(
                    [f.timestamp_ms, f.altitude_dm, f.temp_c10, f.vibration, f.mode, f.critical, lbl]
                )
        else:
            writer.writerow(["timestamp_ms", "altitude_dm", "temp_c10", "vibration", "mode", "critical"])
            for f in frames:
                writer.writerow([f.timestamp_ms, f.altitude_dm, f.temp_c10, f.vibration, f.mode, f.critical])


def _make_uav_mission(
    n_frames: int,
    *,
    seed: int = 2026,
    frequent_switches: bool = False,
) -> tuple[list[TelemetryFrame], list[int]]:
    """
    Генерує реалістичну місію БПЛА довільної довжини `n_frames` (~20 Гц) з усіма 3 класами:
    - клас 0 (норма): стабільна висота/температура, мала вібрація;
    - клас 1 (аномалія): різкі сплески вібрації (`|vib| ~ 120..420`);
    - клас 2 (перехідний): інтенсивний набір/зниження висоти (`|Δalt| ~ 4..8`) та зміна `mode`.
    """
    rng = random.Random(seed)
    frames: list[TelemetryFrame] = []
    labels: list[int] = []

    # Розбиваємо місію на сегменти (режим, довжина, напрямок висоти)
    segments: list[tuple[int, int, int]] = []
    remaining = n_frames
    phase_idx = 0
    while remaining > 0:
        if frequent_switches:
            pattern = (0, 2, 1, 0, 2, 0, 1)
            seg_cls = pattern[phase_idx % len(pattern)]
            seg_len = rng.randint(25, 55) if seg_cls != 0 else rng.randint(45, 90)
        else:
            pattern = (0, 2, 1, 0, 2, 0, 1, 0)
            seg_cls = pattern[phase_idx % len(pattern)]
            seg_len = rng.randint(50, 90) if seg_cls != 0 else rng.randint(150, 260)
        seg_len = min(seg_len, remaining)
        alt_dir = 1 if (phase_idx % 2 == 0) else -1
        segments.append((seg_cls, seg_len, alt_dir))
        remaining -= seg_len
        phase_idx += 1

    ts = 50
    alt = 1150
    temp = 205
    mode = 2
    global_idx = 0

    for seg_cls, seg_len, alt_dir in segments:
        # Утримуємо висоту в реалістичних межах 700..1800 дм
        if seg_cls == 2:
            if alt > 1550:
                alt_dir = -1
            elif alt < 850:
                alt_dir = 1

        for step in range(seg_len):
            ts += rng.randint(46, 54)
            if seg_cls == 0:
                alt += rng.choice((-1, 0, 0, 0, 1))
                temp += rng.choice((-1, 0, 0, 0, 0, 1))
                vib = rng.randint(-15, 15)
                crit = 1 if (global_idx % 12 == 0) else 0
            elif seg_cls == 2:
                alt += alt_dir * rng.randint(4, 7)
                temp += rng.choice((-1, 0, 0, 1))
                vib = rng.randint(-28, 28)
                if step in (5, seg_len // 2):
                    mode = (mode + 1 + rng.randint(0, 2)) % 8
                crit = 1 if (global_idx % 10 == 0) else 0
            else:
                alt += rng.randint(-3, 3)
                temp += rng.randint(-2, 2)
                vib = rng.choice((-1, 1)) * rng.randint(120, 400)
                crit = 1 if (global_idx % 6 == 0) else 0

            alt = max(-32768, min(32767, alt))
            temp = max(-32768, min(32767, temp))
            frames.append(TelemetryFrame(ts, alt, temp, vib, mode, crit))
            labels.append(seg_cls)
            global_idx += 1

    return frames, labels


def generate_sample_inputs(out_dir: Path, large_frames: int = 3000) -> None:
    """Створює розширений набір коректних і негативних тестових файлів у `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Короткий реалістичний політ (600 кадрів, усі 3 класи + мітки)
    short_frames, short_labels = _make_uav_mission(600, seed=2026, frequent_switches=False)
    save_session_bin(out_dir / "realistic_flight.bin", short_frames)
    _save_csv(out_dir / "realistic_flight.csv", short_frames, short_labels)

    # 2. Повномасштабний політ (за замовчуванням 3000 кадрів = масштаб team/session.bin)
    large_mission, large_labels = _make_uav_mission(large_frames, seed=2027, frequent_switches=False)
    save_session_bin(out_dir / "realistic_flight_large.bin", large_mission)
    _save_csv(out_dir / "realistic_flight_large.csv", large_mission, large_labels)

    # 3. Довга стрес-місія (6000 кадрів = 72 КБ, удвічі більша за офіційну сесію)
    long_mission, long_labels = _make_uav_mission(6000, seed=2028, frequent_switches=False)
    save_session_bin(out_dir / "long_mission_6000.bin", long_mission)
    _save_csv(out_dir / "long_mission_6000.csv", long_mission, long_labels)

    # 4. Часті перемикання режимів (1000 кадрів — складніший тест класифікатора й планувальника)
    noisy_frames, noisy_labels = _make_uav_mission(1000, seed=2029, frequent_switches=True)
    save_session_bin(out_dir / "noisy_transitions.bin", noisy_frames)
    _save_csv(out_dir / "noisy_transitions.csv", noisy_frames, noisy_labels)

    # 5. Один критичний кадр (крайній випадок C1 №1)
    one_frame = [TelemetryFrame(100, 1500, 225, -12, 3, 1)]
    save_session_bin(out_dir / "single_frame.bin", one_frame)
    _save_csv(out_dir / "single_frame.csv", one_frame, [0])

    # 6. Порожній потік (крайній випадок C1 №2)
    (out_dir / "empty.bin").write_bytes(b"")

    # 7. Низькоентропійний стабільний потік (128 кадрів — сильне стиснення Гаффмана)
    const_frames = [
        TelemetryFrame(1000 + i * 50, 1000, 200, 0, 1, 1 if i % 16 == 0 else 0) for i in range(128)
    ]
    save_session_bin(out_dir / "low_entropy_stable.bin", const_frames)
    _save_csv(out_dir / "low_entropy_stable.csv", const_frames)

    # 8. Високоентропійний стрес-потік із граничними int16 (200 кадрів)
    rng = random.Random(777)
    ts = 10
    high_ent_frames: list[TelemetryFrame] = []
    for i in range(200):
        ts += rng.randint(1, 5000)
        if i == 0:
            alt, temp, vib = -32768, 32767, -32768
        elif i == 1:
            alt, temp, vib = 32767, -32768, 32767
        else:
            alt = rng.randint(-32768, 32767)
            temp = rng.randint(-32768, 32767)
            vib = rng.randint(-32768, 32767)
        high_ent_frames.append(
            TelemetryFrame(ts, alt, temp, vib, rng.randint(0, 7), 1 if i % 10 == 0 else 0)
        )
    save_session_bin(out_dir / "high_entropy_stress.bin", high_ent_frames)

    # 9. Потік, де 100% кадрів критичні (300 кадрів)
    rng_em = random.Random(911)
    emerg_frames = [
        TelemetryFrame(
            50 * (i + 1),
            1100 + rng_em.randint(-8, 8),
            230 + rng_em.randint(-3, 3),
            rng_em.choice((-1, 1)) * rng_em.randint(140, 420),
            5,
            1,
        )
        for i in range(300)
    ]
    save_session_bin(out_dir / "all_critical_stress.bin", emerg_frames)
    _save_csv(out_dir / "all_critical_stress.csv", emerg_frames)

    # 10. Негативні файли (мають відхилятися з ProtocolError)
    (out_dir / "invalid_truncated.bin").write_bytes(b"\x00" * 17)
    (out_dir / "invalid_bad_values.csv").write_text(
        "timestamp_ms,altitude_dm,temp_c10,vibration,mode,critical\n"
        "100,1000,200,5,9,0\n"
        "200,1000,200,5,1,2\n",
        encoding="utf-8",
    )
    (out_dir / "invalid_non_monotonic.csv").write_text(
        "timestamp_ms,altitude_dm,temp_c10,vibration,mode,critical\n"
        "500,1000,200,5,1,0\n"
        "450,1000,200,5,1,0\n",
        encoding="utf-8",
    )

    print(f"[OK] Згенеровано розширений набір даних у {out_dir}")


def _evaluate_multi_seed(
    frames: list[TelemetryFrame],
    cfg: dict,
    password: str,
    seeds: tuple[int, ...],
) -> tuple[int, float, dict[str, dict]]:
    """Пакує потік один раз і оцінює по всіх сценаріях та `seeds` (для швидкого й точного усереднення)."""
    t_tx0 = time.perf_counter()
    if not frames:
        packets: list[bytes] = []
    else:
        classes = classify_frames(frames)
        values = compute_frame_values(frames, classes, cfg)
        packets = pack_telemetry_stream(
            frames,
            values=values,
            config=cfg,
            password=password,
            salt=FIXED_SALT,
            session_nonce_prefix=FIXED_PREFIX,
            schedule=True,
            seal=True,
        )
    tx_time_s = time.perf_counter() - t_tx0

    thresholds = cfg.get("пороги_втрати_критичних_кадрів", {})
    out: dict[str, dict] = {}
    for sc_name, sc_cfg in cfg["канал"]["сценарії"].items():
        s_l, r_l, t_l, q_l, loss_l, ob_l = [], [], [], [], [], []
        for idx, sd in enumerate(seeds):
            delivered, ch_stats = transmit_packets(packets, sc_cfg, seed=sd)
            t_rx0 = time.perf_counter()
            recovered, _ = receive_stream(delivered, password, deduplicate=True, sort_by_timestamp=True)
            rx_time_s = time.perf_counter() - t_rx0
            if sc_name == "спокійний" and idx == 0:
                if int(ch_stats["не_вміщено_в_бюджет"]) == 0:
                    assert recovered == frames, "У спокійному сценарії порушено побайтове відновлення"
                else:
                    orig_bytes = {f.to_bytes() for f in frames}
                    assert all(f.to_bytes() in orig_bytes for f in recovered), (
                        "Відновлено пошкоджені кадри при перевищенні бюджету спокійного каналу"
                    )
            m = evaluate_metrics(
                frames,
                recovered,
                int(ch_stats["bytes_sent"]),
                tx_time_s + rx_time_s,
                cfg["метрика_Q"],
            )
            s_l.append(float(m["S"]))
            r_l.append(float(m["R"]))
            t_l.append(float(m["T"]))
            q_l.append(float(m["Q"]))
            loss_l.append(float(m["crit_loss_rate"]))
            ob_l.append(int(ch_stats["не_вміщено_в_бюджет"]))

        thr = float(thresholds.get(sc_name, 1.0))
        loss_mean = statistics.mean(loss_l)
        out[sc_name] = {
            "S": statistics.mean(s_l),
            "R": statistics.mean(r_l),
            "T": statistics.mean(t_l),
            "Q": statistics.mean(q_l),
            "loss": loss_mean,
            "over_budget": statistics.mean(ob_l),
            "passes_thr": loss_mean <= thr + 1e-12,
        }
    return len(packets), tx_time_s, out


def verify_sample_inputs(data_dir: Path, seeds: tuple[int, ...]) -> int:
    """Проганяє всі файли з `data_dir` через повний конвеєр TLM1."""
    cfg = load_team_config()
    password = resolve_session_password(cfg)

    valid_cases: list[tuple[str, str | None]] = [
        ("realistic_flight.bin", "realistic_flight.csv"),
        ("realistic_flight_large.bin", "realistic_flight_large.csv"),
        ("long_mission_6000.bin", "long_mission_6000.csv"),
        ("noisy_transitions.bin", "noisy_transitions.csv"),
        ("single_frame.bin", "single_frame.csv"),
        ("empty.bin", None),
        ("low_entropy_stable.bin", "low_entropy_stable.csv"),
        ("high_entropy_stress.bin", None),
        ("all_critical_stress.bin", "all_critical_stress.csv"),
    ]

    seeds_str = ",".join(str(s) for s in seeds)
    print(f"\n=== Перевірка коректних вхідних наборів ({data_dir}, seeds=[{seeds_str}]) ===")
    for bin_name, csv_name in valid_cases:
        bin_path = data_dir / bin_name
        frames = load_session_bin(bin_path)

        extra_info = ""
        if csv_name is not None:
            csv_path = data_dir / csv_name
            has_label = "label" in csv_path.read_text(encoding="utf-8").splitlines()[0]
            csv_frames, labels = load_frames_csv(csv_path, with_labels=has_label)
            assert csv_frames == frames, f"Розбіжність між {bin_name} та {csv_name}"
            if labels is not None and len(frames) >= 10:
                preds = classify_frames(frames)
                f1, per_cls = macro_f1(labels, preds)
                extra_info = (
                    f" | MLP macro-F1={f1:.4f} "
                    f"(0:{per_cls[0]:.2f}, 1:{per_cls[1]:.2f}, 2:{per_cls[2]:.2f})"
                )

        n_pkts, tx_s, sc = _evaluate_multi_seed(frames, cfg, password, seeds)
        calm, work, emerg = sc["спокійний"], sc["робочий"], sc["аварійний"]
        print(
            f"[PASS] {bin_name:26s} ({len(frames):4d} кадр., {sum(f.critical for f in frames):3d} crit, "
            f"pkts={n_pkts:3d}, tx={tx_s:.2f}s){extra_info}\n"
            f"       спокійний: Q={calm['Q']:.4f} (S={calm['S']:.3f}, R={calm['R']:.3f}, loss={calm['loss']:.3f}, ob={calm['over_budget']:.1f})\n"
            f"       робочий:   Q={work['Q']:.4f} (S={work['S']:.3f}, R={work['R']:.3f}, loss={work['loss']:.3f}, thr_ok={work['passes_thr']})\n"
            f"       аварійний: Q={emerg['Q']:.4f} (S={emerg['S']:.3f}, R={emerg['R']:.3f}, loss={emerg['loss']:.3f}, ob={emerg['over_budget']:.1f}, thr_ok={emerg['passes_thr']})"
        )

    # Окрема пряма перевірка публічного run_all_scenarios (ліміт 120 с задано для 3000 кадрів)
    large_frames = load_session_bin(data_dir / "realistic_flight_large.bin")
    e2e = run_all_scenarios(large_frames, config=cfg, seed=seeds[0])
    if len(large_frames) <= 3000:
        assert e2e["passes_time_limit"], "Перевищено ліміт часу run_all_scenarios"
    status = "PASS" if e2e["passes_time_limit"] else "INFO"
    print(
        f"\n[{status}] run_all_scenarios(realistic_flight_large.bin, {len(large_frames)} кадр.): "
        f"{e2e['total_wall_time_s']:.2f} с (ліміт конфігу для 3000 кадрів: {e2e['time_limit_s']:.0f} с)"
    )

    print("\n=== Перевірка негативних вхідних файлів (мають бути відхилені) ===")
    negative_checks = [
        (data_dir / "invalid_truncated.bin", lambda p: load_session_bin(p)),
        (data_dir / "invalid_bad_values.csv", lambda p: load_frames_csv(p, with_labels=False)),
        (data_dir / "invalid_non_monotonic.csv", lambda p: load_frames_csv(p, with_labels=False)),
    ]
    for path, loader in negative_checks:
        try:
            loader(path)
        except ProtocolError as err:
            print(f"[PASS] {path.name:26s} -> відхилено коректно ({err.code}: {err.message})")
        else:
            raise AssertionError(f"{path.name} мав викликати ProtocolError")

    print("\nУсі перевірки на розширеному власному наборі даних пройшли успішно!")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=str(DEFAULT_DIR), help="Папка з тестовими даними")
    ap.add_argument(
        "--generate", action="store_true", help="Згенерувати/оновити набір файлів перед перевіркою"
    )
    ap.add_argument(
        "--frames",
        type=int,
        default=3000,
        help="Кількість кадрів у realistic_flight_large.bin (за замовчуванням 3000)",
    )
    ap.add_argument(
        "--seeds",
        default="21,22,23,24,25",
        help="Список seed каналу через кому (за замовчуванням 21,22,23,24,25)",
    )
    args = ap.parse_args()

    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    if not seeds:
        raise ValueError("Потрібно вказати хоча б один seed")

    target_dir = Path(args.dir)
    if args.generate or not (target_dir / "long_mission_6000.bin").exists():
        generate_sample_inputs(target_dir, large_frames=args.frames)

    return verify_sample_inputs(target_dir, seeds=seeds)


if __name__ == "__main__":
    raise SystemExit(main())
