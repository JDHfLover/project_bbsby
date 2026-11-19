# Проєкт 1 — Захищений канал телеметрії з обмеженою пропускною здатністю (`TLM1`, команда `T01`)

Наскрізна система захищеного передавання телеметрії БПЛА через вузький канал із пачковими втратами та бюджетом часу передавання (`FORMAT_SPEC_1`, `CHANNEL_SPEC_1`).

## Архітектура конвеєра

```text
team/session.bin (3000 кадрів × 12 Б = 36 000 Б)
  └─> [1] src/telemetry.py       — валідація 12-байтових кадрів (>IhhhBB), потокове читання, атомарний запис
  └─> [2] src/classification.py  — самописний MLP (7→32 ReLU→3 softmax, causal-вікно w=5, macro-F1 ≈ 0.923)
  └─> [3] src/huffman.py         — канонічний Гаффман ≤ 15 біт (package-merge, Крафт K=1, MSB-first)
          (+ src/arithmetic.py   — експертний 32-бітний арифметичний кодер E1/E2/E3 для порівняння)
  └─> [4] src/dp.py              — рюкзак 0/1 ДП O(n·W), Greedy, Brute-force, NO_COMPRESSION, жадібний планувальник
  └─> [5] src/crypto.py          — PBKDF2-HMAC-SHA256 (100k, 1 раз/сеанс), AES-256-CTR (i=1), HMAC-SHA256 (EtM)
  └─> [6] team/channel.py        — еталонна двостанна модель каналу Маркова (не змінюється)
  └─> [7] src/receiver.py        — перевірка заголовка → HMAC у const-time до decrypt → E_SHORT..E_CODE → дедуплікація
  └─> [8] src/channel_runner.py  — наскрізний прогін run_end_to_end / run_all_scenarios, метрика Q = 0.5S + 0.35R + 0.15T
  └─> [9] src/ga.py              — генетичний алгоритм (3 гени, турнір k=3, елітизм) vs Random Search (бюджет 200)
```

---

## Вимоги

- **Python 3.11+** (перевірено на Python 3.12 і 3.14)
- **Git**

Усі залежності (`cryptography`, `numpy`, `pytest`, `pytest-cov`, `ruff`) описані в [requirements.txt](requirements.txt).

---

## Встановлення та запуск на Linux / macOS

Усі команди виконуються з кореневої папки проєкту (`project_bbsby`).

### 1. Створення віртуального середовища та встановлення залежностей

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Перевірка коду лінтером і запуск усіх тестів

```bash
# Статичний аналіз коду (ruff)
ruff check .

# Усі 74 автотести з коротким виводом
pytest -q

# Усі 74 автотести зі звітом покриття коду
pytest --cov=src --cov-report=term-missing

# Запуск окремого тестового файлу
pytest tests/test_channel_integration.py -v
```

*(Без активації `.venv` можна запускати напряму: `.venv/bin/python -m pytest --cov=src --cov-report=term-missing`)*

### 3. Запуск демонстрації та відтворення всіх експериментів

```bash
# Перевірка офіційних еталонів ref_01..06, bad_01..05 та базовий прогін 3 сценаріїв
python scripts/run_checkpoint1.py

# Відтворення всіх 13 таблиць звіту у results/ однією командою (~1 хв)
python experiments/run_all.py --skip-train

# Повний прогін із перенавчанням MLP та перебудовою 3D-сітки ГА
python experiments/run_all.py --rebuild-ga-table
```

### 4. Повна перевірка проєкту однією командою (Linux / macOS)

```bash
ruff check . && pytest --cov=src --cov-report=term-missing && python scripts/run_checkpoint1.py && python experiments/run_all.py --skip-train
```

### 5. Перевизначення пароля сеансу через змінну середовища (необов'язково)

За замовчуванням пароль береться з `team/team_config.json`, але його можна перевизначити:

```bash
export TLM1_SESSION_PASSWORD="your_password_here"
```

---

## Встановлення та запуск на Windows

Усі команди виконуються з кореневої папки проєкту (`project_bbsby`). Рекомендується **PowerShell** (або **Command Prompt / CMD**).

### Варіант A: Windows PowerShell (рекомендовано)

#### 1. Створення віртуального середовища та встановлення залежностей

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Якщо PowerShell забороняє запуск скрипта активації (`ExecutionPolicy`), дозвольте його лише для поточного вікна термінала:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\.venv\Scripts\Activate.ps1
> ```

> Щоб українські назви сценаріїв (`спокійний`, `робочий`, `аварійний`) відображалися коректно в консолі Windows, увімкніть UTF-8:
> ```powershell
> [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
> $env:PYTHONUTF8 = "1"
> ```

#### 2. Перевірка коду лінтером і запуск тестів

```powershell
# Статичний аналіз коду (ruff)
ruff check .

# Усі 74 автотести з коротким виводом
pytest -q

# Усі 74 автотести зі звітом покриття коду (як у CI)
pytest --cov=src --cov-report=term-missing

# Запуск окремого тестового файлу
pytest tests/test_channel_integration.py -v
```

*(Без активації `.venv` можна викликати напряму: `.\.venv\Scripts\python.exe -m pytest --cov=src --cov-report=term-missing`)*

#### 3. Запуск демонстрації та відтворення всіх експериментів

```powershell
# Перевірка еталонів ref_01..06, bad_01..05 та базовий прогін
python scripts/run_checkpoint1.py

# Відтворення всіх 13 таблиць звіту у results/ однією командою (~1 хв)
python experiments/run_all.py --skip-train

# Повний прогін із перенавчанням MLP та перебудовою 3D-сітки ГА
python experiments/run_all.py --rebuild-ga-table
```

#### 4. Повна перевірка проєкту однією командою (PowerShell)

```powershell
ruff check .; if ($?) { pytest --cov=src --cov-report=term-missing }; if ($?) { python scripts/run_checkpoint1.py }; if ($?) { python experiments/run_all.py --skip-train }
```

#### 5. Перевизначення пароля сеансу в PowerShell (необов'язково)

```powershell
$env:TLM1_SESSION_PASSWORD = "your_password_here"
```

---

### Варіант B: Windows Command Prompt (CMD)

```bat
py -3 -m venv .venv
.venv\Scripts\activate.bat
chcp 65001
set PYTHONUTF8=1

python -m pip install --upgrade pip
pip install -r requirements.txt

ruff check . && pytest --cov=src --cov-report=term-missing && python scripts\run_checkpoint1.py && python experiments\run_all.py --skip-train
```

Перевизначення пароля в CMD (необов'язково):
```bat
set TLM1_SESSION_PASSWORD=your_password_here
```

---

## Як користуватися проєктом із Python-коду

Приклад наскрізного запуску конвеєра (`TX → Channel → RX → оцінка Q`) для одного або всіх трьох сценаріїв:

```python
from src.channel_runner import load_team_config, run_all_scenarios, run_end_to_end
from src.telemetry import load_session_bin

cfg = load_team_config()                      # team/team_config.json
frames = load_session_bin("team/session.bin") # 3000 кадрів телеметрії

# Прогін одного сценарію ("спокійний", "робочий" або "аварійний")
res = run_end_to_end(frames, config=cfg, scenario_name="аварійний", seed=21)
print(
    f"S={res['S']:.4f}, R={res['R']:.4f}, T={res['T']:.4f}, "
    f"crit_loss={res['crit_loss_rate']:.4f}, Q={res['Q']:.4f}"
)

# Прогін усіх трьох сценаріїв із перевіркою порогів та ліміту часу (< 120 с)
summary = run_all_scenarios(frames, config=cfg, seed=21)
print("Пороги пройдені:", summary["all_thresholds_passed"])
print("Час прогону (с):", round(summary["total_wall_time_s"], 2))
```

Для запуску з оптимізованими параметрами ГА передайте словник `best_genes` із [results/best_params.json](results/best_params.json) у параметр `hyperparams=...`.

---

## Окремі скрипти експериментів і таблиці звіту

| Команда | Призначення | Згенеровані артефакти |
|---|---|---|
| `python experiments/train_classifier.py` | Навчання самописного MLP (10 ініціалізацій, 80/20 блоковий спліт із розривом `w-1`) | `models/classifier_v1.npz`, `models/classifier_v1.json`, `results/classifier_metrics.json`, `results/confusion_matrix.csv` |
| `python experiments/run_dp_vs_greedy.py` | Порівняння ДП-рюкзака та жадібного алгоритму (цінність, час за ≥5 прогонів, похибка, 3 сценарії) | `results/dp_vs_greedy.csv` |
| `python experiments/run_crypto_order.py` | Порівняння `compress -> encrypt` проти `encrypt -> compress` | `results/crypto_order_comparison.csv` |
| `python experiments/run_compression_comparison_fixed.py` | Експертне порівняння канонічного Гаффмана та 32-бітного арифметичного кодера | `results/huffman_vs_arithmetic_v2.csv` |
| `python experiments/run_ga_study.py` | 10 запусків ГА проти 10 запусків Random Search (`seed 0..9`, бюджет 200, валідація на `seeds 21..25`) | `results/ga_history.csv`, `results/best_params.json`, `results/ga_study_compare.csv`, `results/ga_study_curves.csv`, `results/ga_study_sensitivity_summary.csv` |
| `python experiments/run_ablation.py` | Поведінка у 3 сценаріях каналу, ablation кожного блоку та зведена таблиця метрик | `results/channel_scenarios.csv`, `results/ablation.csv`, `results/final_metrics.json` |
| `python experiments/run_all.py --skip-train` | Єдина команда відтворення всіх таблиць звіту | Усі 13 підсумкових файлів у `results/` |

---

## Структура репозиторію

- `src/` — основні модулі конвеєра (`telemetry.py`, `classification.py`, `huffman.py`, `arithmetic.py`, `dp.py`, `packet_format.py`, `crypto.py`, `receiver.py`, `channel_runner.py`, `ga.py`).
- `team/` — незмінний офіційний комплект команди `T01` (`team_config.json`, `channel.py`, `session.bin`, `session.csv`, `train.csv`, `valid.csv`, `reference/`).
- `models/` — збережені ваги навченого MLP (`classifier_v1.npz`, `classifier_v1.json`).
- `tests/` — 74 модульні, граничні, негативні, криптографічні та наскрізні `pytest`-тести (покриття `src/` — ~90%, включно з офіційними `team/reference/ref_01..06` і `bad_01..05`).
- `experiments/` — скрипти відтворюваних експериментів та журнал запусків [experiments/log.md](experiments/log.md).
- `results/` — машинно згенеровані таблиці CSV/JSON для звіту.
- `scripts/` — демонстраційний скрипт перевірки еталонів [scripts/run_checkpoint1.py](scripts/run_checkpoint1.py) та генератор власних тестових даних [scripts/verify_custom_inputs.py](scripts/verify_custom_inputs.py).
- `docs/` — журнал інженерних рішень [docs/decisions.md](docs/decisions.md) та журнал помилок [docs/bugs.md](docs/bugs.md).
- [META-USAGE.md](META-USAGE.md) — декларація використання інструментів ШІ.
- [pyproject.toml](pyproject.toml) — конфігурація `ruff` і `pytest`.
- [.github/workflows/ci.yml](.github/workflows/ci.yml) — автоматичний CI (`ruff check` + `pytest --cov=src`).

---

## Часті проблеми

- **`python` не знайдено на Windows:** використовуйте `py -3` для створення `.venv` (`py -3 -m venv .venv`). Після активації `.venv` команда `python` працюватиме автоматично.
- **`python` не знайдено на Linux:** використовуйте `python3 -m venv .venv`, а після `source .venv/bin/activate` команда `python` працюватиме автоматично.
- **Помилка `ruff` або `--cov` не розпізнано:** переконайтеся, що віртуальне середовище активоване й виконано `pip install -r requirements.txt`.
- **Кракозябри замість українського тексту на Windows:** перед запуском виконайте `chcp 65001` (CMD) або `$env:PYTHONUTF8 = "1"` (PowerShell).

---

## Відомі обмеження

- Основний формат пакетів відповідає версії 1 специфікації `FORMAT_SPEC_1.md` (`flags & 0x03 == 0`); командні розширення `PREDICTION` (`0x01`) та `DUPLICATION` (`0x02`) свідомо відхиляються приймачем із кодом `E_PAYLOAD` задля крос-сумісності.
- Арифметичне кодування (`src/arithmetic.py`) реалізовано як окреме експертне розширення для кількісного порівняння з Гаффманом і не змінює базовий двійковий формат `TLM1`.
- Пароль сеансу зчитується через змінну середовища `TLM1_SESSION_PASSWORD` або з `team/team_config.json` (`resolve_session_password`) і не потрапляє у логи чи відкриту частину пакета.
