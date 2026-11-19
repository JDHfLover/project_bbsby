# Журнал помилок (`docs/bugs.md`)

> Кожен суттєвий баг супроводжується мінімальним відтворенням, аналізом причини, виправленням і regression-тестом (Методичка Дод. К).

---

### BUG-001 — Помилкові формули `S`, `T`, відсутність обнулення `Q` при `R = 0`, підрахунок `over_budget` у `B_sent` та імпорт чернеткового `channel.py`
- **Дата / хто знайшов:** 2026-11-17..19 / аудит інтеграції (`Сергій Бобенко`, `Андрій Стрикун`)
- **Опис:** у першій версії `src/channel_runner.py` імпортувався кореневий заглушковий `channel.py` замість `team/channel.py`, метрика стиснення рахувалася як `S = B_ref / B_sent` (могла перевищувати `1.0` і ділити на `0`), швидкодія як `T = 1 / (1 + t)` без `t_ref = 20.0`, при `R == 0` (або падінні приймача) `Q` не обнулялася, а в `B_sent` потрапляли навіть пакети, які не вмістилися в бюджет часу каналу.
- **Мінімальний приклад:** `evaluate_metrics(frames, [], bytes_sent=36000, compute_time_s=20.0, q_weights=cfg["метрика_Q"])` та прогін 100 пакетів через вузький канал із малим бюджетом.
- **Очікувано:** `from team.channel import Channel`, `S = B_ref / (B_ref + B_sent) = 0.5`, `T = 20 / (20 + 20) = 0.5`, `Q = 0.0` при `R == 0` (за наявності критичних кадрів) або `crashed=True`, `bytes_sent` враховує лише пакети, для яких `ch.sent` зріс.
- **Фактично:** `S = 1.0`, `T = 0.0476`, `Q > 0` при `R == 0`, `bytes_sent` завищений за рахунок відкинутих `over_budget`-пакетів.
- **Причина:** формули та імпорти були взяті з чорнової заготовки до звірки з «Проєкт 1» §4, `team/team_config.json` та `CHANNEL_SPEC_1.md`.
- **Виправлення:** `src/channel_runner.py` (`transmit_packets`, `evaluate_metrics`), коміти `301231b`, `cf2fc15`, `63c5e2d`.
- **Regression-тест:** `tests/test_channel_integration.py::test_evaluate_metrics_exact_formulas_and_edge_cases`, `tests/test_channel_integration.py::test_transmit_packets_over_budget_not_counted_in_b_sent`.

---

### BUG-002 — Стагнація черги невибраних кадрів у `src/dp.py` (втрати 70.5% критичних кадрів в аварійному режимі)
- **Дата / хто знайшов:** 2026-11-19 / аудит наскрізного тракту (`Андрій Стрикун`, `Сергій Бобенко`)
- **Опис:** при пакуванні `team/session.bin` (3000 кадрів, 300 критичних) в аварійному сценарії середня частка втрачених критичних кадрів складала `0.7047` (понад утричі вище порогу `≤ 0.20`).
- **Мінімальний приклад:** `run_end_to_end(load_session_bin("team/session.bin"), scenario_name="аварійний", seed=21)`.
- **Очікувано:** критичні кадри концентруються в перших пакетах, аварійні втрати `≤ 0.20`.
- **Фактично:** 188 пакетів, критичні кадри розпорошені по 185 пакетах, ~190 критичних кадрів обрізаються бюджетом `16.9 с` (`crit_loss_rate = 0.677..0.733`, середнє `0.7047`).
- **Причина:** у `pack_telemetry_stream` невибрані кадри поверталися в голову черги (`unchosen + queue[eff_win:]`), тож ~44 важкі некритичні кадри (`v = 1`) назавжди застрягали у 64-кадровому вікні, збільшуючи `table_nsym` до 145–155 і зменшуючи `W` до ~1000 біт.
- **Виправлення:** розділення невибраних на критичні (в голову) і некритичні (у хвіст): `queue = crit_unchosen + queue[eff_win:] + noncrit_unchosen` (`src/dp.py`), коміт `f3220c6`. Аварійні втрати впали до **`0.1833 ≤ 0.20`**, число пакетів — зі 188 до 128.
- **Regression-тест:** `tests/test_dp.py::test_regression_k1_no_queue_stagnation_concentrates_critical_frames`, `tests/test_channel_integration.py::test_end_to_end_three_scenarios_thresholds_time_limit_and_determinism`.

---

### BUG-003 — Немонотонний `compression_threshold` і сліпий FIFO в `NO_COMPRESSION`
- **Дата / хто знайшов:** 2026-11-19 / аудит `src/dp.py` (`Андрій Стрикун`, `Сергій Бобенко`)
- **Опис:** функція `_should_use_no_compression` змінювала знак/сенс порівняння при `compression_threshold > 0.5`, через що `thr = 0.90` давав той самий результат (`False`), що й `thr = 0.00`. Додатково при переході в `NO_COMPRESSION` кадри бралися просто як перші `take_cnt` (`win_frames[:take_cnt]`) без урахування пріоритетів.
- **Мінімальний приклад:** `[_should_use_no_compression(1000, 600, t) for t in (0.0, 0.3, 0.5, 0.7, 0.9)]`.
- **Очікувано:** монотонний перехід `False → True` (у даному прикладі при `t > 0.4`, бо економія `40%`), а в `NO_COMPRESSION` обираються найцінніші кадри вікна.
- **Фактично:** при `t = 0.7, 0.9` поверталося `False` (бо перевірялося `0.6 > 0.7`), а `NO_COMPRESSION` ігнорував `win_values`.
- **Причина:** змішування двох несумісних визначень порогу (`bit_gain / raw_bits` і `compressed_bits / raw_bits`) в одній функції.
- **Виправлення:** єдина монотонна формула `(bit_gain / raw_bits) < compression_threshold` на `(0.0, 1.0]` та вибір `top take_cnt` за `(-win_values[i], i)` у `NO_COMPRESSION` (`src/dp.py`), коміт `f3220c6`.
- **Regression-тест:** `tests/test_dp.py::test_regression_k2_compression_threshold_monotonic_on_unit_interval`.

---

### BUG-004 — Розбіжність ключів пріоритетів у `compute_frame_values`
- **Дата / хто знайшов:** 2026-11-18 / `Андрій Стрикун`
- **Опис:** рання версія `compute_frame_values` у `src/classification.py` шукала у конфігурації ключі `"0"`, `"1"`, `"2"`, тоді як в офіційному `team/team_config.json` ключі мають назви `"норма"`, `"аномалія"`, `"перехідний_режим"`.
- **Мінімальний приклад:** `compute_frame_values(frames, [0, 1, 2], load_team_config())`.
- **Очікувано:** пріоритети `[1, 5, 3]` (+ `5` за `critical == 1`).
- **Фактично:** `KeyError` або використання некоректного значення за замовчуванням для всіх класів.
- **Причина:** зміна структури `team/team_config.json` порівняно з чернетковим конфігом.
- **Виправлення:** додано явний мапінг `CLASS_TO_KEY = {0: "норма", 1: "аномалія", 2: "перехідний_режим"}` у `src/classification.py`, коміт `1aa2b4e`.
- **Regression-тест:** `tests/test_classification.py::test_compute_frame_values_mapping_and_errors`.

---

### BUG-005 — Захардкоджений пароль `"algo2026"` у модулях `dp.py` та `ga.py`
- **Дата / хто знайшов:** 2026-11-19 / аудит безпеки `C5` (`Андрій Стрикун`, `Мілана Будзановська`)
- **Опис:** при відсутності явного аргументу `password` функції `pack_telemetry_stream` (`src/dp.py`) та `PipelineSessionContext` (`src/ga.py`) підставляли літерал `"algo2026"`, ігноруючи змінну середовища `TLM1_SESSION_PASSWORD`.
- **Мінімальний приклад:** запуск конвеєра з установленою змінною `TLM1_SESSION_PASSWORD="CustomPass"` без явного аргументу `password`.
- **Очікувано:** єдине джерело пароля `resolve_session_password(config)` (`TLM1_SESSION_PASSWORD` → `config["пароль_сеансу"]` → `team/team_config.json`).
- **Фактично:** `dp.py` і `ga.py` читали `"algo2026"` напряму.
- **Причина:** модулі розроблялися паралельно до появи централізованої функції `resolve_session_password` у `src/crypto.py`.
- **Виправлення:** реалізовано централізований `resolve_session_password` у `src/crypto.py` (`d79ed3b`) та переведено на нього `src/dp.py`, `src/ga.py`, `src/channel_runner.py`, коміти `d79ed3b`, `f3220c6`, `36eafc9`, `63c5e2d`.
- **Regression-тест:** `tests/test_crypto.py::test_edge_cases_and_c5_no_secrets_leaked`.

---

### BUG-006 — Падіння GitHub Actions CI через неповний `requirements.txt` і відсутність `pyproject.toml`
- **Дата / хто знайшов:** 2026-11-19 / перевірка CI (`Андрій Стрикун`)
- **Опис:** workflow `.github/workflows/ci.yml` падав за ~10 с на кроці `Lint with ruff`, а потім падав би на `pytest --cov=src`.
- **Мінімальний приклад:** `pip install -r requirements.txt && ruff check src/ tests/ experiments/ && pytest --cov=src` у чистому середовищі.
- **Очікувано:** усі залежності CI ставляться з `requirements.txt`, `ruff` повертає 0 помилок, `pytest --cov=src` проходить.
- **Фактично:** `ruff` і `pytest-cov` були відсутні в `requirements.txt`, а без `pyproject.toml` лінтер видавав 24–78 помилок (включно з `E402` у `experiments/`, реекспортами `F401` у `src/crypto.py` та `UP009` на незмінному `team/channel.py`).
- **Причина:** `.github/workflows/ci.yml` додали раніше, ніж оновили `requirements.txt` і конфігурацію лінтера.
- **Виправлення:** додано `ruff>=0.4.0` і `pytest-cov>=4.1.0` у `requirements.txt`, створено `pyproject.toml` (`extend-exclude = ["team", "artifacts"]`, `pythonpath = ["."]`), виправлено зауваження коду (`B904`, `E741`, `UP*`), коміт `c11c30d`.
- **Regression-тест:** автоматична перевірка `ruff check .` (0 помилок) та `pytest --cov=src` (74/74 PASS, 90% покриття) у `.github/workflows/ci.yml`.

---

### BUG-007 — Розсинхронізація `results/best_params.json` та вкладене поле `valid_macro_f1` у `run_ablation.py`
- **Дата / хто знайшов:** 2026-11-19 / перевірка `results/` (`Андрій Стрикун`)
- **Опис:** після мерджу `origin/ga` у `dev` файл `results/best_params.json` залишився від старого до-фіксового коміту `e0ed44d` (`block_size = 36`, `val_q = 0.6965`), суперечачи оновленому `results/ga_history.csv` (`block_size = 64`, `val_mean = 0.7195`). Крім того, перша чернетка `run_ablation.py` читала `valid_macro_f1` з кореня `classifier_metrics.json` замість `selected_model.valid_macro_f1`.
- **Мінімальний приклад:** звірка `results/best_params.json` із `results/ga_history.csv` і перегляд секції `classifier` у `results/final_metrics.json`.
- **Очікувано:** усі файли в `results/` узгоджені між собою, а `final_metrics.json["classifier"]["valid_macro_f1"] == 0.92265`.
- **Фактично:** `best_params.json` містив старі гени, а `valid_macro_f1` міг бути `null`.
- **Причина:** частковий коміт артефактів ГА до виправлення `src/dp.py` та різна вкладеність JSON-схем.
- **Виправлення:** виправлено зчитування `selected_model.valid_macro_f1` у `experiments/run_ablation.py`, усі 13 таблиць перегенеровано єдиним прогоном `experiments/run_all.py --skip-train`, коміт `c11c30d`.
- **Regression-тест:** відтворення через `python experiments/run_all.py --skip-train` (перевіряє наявність і цілісність усіх 13 файлів у `results/`).

---

### BUG-008 — Відсутність ліміту `symbol_count` в арифметичному декодері (ризик DoS)
- **Дата / хто знайшов:** 2026-11-18 / тестування стиснення (`Дарина Ємець`)
- **Опис:** пошкоджений або спеціально сформований заголовок арифметичного блоку з гігантським значенням `symbol_count` (наприклад, `2^31`) міг змусити декодер `decode_arithmetic_block` / `decode_arithmetic_bitstream` виконувати мільярди ітерацій і вичерпати пам'ять.
- **Мінімальний приклад:** підміна поля `symbol_count = MAX_BLOCK_SYMBOLS + 1` (`1_048_577`) у заголовку арифметичного блоку перед передачею в декодер.
- **Очікувано:** миттєва відмова з контрольованим кодом `A_LENGTH` до початку декодування.
- **Фактично:** відсутність верхньої межі дозволяла неконтрольований цикл розпакування.
- **Причина:** у першій чернетці декодера перевірялися лише коректність таблиці частот і CRC, без обмеження максимальної довжини блоку.
- **Виправлення:** запроваджено константу `MAX_BLOCK_SYMBOLS = 1_048_576` і ранню валідацію `symbol_count` із винятком `A_LENGTH` у `src/arithmetic.py`, коміти `652a057`, `5eff96a`.
- **Regression-тест:** `tests/test_arithmetic.py::test_reject_oversized_symbol_count`, `tests/test_arithmetic.py::test_direct_decoder_rejects_oversized_symbol_count`.

---

### BUG-009 — Неповний облік службових заголовків блоку у порівнянні Гаффмана та арифметичного кодера
- **Дата / хто знайшов:** 2026-11-18 / експерименти стиснення (`Дарина Ємець`)
- **Опис:** перша версія порівняльного скрипта (`experiments/run_compression_comparison.py`, таблиця `huffman_vs_arithmetic.csv`) не враховувала окремо 14-байтовий службовий заголовок блоку (`header_bytes`) і точні `code_bits`, через що підсумковий коефіцієнт `full_ratio` на 64-кадрових блоках телеметрії був неповним.
- **Мінімальний приклад:** порівняння сумарного розміру 47 вікон `team/session.bin` із урахуванням таблиці та 14-байтових заголовків кожного блока.
- **Очікувано:** чесний облік `data_bytes + table_bytes + header_bytes` (658 Б заголовків на 47 блоків) та окремі колонки `compression_ratio` і `full_ratio`.
- **Фактично:** порівняння велося без 14-байтових заголовків блоків.
- **Причина:** на першому етапі оцінювалися лише `bitstream + table` без повного формату блока.
- **Виправлення:** створено уточнений скрипт `experiments/run_compression_comparison_fixed.py` із генерацією `results/huffman_vs_arithmetic_v2.csv`, коміти `652a057`, `5eff96a`.
- **Regression-тест:** `tests/test_arithmetic.py::test_arithmetic_size_matches_serialized_block`, відтворення через `experiments/run_compression_comparison_fixed.py`.

---

### BUG-010 — Відсутність контролю унікальності `nonce` і переповнення `uint32 packet_seq` у першій версії `src/crypto.py`
- **Дата / хто знайшов:** 2026-11-18 / рев'ю криптографії `C5` (`Мілана Будзановська`)
- **Опис:** у початковій версії `src/crypto.py` (`27bf809`) функція `seal_packet` приймала довільний `nonce` без відстеження повторного використання в межах сеансу й без перевірки переповнення `uint32 packet_seq`, що в режимі `AES-256-CTR` призводить до повторного використання keystream.
- **Мінімальний приклад:** виклик шифрування двох пакетів із тим самим `nonce` у межах одного сеансу або передача `packet_seq = 0x1_0000_0000`.
- **Очікувано:** відмова (`ValueError`) при спробі повторного використання `nonce` або перевищенні `MAX_PACKET_SEQ = 0xFFFFFFFF`.
- **Фактично:** `seal_packet` шифрував повторний `nonce` без помилки.
- **Причина:** базова версія ЧП1 реалізувала лише stateless-функції без стану сеансу.
- **Виправлення:** додано `SessionCryptoContext` (множина `_used_nonces`, генерація `nonce = prefix8 || u32_BE(seq)`, валідація `validate_header_fields` у `src/packet_format.py`), коміти `d79ed3b`, `c264e84`, `f486fe8`.
- **Regression-тест:** `tests/test_crypto.py::test_nonce_uniqueness_and_u32_overflow`.

---

### BUG-011 — Приймання `bool` і неконтрольований `struct.error` для `float` у полях `TelemetryFrame`
- **Дата / хто знайшов:** 2026-11-18 / рев'ю телеметрії `C2.1` (`Сергій Бобенко`)
- **Опис:** у початковій реалізації `src/telemetry.py` перевірялися лише числові діапазони полів кадру, але не точний тип `int` без `bool`: у Python `isinstance(True, int) is True`, тож `True`/`False` мовчки перетворювалися на `1`/`0`, а `float` викликав необроблений `struct.error` під час `to_bytes()`.
- **Мінімальний приклад:** `TelemetryFrame(timestamp_ms=True, altitude_dm=100, temp_c10=20, vibration=0, mode=0, critical=1).to_bytes()` або `altitude_dm=12.5`.
- **Очікувано:** негайна контрольована відмова `ProtocolError("E_PAYLOAD", ...)` для будь-якого `bool` чи `float`.
- **Фактично:** `bool` серіалізувався без помилки, а `float` падав із `struct.error`.
- **Причина:** відсутність явного фільтра `isinstance(val, bool) or not isinstance(val, int)` у валідації кадру.
- **Виправлення:** запроваджено `_require_exact_int` і `ProtocolError` у `src/telemetry.py`, коміт `564ef12`.
- **Regression-тест:** `tests/test_telemetry.py::test_strict_type_and_range_validation_boundaries`.

---

### BUG-012 — Обрив черги (`break`) при `W <= 0` і відсутність автоматичного fallback на `NO_COMPRESSION`
- **Дата / хто знайшов:** 2026-11-18 / аудит ЧП1 (`Сергій Бобенко`)
- **Опис:** у початковому циклі пакування (`scripts/run_checkpoint1.py`) за порожнього результату рюкзака (`chosen_idx == []`, наприклад при `W <= 0` через велику таблицю Гаффмана на високій ентропії) виконувався `break`, через що вся решта черги кадрів мовчки губилася.
- **Мінімальний приклад:** виклик `pack_telemetry_stream` на вікні кадрів із високою ентропією байтів, де `2 * table_nsym` з'їдає місткість `W <= 0`.
- **Очікувано:** свідомий перехід на режим `FLAG_NO_COMPRESSION` (`table_nsym = 0`) і продовження пакування всіх кадрів черги без втрат (`C2.5`).
- **Фактично:** цикл переривався (`break`), решта кадрів не пакувалася.
- **Причина:** `FLAG_NO_COMPRESSION` був реалізований у форматі пакета, але не був підключений як автоматичний fallback у пакувальнику.
- **Виправлення:** додано автоматичний перехід на `FLAG_NO_COMPRESSION` у `pack_telemetry_stream` (`src/dp.py`) та `pack_stream_baseline` (`scripts/run_checkpoint1.py`), коміти `0a8444b`, `301231b`.
- **Regression-тест:** `tests/test_dp.py::test_no_compression_fallback_when_w_non_positive`.

---

### BUG-013 — Неізольований кеш KDF у приймачі (ризик DoS солями) та ім'я тимчасового файлу запису
- **Дата / хто знайшов:** 2026-11-18 / рев'ю приймача `C5` (`Сергій Бобенко`)
- **Опис:** у першій ітерації `src/receiver.py` функція `_cached_derive_keys` з `@functools.lru_cache(maxsize=16)` делегувала виклик до `src.crypto.derive_keys`, яка додатково записувала всі унікальні солі в необмежений словник `src.crypto._KEY_CACHE` (при спамі солями пам'ять усе одно зростала). Також `_atomic_write_frames` створювала випадковий файл `NamedTemporaryFile` замість специфікованого `output.bin.tmp`.
- **Мінімальний приклад:** передача в `receive_packet` 30 пакетів із різними 16-байтовими солями та перехоплення аргументу `src` у `os.replace` під час `receive_stream(..., output_path=out_bin)`.
- **Очікувано:** кеш приймача жорстко обмежений `maxsize=16` без побічного росту чужого словника, а тимчасовий файл має шлях `<target>.tmp` (`output.bin.tmp`) і видаляється при збої `os.replace`.
- **Фактично:** ріст кешу в `src.crypto` та випадкова назва `.tmp`-файла.
- **Причина:** повторне використання внутрішньої функції `derive_keys` без ізоляції кешу та використання `tempfile.NamedTemporaryFile` замість фіксованого `Path(f"{target_path}.tmp")`.
- **Виправлення:** переведено `_cached_derive_keys` на прямий виклик `hashlib.pbkdf2_hmac` під `@functools.lru_cache(maxsize=16)` та оновлено `_atomic_write_frames` у `src/receiver.py`, коміт `63d897c`.
- **Regression-тест:** `tests/test_receiver.py::test_kdf_lru_cache_dos_bound_and_c5_safe_logging`, `tests/test_receiver.py::test_receive_stream_reordering_deduplication_and_atomic_write`.

---

### BUG-014 — Відсутність перевірки порожності таблиці (`not lengths`) у `unpack_bitstream` при `symbol_count == 0`
- **Дата / хто знайшов:** 2026-11-18 / рев'ю стиснення (`Дарина Ємець`)
- **Опис:** у початковій реалізації `src/huffman.py` функція `unpack_bitstream(bitstream, lengths, symbol_count=0)` одразу повертала `b""` навіть за непорожньої таблиці `lengths` або непорожнього `bitstream`, що суперечило контракту порожнього потоку (`FORMAT_SPEC_1.md` §3).
- **Мінімальний приклад:** `unpack_bitstream(b"", {0: 1}, symbol_count=0)` або `unpack_bitstream(b"\x00", {}, symbol_count=0)`.
- **Очікувано:** відхилення з контрольованим кодом `E_PAYLOAD` / `E_CODE`.
- **Фактично:** мовчазне повернення `b""` без перевірки узгодженості `lengths` і `bitstream`.
- **Причина:** ранній вихід `if symbol_count == 0: return b""` до валідації аргументів.
- **Виправлення:** додано перевірку порожності `lengths` та `bitstream` при `symbol_count == 0` у `src/huffman.py` (та в `src/receiver.py`), коміти `652a057`, `63d897c`.
- **Regression-тест:** `tests/test_huffman.py::test_negative_errors_e_code_and_e_payload`, `tests/test_receiver.py::test_all_negative_mutations_and_ambiguous_codes`.

---

### BUG-015 — Опечатка імені файла ініціалізації пакета `team/__init.py__`
- **Дата / хто знайшов:** 2026-11-19 / перевірка CI та структури репозиторію (`Андрій Стрикун`)
- **Опис:** під час додавання пакета `team/` (коміт `301231b`) замість `team/__init__.py` було створено порожній файл `team/__init.py__` із пропущеним подвійним підкресленням, через що `team` не був повноцінним Python-пакетом (працював лише як implicit namespace package).
- **Мінімальний приклад:** перевірка вмісту `team/` та імпорт `from team.channel import Channel` у середовищах або інструментах збору пакетів, що очікують явний `__init__.py`.
- **Очікувано:** наявність `team/__init__.py` і відсутність зайвого `team/__init.py__`.
- **Фактично:** у репозиторії лежав `team/__init.py__`.
- **Причина:** опечатка під час створення файла вручну.
- **Виправлення:** перейменовано `team/__init.py__` → `team/__init__.py`, коміт `c11c30d`.
- **Regression-тест:** імпорт `from team.channel import Channel` у `tests/test_channel_integration.py` та прогін `pytest` у CI (`.github/workflows/ci.yml`).
