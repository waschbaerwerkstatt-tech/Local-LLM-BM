# LM Studio Test-Suite (V2 + Hardmode)

Dieses Repository enthält zwei Benchmark-Skripte für lokale LM-Studio-Modelle (OpenAI-kompatible API auf `http://localhost:1234/v1`) und die dazugehörigen Logs.
100% mit Codex und Antigravity erstellt.

## Was machen die Skripte?

| Skript | Zweck | Tests | Besonderheiten | Ausgabepfad (Default) |
|---|---|---:|---|---|
| `lmstudio_testsuite_v2.py` | Normaler Benchmark für Reasoning, Coding und Tool-Use | 8 Tests pro Wiederholung (4 Reasoning, 2 Coding, 2 Tool-Use) | Tool-Use optional, robuste Zahlenerkennung, Retry bei API-Fehlern, Resume/Checkpoint | `logs_v2/` |
| `lmstudio_testsuite_hard.py` | Strenger Benchmark (Hardmode) mit höherem Formatdruck | 9 Tests pro Wiederholung (3 Reasoning, 3 Coding, 3 Tool-Use) | `final answer only`-Heuristik, Tool-Use in Tool-Tests verpflichtend, Multi-Round-Toolcalls (bis 4 Runden / 8 Calls) | `logs_hard_v2/` |

### Was wird pro Run geschrieben?

| Datei | Inhalt |
|---|---|
| `*.jsonl` | Vollständige Einzelresultate pro Testlauf (inkl. Output, Validator-Details, Tool-Calls) |
| `*_summary_*.csv` | Flache Tabellen-Zusammenfassung pro Testlauf |
| `*_report_*.txt` | Kompakter Report pro Modell (Pass-Rate, Latenz, Kategorien) |
| `.lmstudio_test_run_state_*.json` | Checkpoint/Resume-Status |

## Was wird gemessen und warum?

Ziel der Test-Suite ist ein **praxisnaher Vergleich lokaler Modelle** auf einem Mac Mini (16 GB RAM), damit man schneller entscheiden kann, welches Modell für den eigenen Workflow am sinnvollsten ist.

Gemessen wird pro Modell:

- **Qualität/Robustheit** über die Pass-Rate (`Passes/Tests`) in vier Sichten:
  - Gesamt
  - Coding
  - Tool-Use
  - Reasoning
- **Geschwindigkeit** über die durchschnittliche Latenz pro Test.
- **Verlässlichkeit bei Tool-Aufgaben** (insbesondere im Hardmode), also ob das Modell Tool-Aufrufe korrekt nutzt und trotzdem die richtige finale Antwort liefert.

Warum das wichtig ist:

- Ein Modell kann z. B. sehr schnell sein, aber in Reasoning schwächer.
- Ein anderes Modell kann sehr gute Genauigkeit haben, aber deutlich höhere Latenz.
- Durch die getrennten Kategorien sieht man sofort, **wo** ein Modell stark ist (z. B. Coding) und **wo nicht** (z. B. Tool-Use unter strengen Formatregeln).

Kurz: Die Suite soll kein theoretischer Benchmark sein, sondern eine **konkrete Entscheidungsgrundlage** für den lokalen Einsatz.

## Testumgebung

Diese Log-Dateien enthalten Ergebnisse für einen **Mac Mini mit 16 GB Arbeitsspeicher**.

## Vorhandene Runs (Stand: 2026-03-01)

| Run | Hauptdatei | Umfang | Kommentar |
|---|---|---|---|
| V2 Run 1 | `logs_v2/lmstudio_v2_20260301_105306.jsonl` | 216 Test-Records (9 Modelle × 3 Repeats × 8 Tests) | Vollständig; CSV-`summary` enthält hier nur Header |
| V2 Run 2 | `logs_v2/lmstudio_v2_20260301_122234.jsonl` | 216 Test-Records (vollständig) | Beste Grundlage für V2-Vergleich |
| Hardmode | `logs_hard_v2/lmstudio_hard_v2_20260301_134249.jsonl` | 45 Test-Records (5 Modelle × 1 Repeat × 9 Tests) | Neuester Hardmode-Stand; CSV enthält nur Header |

## Ergebnis-Aufbereitung

### 1) V2 (neuester kompletter Run: `20260301_122234`)

| Modell | Gesamt (Passrate / Passes / Ø Latenz) | Coding (Passrate / Passes / Ø Latenz) | Tool-Use (Passrate / Passes / Ø Latenz) | Reasoning (Passrate / Passes / Ø Latenz) |
|---|---|---|---|---|
| `qwen/qwen3-14b` | **91.7% (22/24) - 23.87s** | 100.0% (6/6) - 4.54s | 83.3% (5/6) - 33.04s | 91.7% (11/12) - 28.96s |
| `mistralai/ministral-3-14b-reasoning` | 87.5% (21/24) - 4.13s | 100.0% (6/6) - 4.26s | 100.0% (6/6) - 4.46s | 75.0% (9/12) - 3.90s |
| `google/gemma-3-12b` | 83.3% (20/24) - 6.18s | 83.3% (5/6) - 9.07s | 100.0% (6/6) - 6.05s | 75.0% (9/12) - 4.81s |
| `google/gemma-3-4b` | 83.3% (20/24) - **2.03s** | 83.3% (5/6) - 2.44s | 100.0% (6/6) - 2.22s | 75.0% (9/12) - 1.74s |
| `qwen2.5-coder-7b-instruct` | 83.3% (20/24) - 2.56s | 100.0% (6/6) - 2.40s | 83.3% (5/6) - 2.94s | 75.0% (9/12) - 2.46s |
| `zai-org/glm-4.6v-flash` | 79.2% (19/24) - 19.15s | 66.7% (4/6) - 32.90s | 50.0% (3/6) - 20.26s | 100.0% (12/12) - 11.72s |
| `deepseek/deepseek-r1-0528-qwen3-8b` | 66.7% (16/24) - 38.44s | 100.0% (6/6) - 16.75s | 50.0% (3/6) - 70.67s | 58.3% (7/12) - 33.16s |
| `qwen/qwen3-4b-thinking-2507` | 50.0% (12/24) - 15.59s | 50.0% (3/6) - 16.65s | 100.0% (6/6) - 15.10s | 25.0% (3/12) - 15.31s |
| `qwen/qwen3-8b` | 45.8% (11/24) - 22.89s | 83.3% (5/6) - 24.86s | 50.0% (3/6) - 20.27s | 25.0% (3/12) - 23.21s |

### 2) Hardmode (neuester JSONL-Stand: `20260301_134249`)

| Modell | Gesamt (Passrate / Passes / Ø Latenz) | Coding (Passrate / Passes / Ø Latenz) | Tool-Use (Passrate / Passes / Ø Latenz) | Reasoning (Passrate / Passes / Ø Latenz) |
|---|---|---|---|---|
| `qwen/qwen3-14b` | **88.9% (8/9) - 41.28s** | 66.7% (2/3) - 12.02s | 100.0% (3/3) - 62.34s | 100.0% (3/3) - 49.47s |
| `qwen2.5-coder-7b-instruct` | 66.7% (6/9) - 4.04s | 66.7% (2/3) - 6.21s | 100.0% (3/3) - 5.48s | 33.3% (1/3) - 0.42s |
| `google/gemma-3-4b` | 44.4% (4/9) - 4.89s | 33.3% (1/3) - 5.82s | 66.7% (2/3) - 8.40s | 33.3% (1/3) - 0.44s |
| `mistralai/ministral-3-14b-reasoning` | 44.4% (4/9) - 6.33s | 33.3% (1/3) - 12.36s | 66.7% (2/3) - 5.87s | 33.3% (1/3) - 0.76s |
| `zai-org/glm-4.6v-flash` | 44.4% (4/9) - 33.50s | 33.3% (1/3) - 35.25s | 33.3% (1/3) - 52.66s | 66.7% (2/3) - 12.60s |

## Kurz-Kommentar zu den Logs

- ✅ **Stärkstes Gesamtmodell (Accuracy):** `qwen/qwen3-14b` in V2 und Hardmode.
- ⚡ **Schnell und solide:** `google/gemma-3-4b` und `qwen2.5-coder-7b-instruct` liefern gute Pass-Raten bei sehr niedriger Latenz.
- ⚠️ **Schwerpunkt-Probleme in den Tests:**
  - V2: `reasoning_04_two_numbers` ist klar der häufigste Stolperstein (nur 29.6% Pass).
  - Hardmode: `coding_hard_02` (BFS-Shortest-Path) war im aktuellen Stand bei 0%.
- 🧪 **Tool-Use ist insgesamt stabiler als strenges Format-Reasoning**: In Hardmode sind viele Fails kein Tool-Problem, sondern Format-/Antwortfehler (`not_final_answer_only`, falscher Zahlenwert).
- 📝 **Wichtig zur Einordnung:** Einige `summary.csv`-Dateien enthalten nur Header. Für vollständige Analyse daher `jsonl` und `report.txt` priorisieren.

## Ausführen

```bash
python3 lmstudio_testsuite_v2.py
python3 lmstudio_testsuite_hard.py
```

## CLI-Argumente (für beide Skripte)

| Argument | Bedeutung | Default `v2` | Default `hard` |
|---|---|---|---|
| `--base-url` | LM-Studio API Endpoint | `http://localhost:1234/v1` | `http://localhost:1234/v1` |
| `--api-key` | API-Key für kompatible OpenAI-Route | `lm-studio` | `lm-studio` |
| `--models` | Liste der zu testenden Modelle | interne Default-Modellliste | interne Default-Modellliste |
| `--repeats` | Wie oft jeder Test pro Modell wiederholt wird | `3` | `3` |
| `--temperature` | Fallback-Temperatur (für Modelle ohne Override) | `0.2` | `0.2` |
| `--max-tokens` | Maximale Antwortlänge pro Request | `900` | `1000` |
| `--timeout` | Timeout pro Request (Sekunden) | `300` | `180` |
| `--seed` | Optionaler Seed (falls Server unterstützt) | `None` | `None` |
| `--outdir` | Zielordner für Logs | `logs_v2` | `logs_hard_v2` |

**Beispiele:**

```bash
# Nur 2 Modelle, dafür 5 Wiederholungen
python3 lmstudio_testsuite_v2.py \
  --models "qwen/qwen3-14b" "google/gemma-3-4b" \
  --repeats 5

# Hardmode mit eigenem Timeout und Output-Ordner
python3 lmstudio_testsuite_hard.py \
  --models "qwen/qwen3-14b" "qwen/qwen3-8b" \
  --repeats 2 \
  --timeout 240 \
  --outdir logs_hard_custom

# Reproduzierbarer Lauf (wenn Seed von der API unterstützt wird)
python3 lmstudio_testsuite_v2.py --seed 42
```

Hinweis: In beiden Skripten existieren modell-spezifische Temperatur-Overrides. `--temperature` wirkt primär als Fallback für Modelle ohne eigenen Override.

## Links: LM Studio + Modelkarten

- LM Studio: [https://lmstudio.ai](https://lmstudio.ai)
- LM Studio Models: [https://lmstudio.ai/models](https://lmstudio.ai/models)
- OpenRouter: [https://openrouter.ai](https://openrouter.ai)

Modellgrößen unten sind die **lokalen Installationsgrößen auf diesem Mac** (konkrete MLX/GGUF-Varianten, nicht die theoretische FP16-Größe).

| Modell (im Repo) | Größe (lokal, GB) | OpenRouter Modelkarte | LM Studio Modelkarte |
|---|---:|---|---|
| `zai-org/glm-4.6v-flash` | 6.6 | [z-ai/glm-4.6v](https://openrouter.ai/z-ai/glm-4.6v) *(nächstes verfügbares Pendant)* | [z-ai/glm-4.6v](https://lmstudio.ai/models/z-ai/glm-4.6v) *(nächstes verfügbares Pendant)* |
| `google/gemma-3-12b` | 7.5 | [google/gemma-3-12b-it](https://openrouter.ai/google/gemma-3-12b-it) | [google/gemma-3-12b](https://lmstudio.ai/models/google/gemma-3-12b) |
| `mistralai/ministral-3-14b-reasoning` | 8.5 | [mistralai/ministral-14b-2410](https://openrouter.ai/mistralai/ministral-14b-2410) *(nächstes verfügbares Pendant)* | [Mistral-Modelle bei LM Studio](https://lmstudio.ai/models/mistralai) *(Provider-Seite)* |
| `qwen/qwen3-8b` | 4.3 | [qwen/qwen3-8b](https://openrouter.ai/qwen/qwen3-8b) | [qwen/qwen3-8b](https://lmstudio.ai/models/qwen/qwen3-8b) |
| `qwen/qwen3-14b` | 7.8 | [qwen/qwen3-14b](https://openrouter.ai/qwen/qwen3-14b) | [qwen/qwen3-14b](https://lmstudio.ai/models/qwen/qwen3-14b) |
| `qwen/qwen3-4b-thinking-2507` | 2.1 | [qwen/qwen3-4b-thinking](https://openrouter.ai/qwen/qwen3-4b-thinking) *(nahes Pendant)* | [qwen/qwen3-4b-thinking-2507](https://lmstudio.ai/models/qwen/qwen3-4b-thinking-2507) |
| `qwen2.5-coder-7b-instruct` | 4.4 | [qwen/qwen2.5-coder-7b-instruct](https://openrouter.ai/qwen/qwen2.5-coder-7b-instruct) | [qwen/qwen2.5-coder-7b-instruct](https://lmstudio.ai/models/qwen/qwen2.5-coder-7b-instruct) |
| `deepseek/deepseek-r1-0528-qwen3-8b` | 4.3 | [deepseek/deepseek-r1-0528-qwen3-8b](https://openrouter.ai/deepseek/deepseek-r1-0528-qwen3-8b) | [deepseek/deepseek-r1-0528-qwen3-8b](https://lmstudio.ai/models/deepseek/deepseek-r1-0528-qwen3-8b) |
| `google/gemma-3-4b` | 2.8 | [google/gemma-3-4b-it](https://openrouter.ai/google/gemma-3-4b-it) | [google/gemma-3-4b](https://lmstudio.ai/models/google/gemma-3-4b) |
