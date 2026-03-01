# LM Studio Test-Suite (V2 + Hardmode)

Dieses Repository enthält zwei Benchmark-Skripte für lokale LM-Studio-Modelle (OpenAI-kompatible API auf `http://localhost:1234/v1`) und die dazugehörigen Logs.

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

| Modell | Pass-Rate (gesamt) | Ø Latenz |
|---|---:|---:|
| `qwen/qwen3-14b` | **91.7%** (22/24) | 23.87s |
| `mistralai/ministral-3-14b-reasoning` | 87.5% (21/24) | 4.13s |
| `google/gemma-3-12b` | 83.3% (20/24) | 6.18s |
| `google/gemma-3-4b` | 83.3% (20/24) | **2.03s** |
| `qwen2.5-coder-7b-instruct` | 83.3% (20/24) | 2.56s |
| `zai-org/glm-4.6v-flash` | 79.2% (19/24) | 19.15s |
| `deepseek/deepseek-r1-0528-qwen3-8b` | 66.7% (16/24) | 38.44s |
| `qwen/qwen3-4b-thinking-2507` | 50.0% (12/24) | 15.59s |
| `qwen/qwen3-8b` | 45.8% (11/24) | 22.89s |

**Kategorie-Überblick (V2):**

| Kategorie | Pass-Rate | Ø Latenz |
|---|---:|---:|
| Coding | 85.2% (46/54) | 12.65s |
| Tool-Use | 79.6% (43/54) | 19.45s |
| Reasoning | 66.7% (72/108) | 13.92s |

### 2) Hardmode (neuester JSONL-Stand: `20260301_134249`)

| Modell | Pass-Rate (gesamt) | Ø Latenz |
|---|---:|---:|
| `qwen/qwen3-14b` | **88.9%** (8/9) | 41.28s |
| `qwen2.5-coder-7b-instruct` | 66.7% (6/9) | 4.04s |
| `google/gemma-3-4b` | 44.4% (4/9) | 4.89s |
| `mistralai/ministral-3-14b-reasoning` | 44.4% (4/9) | 6.33s |
| `zai-org/glm-4.6v-flash` | 44.4% (4/9) | 33.50s |

**Kategorie-Überblick (Hardmode):**

| Kategorie | Pass-Rate | Ø Latenz |
|---|---:|---:|
| Tool-Use | 73.3% (11/15) | 26.95s |
| Reasoning | 53.3% (8/15) | 12.74s |
| Coding | 46.7% (7/15) | 14.33s |

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

Optional lassen sich Modelle, Repeats, Timeout, Seed, Output-Verzeichnis usw. über CLI-Argumente setzen (siehe `--help`).
