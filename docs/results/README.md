# Ergebnis-Artefakte der Relevanz-Regularisierung

Die Zahlen, auf die sich [`../relevance_regularization.md`](../relevance_regularization.md)
§13 stützt. Sie liegen hier unter Versionskontrolle, weil sie sonst nur in `temp/`
existierten — ein Ergebnisdokument, dessen Zahlen sich nicht aus dem Repository
nachvollziehen lassen, ist nur halb belegt.

Erzeugt mit `scripts/eval_localization.py`, jeweils auf **911 Chunks aus 624 Test-Clips**
(`data/processed/test.h5` + `test_masks.npz`).

| Datei | Checkpoint | trainierte Batches | ausgewertet bei | `ratio_over_chance` |
|---|---|---|---|---|
| `loc_baseline.json` | `checkpoints/videomae_phase2.ckpt` | — | — | 1,921 |
| `loc_baseline_regions.json` | wie oben, mit `--per-region` | — | — | — |
| `loc_lambda0_control.json` | Sweep-Arm λ=0 | 6.000 | **6.000** | 1,867 |
| `loc_lambda002.json` | Sweep-Arm λ=0,02 | 6.000 | **6.000** | 8,210 |
| `loc_lambda01.json` | Sweep-Arm λ=0,1 | 6.000 | **6.000** | 11,418 |
| `loc_aux_head.json` | `experiment=train_video_loc_head` | 6.000 | **6.000** | 2,200 |

Alle Arme sind schrittgleich bei Batch 6.000 ausgewertet; geprüft über den in jedem
Checkpoint gespeicherten `global_step`, nicht über Dateinamen oder Änderungsdatum.

> **Frühere Fassung war nicht schrittgleich.** Bis zum 2026-08-17 enthielten
> `loc_lambda002.json` und `loc_lambda01.json` Werte aus Batch **3.000** (3,410 bzw.
> 4,689), weil `save_top_k=2` bei steigendem `val/loss` nur die frühesten Checkpoints
> behielt und `last.ckpt` bitweise eine Kopie des letzten Speicherstands ist. Beide
> Arme wurden mit `save_top_k: -1` wiederholt; die Kontrolle blieb dabei exakt
> unverändert (1,867), was als Kontrollprobe für den Wiederholungslauf dient.
> Details: §13.1 des Hauptdokuments.

## `training_curve.csv` — Lokalisierung über die Trainingsdauer

Dieselbe Auswertung, angewandt auf **alle** Zwischen-Checkpoints statt nur auf den
letzten: 12 Messpunkte (λ=0 bei 4.500/6.000, λ=0,02 und λ=0,1 je bei
1.500/3.000/4.500/6.000, Aux-Head bei 5.000/6.000). Eine Zeile je Checkpoint, mit
Bootstrap-Intervallen für jede Metrik sowie `val_loss` / `val_auc_video` aus der
zugehörigen `metrics.csv`.

Möglich ohne erneutes Training, weil `save_top_k: -1` jeden Validierungs-Checkpoint
erhält. Ausgewertet in §13.5; die Kernaussage ist, dass die Lokalisierung bei Batch 6.000
noch **beschleunigt** (λ=0,02: +0,774 → +1,800 je 1.000 Batches), die Kontrolle dagegen
exakt flach bleibt (−0,000/1k).

```bash
powershell -File scripts/eval_training_curve.ps1   # ~30 min, 12 Checkpoints
python -m scripts.build_training_curve             # -> docs/results/training_curve.csv
```

`build_training_curve.py` gibt zusätzlich eine Plateau-Diagnose auf stdout aus: die
Zuwachsrate je 1.000 Batches pro Abschnitt. Eine Rate, die nicht gegen null fällt,
bedeutet, dass der Lauf abgeschnitten und nicht ausgelaufen ist.

> Die Lauf-Verzeichnisse sind in beiden Skripten **fest eingetragen**, nicht gesucht.
> Grund: die Experimentnamen sind Präfixe voneinander (`…lambda0` liegt in
> `…lambda01`), und eine Teilstring-Suche hatte genau deshalb schon einmal den falschen
> Lauf getroffen.

## `relevance_method_ablation.csv` — LRP-unabhängige Gegenprobe

Beantwortet die Frage, die der λ-Sweep nicht beantworten kann: Ist der
Lokalisierungsgewinn eine Eigenschaft des **Modells** oder nur der Grösse, auf die
optimiert wurde? Der Loss minimiert ein Massenverhältnis auf AttnLRP-Relevanz; Chefer
et al. (ICCV 2021) teilt damit keine Berechnung und ist deshalb die unabhängige Probe.

2 Methoden × 3 Arme (Baseline, Kontrolle λ=0, λ=0,02), alle auf denselben 911 maskierten
Test-Chunks aus 624 Clips. `_tests.csv` enthält die gepaarten Wilcoxon-Tests über Clips.

| `ratio_over_chance` | Baseline | Kontrolle | λ=0,02 | reg/Kontrolle |
|---|---|---|---|---|
| AttnLRP (bivariat) | 1,953 | 1,898 | 7,910 | 4,17× |
| Chefer | 1,574 | 1,536 | 2,360 | 1,54× |

| `pointing_game` | Baseline | Kontrolle | λ=0,02 |
|---|---|---|---|
| AttnLRP (bivariat) | 0,299 | 0,280 | 0,769 |
| Chefer | 0,263 | 0,221 | 0,747 |

Kernaussage (§9.3 in [`../chefer_ablation.md`](../chefer_ablation.md)): Die Kontrolle
liegt in **beiden** Methoden unter der Baseline (0,97×) — Weitertrainieren allein
lokalisiert nicht. Beim Pointing Game, der einzigen auf [0,1] beschränkten und damit
methodenübergreifend vergleichbaren Metrik, landen beide Verfahren fast auf demselben
Endwert. Die Massenkonzentration steigt dagegen unter AttnLRP rund dreimal stärker als
unter Chefer, weil sie die optimierte Grösse ist.

```bash
python -m scripts.build_method_ablation
```

> Die absoluten `ratio_over_chance`-Höhen sind zwischen den Methoden **nicht**
> vergleichbar (Chefer ist bauartbedingt nicht-negativ und flacher). Vergleichbar sind
> die Verhältnisse innerhalb einer Methode und das Pointing Game.

## Format

Jede Datei enthält pro Metrik den Mittelwert über Clips und ein 95-%-Bootstrap-Intervall:

```json
{
  "ratio_over_chance": {"mean": 1.9206, "ci": [1.8393, 2.0018]},
  "rma": {...}, "pointing_game": {...}, "iou": {...},
  "n_clips": 624
}
```

`ratio_over_chance` ist die Leitgröße: Anteil der Relevanzmasse innerhalb der
Manipulationsmaske, geteilt durch den Flächenanteil der Maske. **1,0 = die Relevanz
ignoriert die Maske vollständig.**

## Nachrechnen

```bash
python -m scripts.eval_localization \
    --ckpt <checkpoint> --split test \
    --resume-csv temp/loc_<name>.csv \
    --summary-json temp/loc_<name>.json
```

Die zugehörigen Per-Chunk-CSVs bleiben in `temp/` (je ~200 KB) und sind nicht versioniert;
sie lassen sich mit demselben Befehl neu erzeugen. Voraussetzung sind die Masken-Stores
aus `scripts/build_manipulation_masks.py`.
