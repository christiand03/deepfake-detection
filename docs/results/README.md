# Ergebnis-Artefakte der Relevanz-Regularisierung

Die Zahlen, auf die sich [`../relevance_regularization.md`](../relevance_regularization.md)
§13 stützt. Sie liegen hier unter Versionskontrolle, weil sie sonst nur in `temp/`
existierten — ein Ergebnisdokument, dessen Zahlen sich nicht aus dem Repository
nachvollziehen lassen, ist nur halb belegt.

Erzeugt mit `scripts/eval_localization.py`, jeweils auf **911 Chunks aus 624 Test-Clips**
(`data/processed/test.h5` + `test_masks.npz`).

| Datei | Checkpoint | trainierte Batches | ausgewertet bei |
|---|---|---|---|
| `loc_baseline.json` | `checkpoints/videomae_phase2.ckpt` | — | — |
| `loc_baseline_regions.json` | wie oben, mit `--per-region` | — | — |
| `loc_lambda0_control.json` | Sweep-Arm λ=0 | 6.000 | **6.000** |
| `loc_lambda002.json` | Sweep-Arm λ=0,02 | 6.000 | **3.000** |
| `loc_lambda01.json` | Sweep-Arm λ=0,1 | 6.000 | **3.000** |
| `loc_aux_head.json` | `experiment=train_video_loc_head` | 6.000 | **6.000** |

> **Die beiden λ-Arme sind bei Batch 3.000 ausgewertet, nicht bei 6.000.** Ursache und
> Tragweite stehen in §13.1 des Hauptdokuments: `save_top_k` hörte auf zu speichern,
> sobald `val/loss` stieg, und `last.ckpt` ist bitweise eine Kopie des letzten
> Speicherstands. Die Aussagen bleiben gültig (die λ-Arme schlagen die Kontrolle mit der
> halben Trainingsmenge), aber die Kurvenform in §13.4 mischt Lokalisierung aus Batch
> 3.000 mit Accuracy aus Batch 6.000. Behoben für künftige Läufe durch
> `save_top_k: -1`, abgesichert von `tests/test_checkpoint_config.py`.

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
