# Runbook — Phase 3 & 4 Ergebnisse erzeugen

Die gesamte Phase-3/4-Infrastruktur ist code-seitig fertig (inkl. der *joint*-
multimodalen Sweeps). Was fehlt, ist das **Ausführen** der Sweeps auf den aktuellen
Daten (post-2026-06-11) und das Eintragen der W&B-Zahlen in
[`model.md`](model.md) §7.14 / §7.15. Dieses Runbook listet die exakten Befehle und
ordnet jede Ausgabespalte der zu beantwortenden Forschungsfrage zu.

> **Voraussetzungen:** trainierte Checkpoints unter `checkpoints/`, regenerierte
> Daten (`data/processed/test_metadata.csv` + `data/normalized/*.mp4`), GPU.
> Die Sweeps sind langläufig — pro Grid-Punkt wird das gesamte Testset re-enkodiert
> bzw. angegriffen.

## 0. Checkpoints setzen (PowerShell)

Phase 3 und 4 trainieren **nichts** — sie laden einen eingefrorenen Checkpoint und
*bewerten* ihn. (Die FGSM/PGD/UAP-Angriffe berechnen Gradienten nur bezüglich des
*Inputs*, um eine Perturbation zu erzeugen; die Modellgewichte bleiben unverändert.)
Verwendet werden die **besten Phase-1/2-Checkpoints** — die end-to-end-feingetunten
Phase-2-Modelle (`*_phase2.ckpt`, siehe [`model.md`](model.md) §7.11):

```powershell
$env:VIDEOMAE_CKPT_PATH   = "checkpoints/videomae_phase2.ckpt"
$env:WAV2VEC2_CKPT_PATH   = "checkpoints/wav2vec2_phase2.ckpt"
$env:MULTIMODAL_CKPT_PATH = "checkpoints/multimodal_phase2.ckpt"
```

Vor jedem Volllauf empfiehlt sich ein **Dry-Run** mit `--max-videos 2`, um die
Verdrahtung zu prüfen (siehe Verifikation unten).

## 1. Phase 3 — Robustness (→ `model.md` §7.14)

```powershell
# Unimodal: Video-Branch (CRF×FPS) + Audio-Branch (Bitrate) + Upscale
python scripts/eval_robustness_sweep.py

# Multimodal: fusionierter Detektor unter JOINT Video+Audio-Degradation
python scripts/eval_robustness_sweep.py --multimodal --no-video-sweep --no-audio-sweep --no-upscale-sweep
```

**W&B-Table `sweep_results` → Forschungsfrage:**

| Spalte | beantwortet |
| --- | --- |
| `auc` über `crf`/`fps` (modality=video) | **Breaking Point** Video (RQ1) |
| `auc` über `audio_bitrate_kbps` (modality=audio) | Audio- vs. Video-Fragilität (RQ3) |
| `auc` (modality=multimodal) vs. video/audio | Robustheit der Fusion (RQ4) |
| `mean_fake_prob_delta`, `fooling_rate` | Stärke des Qualitätseinflusses |

**Attention-Shift (RQ2):** Region-Scores vor/nach Degradation kommen aus dem
interaktiven Lab (`POST /robustness` → `attentionShift`); für die Schriftfassung
mehrere CRF-Stufen an einem repräsentativen Fake-Clip vergleichen.

## 2. Phase 4 — Adversarial (→ `model.md` §7.15)

```powershell
# Video-only FGSM + PGD über das ε-Grid
python scripts/eval_adversarial_sweep.py

# Multimodal: Audio-only / Video-only / Joint
python scripts/eval_adversarial_sweep.py --multimodal --attack-modalities audio
python scripts/eval_adversarial_sweep.py --multimodal --attack-modalities video
python scripts/eval_adversarial_sweep.py --multimodal --attack-modalities both
```

**W&B-Table `adversarial_sweep_results` → Forschungsfrage:**

| Spalte | beantwortet |
| --- | --- |
| `fooling_rate` über `epsilon` | ε-Schwelle der deterministischen Täuschung (RQ1) |
| `mean_attention_shift` | LRP-Verschiebung Mund/Augen → Hintergrund (RQ2) |
| `fooling_rate` je `attack_modalities` | Audio- vs. Video-Anfälligkeit (RQ3) |

### 2a. UAP (Universal Adversarial Perturbation)

```powershell
python scripts/compute_uap.py --modality video --target-class REAL
python scripts/compute_uap.py --modality multimodal --attack-modalities both --target-class REAL
```

W&B-Table „UAP Transfer": Baseline-AUC vs. Transfer-AUC + Fooling Rate.

### 2b. (Optional) Verteidigung: adversariales Fine-Tuning (RQ4)

> **Optional — nur falls die Verteidigungs-Frage RQ4 untersucht werden soll.**
> Dieser Schritt verwendet die **bereits vorhandenen** adversarial-trainierten
> Checkpoints (`videomae_adv.ckpt` / `multimodal_adv.ckpt`) — es wird hier **nicht**
> neu trainiert, nur evaluiert. Wer nur das Verhalten der besten Phase-1/2-Detektoren
> untersuchen will, überspringt diesen Schritt.

Den adversarialen Sweep gegen die adv-trainierten Checkpoints wiederholen und mit
der Baseline vergleichen (Clean-Accuracy darf nicht einbrechen, Fooling Rate soll
sinken):

```powershell
$env:VIDEOMAE_CKPT_PATH = "checkpoints/videomae_adv.ckpt"
python scripts/eval_adversarial_sweep.py --wandb-run-name adversarial-defense-video

$env:MULTIMODAL_CKPT_PATH = "checkpoints/multimodal_adv.ckpt"
python scripts/eval_adversarial_sweep.py --multimodal --attack-modalities both --wandb-run-name adversarial-defense-mm
```

## 3. Verifikation (vor den Vollläufen)

```powershell
# Code-Pfade ohne Wissenschaft prüfen (klein, braucht Checkpoints)
python scripts/eval_robustness_sweep.py --multimodal --no-video-sweep --no-audio-sweep --no-upscale-sweep --max-videos 2 --crf-grid 28 --fps-grid 25
python scripts/eval_adversarial_sweep.py --multimodal --attack-modalities both --max-videos 2 --epsilon-grid 0.03 --methods FGSM
```

Beide müssen ≥1 Zeile in die W&B-Summary-Table loggen und mit Exit-Code 0 enden.
Die Smoke-Tests (`pytest tests/test_robustness_sweep.py tests/test_adversarial_sweep.py`)
prüfen die Sweep-Logik ohne Checkpoints.
