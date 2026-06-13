# Runbook — Vollständiger Trainingslauf via W&B Launch

Schritt-für-Schritt-Runbook, um die komplette Modell-Pipeline über die
`Desktop_PC`-Queue einzureihen: **Phase 1 → Phase 2 → Phase 4** plus den
SOTA-Performance-Track. Ergänzt [`launch.md`](launch.md) (Queue/Agent-Setup) und
[`phase34_runbook.md`](phase34_runbook.md) (Forschungsfragen-Zuordnung). Alle
Befehle sind PowerShell (Desktop-PC, Windows).

## Kernprinzipien

- **Ein Agent, `max_jobs: 1`** → Jobs laufen **strikt sequenziell in
  Queue-Reihenfolge (FIFO)**. Dadurch existiert der Checkpoint eines früher
  eingereihten Jobs auf der Platte, bevor ein späterer Job davon warm-startet.
- **Ein einziger `train.py`-Git-Job** wird für *alle* Läufe wiederverwendet;
  Experiment + Overrides kommen pro Einreihung über `overrides.args`.
- **Checkpoint-Linie** (Export-Name → Datei in `DEEPFAKE_CKPT_DIR`):
  - Phase 1: `videomae.ckpt`, `wav2vec2.ckpt`, `multimodal.ckpt`, `multimodal_concat.ckpt`
  - Phase 2: `videomae_phase2.ckpt`, `wav2vec2_phase2.ckpt`, `multimodal_phase2.ckpt`, `multimodal_concat_phase2.ckpt`
- **Phase-2-Configs warm-starten automatisch** von den Phase-1-Checkpoints
  (Pfade in den Experiment-Configs hinterlegt). `warmstart_ckpt` wird nur in den
  zwei unten markierten Fällen explizit übergeben.
- **Phase 4 trainiert hier nichts** — statt adversarialem Fine-Tuning werden die
  **Evaluations-Sweeps** aus [`phase34_runbook.md`](phase34_runbook.md) §2
  ausgeführt (lokale `scripts/eval_*.py`, *nicht* über die Launch-Queue), die einen
  eingefrorenen **Phase-2**-Checkpoint angreifen/bewerten.

## Reihenfolge (Abhängigkeiten)

```
Launch-Queue (Agent, FIFO):
  Phase 1 (4)  →  [ckpt prüfen]  →  Phase 2 (4)
                              └──→  LoRA-Arme (3, brauchen Phase-1-ckpts)
  balanced/mixup/robust/smoothing (9) — ohne Abhängigkeit, jederzeit einreihbar

Lokal (kein Launch, GPU-Box):
  Phase 4 — Adversarial-Eval-Sweeps  ←  eingefrorene Phase-2-ckpts
```

Empfehlung: **phasenweise** einreihen und zwischen den Phasen die Checkpoints
prüfen. Ein still fehlgeschlagener Phase-1-Job würde sonst jeden nachgelagerten
Warm-Start brechen. Phase 4 erst starten, wenn die Phase-2-Checkpoints
(`*_phase2.ckpt`) vorliegen.

---

## 0. Einmalige Vorbereitung (Desktop-PC, GPU)

```powershell
# Env-Variablen VOR dem Agentenstart (jeder Job erbt sie via os.environ.copy())
$env:DEEPFAKE_DATA_DIR = "D:/DeepfakeProjekt/Belegarbeit/deepfake-detection/data/"
$env:DEEPFAKE_LOG_DIR  = "D:/DeepfakeProjekt/Belegarbeit/deepfake-detection/logs/"
$env:DEEPFAKE_CKPT_DIR = "D:/DeepfakeProjekt/Belegarbeit/deepfake-detection/checkpoints/"

# Windows-Agent starten (blockiert das Terminal, pollt die Queue)
python launch/agent_windows.py -e christian-debbertin-deepfake-detection -q Desktop_PC -c launch/launch-config.yaml
```

## 1. Job einmalig anlegen (zweites Terminal, beliebiger Rechner)

```powershell
wandb job create git https://github.com/christiand03/deepfake-detection.git `
  --entry-point "python src/train.py experiment=train_video" `
  --entity christian-debbertin-deepfake-detection `
  --project "Deepfake Detection" `
  --name train-deepfake
```

Danach die wiederverwendbaren Variablen setzen (für alle Queue-Befehle):

```powershell
$JOB    = "christian-debbertin-deepfake-detection/Deepfake Detection/train-deepfake:latest"
$COMMON = @("-q","Desktop_PC","-e","christian-debbertin-deepfake-detection","-p","Deepfake Detection")
```

> `$COMMON` wird als **Array** (`@COMMON`) gesplattet, nicht als String —
> PowerShell zerlegt String-Variablen *nicht* in einzelne Argumente.

---

## 2. Phase 1 — Backbones/Köpfe trainieren (erzeugt die Warm-Start-Checkpoints)

```powershell
wandb launch -j $JOB @COMMON --config '{\"overrides\": {\"args\": [\"experiment=train_video\"]}}'
wandb launch -j $JOB @COMMON --config '{\"overrides\": {\"args\": [\"experiment=train_audio\"]}}'
wandb launch -j $JOB @COMMON --config '{\"overrides\": {\"args\": [\"experiment=train_multimodal\"]}}'
wandb launch -j $JOB @COMMON --config '{\"overrides\": {\"args\": [\"experiment=train_multimodal_concat\"]}}'
```

**Prüfen vor Phase 2** — alle vier Exporte vorhanden?

```powershell
Get-ChildItem $env:DEEPFAKE_CKPT_DIR -Filter *.ckpt | Select-Object Name,Length
# erwartet: videomae.ckpt, wav2vec2.ckpt, multimodal.ckpt, multimodal_concat.ckpt
```

## 3. Phase 2 — End-to-End-Finetuning (Warm-Start von den besten Phase-1-ckpts)

```powershell
wandb launch -j $JOB @COMMON --config '{\"overrides\": {\"args\": [\"experiment=train_video_phase2\"]}}'
wandb launch -j $JOB @COMMON --config '{\"overrides\": {\"args\": [\"experiment=train_audio_phase2\"]}}'
wandb launch -j $JOB @COMMON --config '{\"overrides\": {\"args\": [\"experiment=train_multimodal_phase2\"]}}'
# Concat-Arm: EIGENE Linie — warm-startet vom Phase-1-Concat-ckpt, eigener Export-Name.
wandb launch -j $JOB @COMMON --config '{\"overrides\": {\"args\": [\"experiment=train_multimodal_phase2\", \"model.fusion_mode=concat\", \"warmstart_ckpt=${paths.export_dir}/multimodal_concat.ckpt\", \"ckpt_export_name=multimodal_concat_phase2\"]}}'
```

> **Concat Phase 2 (bewusste Entscheidung):** Der Concat-Arm bekommt einen
> **eigenen Checkpoint** (`multimodal_concat_phase2.ckpt`) und warm-startet von
> der **Phase-1-Concat-Linie** (`multimodal_concat.ckpt`) — *nicht* von
> `multimodal.ckpt`. So entsteht ein durchgängig P1→P2 trainiertes Concat-Modell,
> direkt vergleichbar mit dem Cross-Attention-Modell, statt einer reinen
> Ablation. Das `ckpt_export_name`-Override verhindert, dass der Fusion-Arm
> (`multimodal_phase2.ckpt`) überschrieben wird.

## 4. Phase 4 — Adversarial-Evaluation (KEIN Training)

Phase 4 trainiert hier **nichts**. Statt eines adversarialen Fine-Tunings werden
die **Evaluations-Sweeps** aus [`phase34_runbook.md`](phase34_runbook.md) §2
ausgeführt: Sie laden einen *eingefrorenen* Phase-2-Checkpoint und greifen nur den
*Input* an (FGSM/PGD/UAP-Gradienten bzgl. des Inputs; die Modellgewichte bleiben
unverändert).

> **Wichtig — nicht über die Launch-Queue:** Diese Sweeps sind keine
> `train.py`-Jobs, sondern lokale Skripte (`python scripts/eval_*.py`). Sie laufen
> direkt auf dem Desktop-PC (GPU), nicht über `wandb launch`/den Agenten. Sie lesen
> den Checkpoint über Umgebungsvariablen — hier die **Phase-2**-Modelle.

```powershell
# Eingefrorene Phase-2-Checkpoints setzen
$env:VIDEOMAE_CKPT_PATH   = "checkpoints/videomae_phase2.ckpt"
$env:WAV2VEC2_CKPT_PATH   = "checkpoints/wav2vec2_phase2.ckpt"
$env:MULTIMODAL_CKPT_PATH = "checkpoints/multimodal_phase2.ckpt"

# Dry-Run zuerst (Verdrahtung prüfen, ~2 Videos)
python scripts/eval_adversarial_sweep.py --multimodal --attack-modalities both --max-videos 2 --epsilon-grid 0.03 --methods FGSM

# Voller Adversarial-Sweep
python scripts/eval_adversarial_sweep.py                                       # Video-only FGSM + PGD über ε-Grid
python scripts/eval_adversarial_sweep.py --multimodal --attack-modalities audio
python scripts/eval_adversarial_sweep.py --multimodal --attack-modalities video
python scripts/eval_adversarial_sweep.py --multimodal --attack-modalities both

# UAP (Universal Adversarial Perturbation)
python scripts/compute_uap.py --modality video --target-class REAL
python scripts/compute_uap.py --modality multimodal --attack-modalities both --target-class REAL
```

Ergebnisse landen in den W&B-Tables `adversarial_sweep_results` bzw. „UAP
Transfer". Spalten-zu-Forschungsfrage-Zuordnung und der optionale
Verteidigungs-Arm (RQ4, gegen `*_adv.ckpt`) stehen in
[`phase34_runbook.md`](phase34_runbook.md) §2.

> Kein separater Audio-only-Adversarial-Lauf — die Modalitäts-Anfälligkeit
> (Audio vs. Video) deckt der Multimodal-Sweep über `--attack-modalities` ab.

---

## 5. SOTA-Performance-Track (unabhängig; LoRA-Arme brauchen Phase-1-ckpts)

```powershell
# Video
wandb launch -j $JOB @COMMON --config '{\"overrides\": {\"args\": [\"experiment=train_video_balanced\"]}}'
wandb launch -j $JOB @COMMON --config '{\"overrides\": {\"args\": [\"experiment=train_video_mixup\"]}}'
wandb launch -j $JOB @COMMON --config '{\"overrides\": {\"args\": [\"experiment=train_video_robust\"]}}'
wandb launch -j $JOB @COMMON --config '{\"overrides\": {\"args\": [\"experiment=train_video_phase2_lora\"]}}'        # braucht videomae.ckpt

# Audio (Wav2Vec2 hat kein Mixup → smoothing statt mixup)
wandb launch -j $JOB @COMMON --config '{\"overrides\": {\"args\": [\"experiment=train_audio_balanced\"]}}'
wandb launch -j $JOB @COMMON --config '{\"overrides\": {\"args\": [\"experiment=train_audio_smoothing\"]}}'
wandb launch -j $JOB @COMMON --config '{\"overrides\": {\"args\": [\"experiment=train_audio_robust\"]}}'
wandb launch -j $JOB @COMMON --config '{\"overrides\": {\"args\": [\"experiment=train_audio_phase2_lora\"]}}'         # braucht wav2vec2.ckpt

# Multimodal
wandb launch -j $JOB @COMMON --config '{\"overrides\": {\"args\": [\"experiment=train_multimodal_balanced\"]}}'
wandb launch -j $JOB @COMMON --config '{\"overrides\": {\"args\": [\"experiment=train_multimodal_mixup\"]}}'
wandb launch -j $JOB @COMMON --config '{\"overrides\": {\"args\": [\"experiment=train_multimodal_robust\"]}}'
wandb launch -j $JOB @COMMON --config '{\"overrides\": {\"args\": [\"experiment=train_multimodal_phase2_lora\"]}}'    # braucht multimodal.ckpt
```

Die LoRA-Arme (`*_phase2_lora`) warm-starten vom **besten Phase-1-Checkpoint** —
der Pfad (`${paths.export_dir}/{videomae,wav2vec2,multimodal}.ckpt`) ist bereits in
den Configs hinterlegt (wie bei den Phase-2-Vollläufen), daher **kein**
`warmstart_ckpt`-Override nötig. `train.py` exportiert unter diesen Namen jeweils
den besten Checkpoint (höchste `val/auc_video`, `mode: max` — s.
`configs/callbacks/default.yaml`). Erst **nach** abgeschlossener Phase 1 einreihen.

## 6. Optional — Phase 3 (Robustheit / Social-Media-Simulation)

Phase 3 entspricht den `*_robust`-Experimenten aus §5 (`train_video_robust`,
`train_audio_robust`, `train_multimodal_robust`) — Kompression / Rauschen /
Framerate-Drop-Augmentation. Es gibt keine separaten Configs, daher nichts
darüber hinaus einzureihen.

---

## 7. Überwachen

- **Queue/Agent:** wandb.ai → Entity → Launch → Queue `Desktop_PC`
  (queued / running / finished).
- **Run-Metriken:** Projekt **"Deepfake Detection"** (Loss, Accuracy, F1, AUC,
  LRP-Grids). `train.py` testet nach jedem Lauf automatisch auf dem Test-Split.
- **Agent-Logs:** Agent-Terminal bzw. `wandb/debug.log`.

## Checkpoint-Übersicht

| Phase | Experiment | Warm-Start von | Export-Datei |
|---|---|---|---|
| 1 | `train_video` | — | `videomae.ckpt` |
| 1 | `train_audio` | — | `wav2vec2.ckpt` |
| 1 | `train_multimodal` | — | `multimodal.ckpt` |
| 1 | `train_multimodal_concat` | — | `multimodal_concat.ckpt` |
| 2 | `train_video_phase2` | `videomae.ckpt` | `videomae_phase2.ckpt` |
| 2 | `train_audio_phase2` | `wav2vec2.ckpt` | `wav2vec2_phase2.ckpt` |
| 2 | `train_multimodal_phase2` | `multimodal.ckpt` | `multimodal_phase2.ckpt` |
| 2 | `train_multimodal_phase2` (concat) | `multimodal_concat.ckpt` | `multimodal_concat_phase2.ckpt` |
| 4 | `eval_adversarial_sweep.py` (Eval) | `*_phase2.ckpt` (eingefroren) | — (W&B-Table, kein ckpt) |
| 4 | `compute_uap.py` (Eval) | `*_phase2.ckpt` (eingefroren) | — (W&B-Table, kein ckpt) |
</content>
</invoke>
