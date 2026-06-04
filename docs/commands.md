# Befehls-Referenz: Von Rohdaten zur xAI-Analyse

Alle Befehle werden vom **Projektstamm** aus ausgeführt (`deepfake-detection/`).
Voraussetzung: Python-Umgebung aktiviert, Abhängigkeiten aus `requirements.txt` installiert.

---

## 1. Umgebung einrichten

```bash
# Abhängigkeiten installieren
pip install -r requirements.txt
pip install -r requirements-dev.txt
```
Zusätzlich muss von folgender Website: https://www.gyan.dev/ffmpeg/builds/ -> ffmpeg-release-full-shared.7z
heruntergeladen werden, entpackt werden und der darin enthaltene bin/ Ordner zu PATH hinzugefügt werden.

---

## 2. Preprocessing – Rohdaten → HDF5

Das Skript liest alle `.mp4`-Dateien unter `data/train/`, extrahiert Gesichts-Crops
(16 Frames je Chunk) und die zugehörigen Audio-Segmente und schreibt sie in
`data/processed/{train,val,test}.h5`.

**Vollständiger Lauf (alle Videos):**

```bash
python -m src.data_processing.preprocess
```

**Begrenzter Lauf – nur N Videos (empfohlen bei wenig Speicher):**

```bash
# Nur 200 Videos verarbeiten
python -m src.data_processing.preprocess run.max_videos=200
```

Der Parameter `run.max_videos` ist in `conf/preprocess.yaml` definiert.
Der Lauf ist resumierbar: bereits verarbeitete Videos werden dank
`run.skip_existing=true` bei erneutem Ausführen übersprungen.

**Ausgaben:**
- `data/processed/train.h5`, `val.h5`, `test.h5`
- `data/processed/train_metadata.csv`, `val_metadata.csv`, `test_metadata.csv`

---

### clips.json erstellen

# All clips (all splits)
python scripts/build_clips_json.py

# Test split only
python scripts/build_clips_json.py --split test

# First N clips (quick frontend smoke-test)
python scripts/build_clips_json.py --limit 5

# Custom paths
python scripts/build_clips_json.py \
    --normalized-dir data/normalized \
    --output conf/clips.json

---

## 3. Checkpoint eines vortrainierten Modells einbinden

Eine bereitgestellte Checkpoint-File (`.ckpt`) wird **nicht**
in eine spezielle Konfigurationsdatei eingetragen. Es wird stattdessen beim Aufruf
per Hydra-Override übergeben.

**Checkpoint ablegen (empfohlener Speicherort):**

```
deepfake-detection/
└── checkpoints/
    ├── videomae.ckpt   ← VideoMAE-Checkpoint
    └── wav2vec2.ckpt             ← Wav2Vec2-Checkpoint
```

Der Ordner `checkpoints/` muss ggf. manuell angelegt werden:

```bash
mkdir checkpoints
# Datei danach dorthin verschieben oder direkt hineinladen
```

---

## 4. Training (optional – falls kein fertiger Checkpoint vorliegt)

> **Phase 1 / Phase 2 (einheitlich für alle Modelle):** Alle Modelle starten standardmäßig in
> **Phase 1** (`freeze_backbone=true`): Backbone eingefroren, nur der Kopf wird trainiert.
> **Phase 2** (End-to-End-Finetuning) ist optional und für jedes Modell gleich:
> `model.freeze_backbone=false` + `warmstart_ckpt=<phase1.ckpt>` (siehe unten und `docs/model.md` §7.6).

**Video-Modell (VideoMAE) – Phase 1 (Backbone eingefroren, nur Kopf):**

```bash
python src/train.py experiment=train_video
```

**Audio-Modell (Wav2Vec2) – Phase 1 (Backbone eingefroren, nur Kopf):**

```bash
python src/train.py experiment=train_audio
```

**Video/Audio – Phase 2 (End-to-End-Finetuning, Warm-Start):**

> Phase 2 trainiert den Backbone → der große Phase-1-Default-Batch muss heruntergesetzt werden
> (`data.batch_size=2` für Video; Audio ist klein und verträgt mehr).

```bash
python src/train.py experiment=train_video \
    model.freeze_backbone=false warmstart_ckpt=checkpoints/videomae.ckpt \
    data.batch_size=2
# analog: experiment=train_audio ... warmstart_ckpt=checkpoints/wav2vec2.ckpt
```

**Multimodal-Modell – Phase 1 (Backbones eingefroren, nur Fusion-Head):**

```bash
python src/train.py experiment=train_multimodal
```

**Multimodal-Modell – Phase 2 (End-to-End-Finetuning, Warm-Start vom Phase-1-Checkpoint):**

Phase 2 entfriert beide Backbones (`freeze_backbone=false`) und trainiert end-to-end mit
niedrigerer LR, ausgehend von einem Phase-1-Checkpoint. **`warmstart_ckpt` lädt nur die
Gewichte** (frischer Optimizer/LR/Epoch-Zähler) — im Gegensatz zu `ckpt_path`, das ein
**volles Lightning-Resume** ist (stellt alten Optimizer/LR/Epoch wieder her und ignoriert die
LR-Override). Beide schließen sich gegenseitig aus. Auf der 8-GB-GPU / 16-GB-RAM-Box zusätzlich
`data.batch_size=1` (Host-RAM, siehe `docs/model.md` §6.6):

```bash
python src/train.py experiment=train_multimodal \
    model.freeze_backbone=false model.optimizer.lr=1e-5 \
    data.batch_size=1 warmstart_ckpt=checkpoints/multimodal.ckpt
```

**Fusions-Ablation (Multimodal — belegt, dass die Cross-Attention die Leistung treibt):**

> Alle drei sind volle Phase-1-Läufe (~5,5 h). `test/auc` + `test/ap` gegen die Cross-Attention-
> Baseline (`train_multimodal`, **test/auc 0,775**) vergleichen. Mechanismus: `model.fusion_mode`
> (siehe `docs/model.md` §4). Per-CLI auch `experiment=train_multimodal model.fusion_mode=concat`.

```bash
python src/train.py experiment=train_multimodal_concat        # ohne Cross-Attention (nur Concat)
python src/train.py experiment=train_multimodal_video_only    # Audio genullt
python src/train.py experiment=train_multimodal_audio_only    # Video genullt
```

**Adversariale Varianten (Phase 4.2 – PGD-augmentiertes Training):**

> Phase 2 (`freeze_backbone=false`) — die Configs setzen den Batch selbst herunter
> (Video bs 2, Multimodal bs 1; siehe `docs/model.md` §7.7/§7.8).

```bash
python src/train.py experiment=train_video_adversarial
python src/train.py experiment=train_multimodal_adversarial
```

**Training fortsetzen (echtes Resume statt Warm-Start):**

```bash
# Stellt Gewichte + Optimizer + LR-Scheduler + Epoch-Zähler wieder her
python src/train.py experiment=train_multimodal ckpt_path=logs/train/runs/<timestamp>/checkpoints/last.ckpt
```

Checkpoints werden automatisch in `logs/train/runs/<timestamp>/checkpoints/` gespeichert.

**Automatischer Export für die API/Frontend:** Nach jedem Training wird der beste
Checkpoint (niedrigster `val/loss`) zusätzlich an einen stabilen Pfad kopiert –
standardmäßig `checkpoints/<name>.ckpt` (`videomae`, `wav2vec2`, `multimodal`
bzw. `videomae_adv`/`multimodal_adv` für die Adversarial-Varianten). Damit zeigen
die API-Umgebungsvariablen (`VIDEOMAE_CKPT_PATH`, `WAV2VEC2_CKPT_PATH`,
`MULTIMODAL_CKPT_PATH`) immer auf eine vorhersagbare Datei, ohne den
zeitgestempelten Run-Ordner durchsuchen zu müssen.

- Deaktivieren: `python src/train.py experiment=train_video export_ckpt=false`
- Anderen Namen wählen: `... ckpt_export_name=mein_modell`
- Zielordner ändern: Umgebungsvariable `DEEPFAKE_CKPT_DIR` (absolut) setzen –
  wichtig für W&B-Launch-Läufe, die in einem temporären Klon laufen (siehe
  `docs/launch.md`).

---

## 5. Evaluation auf dem Test-Set

**VideoMAE:**

```bash
python src/eval.py experiment=train_video \
    data=deepfake_video \
    model=videomae \
    ckpt_path=checkpoints/videomae_colleague.ckpt
```

**Wav2Vec2:**

```bash
python src/eval.py experiment=train_audio \
    data=deepfake_audio \
    model=wav2vec2 \
    ckpt_path=checkpoints/wav2vec2.ckpt
```

---

## 6. xAI-Visualisierungen erzeugen

### 6.1 Video – AttnLRP-Heatmap (VideoMAE)

```bash
python src/explain.py experiment=train_video \
    ckpt_path=checkpoints/videomae_colleague.ckpt
```

**Optionale Overrides:**

```bash
# Anderen Frame visualisieren (0–15)
python src/explain.py experiment=train_video \
    ckpt_path=checkpoints/videomae_colleague.ckpt \
    explain.frame_idx=4

# Ausgabedatei umbenennen
python src/explain.py experiment=train_video \
    ckpt_path=checkpoints/videomae_colleague.ckpt \
    explain.save_path=outputs/my_heatmap.png

# Bestimmte Klasse erklären (0=Real, 1=Fake) statt vorhergesagter Klasse
python src/explain.py experiment=train_video \
    ckpt_path=checkpoints/videomae_colleague.ckpt \
    explain.target_class=1
```

**Ausgaben:**
- `lrp_explanation.png` (Standard) – enthält Originalframe, Heatmap, Überlagerung

### 6.2 Audio – LRP-Heatmap (Wav2Vec2)

```bash
python src/explain_audio.py experiment=train_audio \
    ckpt_path=checkpoints/wav2vec2.ckpt
```

**Optionale Overrides:**

```bash
# Layer 2 (Wort-Level via WhisperX) und Layer 3 (Frequenzbänder) deaktivieren
python src/explain_audio.py experiment=train_audio \
    ckpt_path=checkpoints/wav2vec2.ckpt \
    explain.enable_layer2=false \
    explain.enable_layer3=false

# Ausgabepfade anpassen
python src/explain_audio.py experiment=train_audio \
    ckpt_path=checkpoints/wav2vec2.ckpt \
    explain.save_path=outputs/audio_lrp.png \
    explain.layer2_save_path=outputs/audio_words.png \
    explain.layer3_save_path=outputs/audio_bands.png
```

**Ausgaben (Standard):**
- `audio_lrp_explanation.png` – LRP-Relevanz über die gesamte Wellenform (Layer 1)
- `audio_lrp_l2_words.png` – Aggregierte Relevanz pro erkanntem Wort (Layer 2, benötigt WhisperX)
- `audio_lrp_l3_bands.png` – Relevanz aufgeschlüsselt nach Frequenzbändern (Layer 3)

---

## 7. Offline-Sweeps (Phase 3 & 4)

Beide Skripte lesen das Testset aus `data/processed/test_metadata.csv` und laden
die rohen MP4-Dateien aus `data/normalized/`. `VIDEOMAE_CKPT_PATH` muss gesetzt
sein; der Robustness-Sweep benötigt zusätzlich `WAV2VEC2_CKPT_PATH` für den
optionalen Audio-Branch.

### 7.1 Robustness-Sweep – Phase 3 (CRF × FPS + Audio-Bitrate)

```powershell
# Umgebungsvariablen setzen (PowerShell)
$env:VIDEOMAE_CKPT_PATH  = "checkpoints/epoch=2-step=837.ckpt"
$env:WAV2VEC2_CKPT_PATH  = "checkpoints/epoch=2-step=261.ckpt"  # optional
```

```bash
# Dry-run: 3 Videos, ein Grid-Punkt, kein Audio-Sweep
python scripts/eval_robustness_sweep.py \
    --max-videos 3 --crf-grid 28 --fps-grid 25 --no-audio-sweep

# Vollständiger Video-Sweep (CRF × FPS, kein Audio)
python scripts/eval_robustness_sweep.py --no-audio-sweep

# Vollständiger Sweep inkl. Audio-Bitrate-Sweep
python scripts/eval_robustness_sweep.py

# Eigenes Grid und W&B-Run-Name
python scripts/eval_robustness_sweep.py \
    --crf-grid 23 28 35 \
    --fps-grid 25 10 \
    --audio-bitrate-grid 128 32 \
    --wandb-run-name robustness-subset
```

```bash
# Nur Upscaling-Sweep (TikTok/WhatsApp-Simulation), kein CRF×FPS- oder Audio-Sweep
python scripts/eval_robustness_sweep.py \
    --no-video-sweep --no-audio-sweep

# Upscaling-Sweep mit eigenen festen Parametern (CRF 28, 15 FPS)
python scripts/eval_robustness_sweep.py \
    --no-video-sweep --no-audio-sweep \
    --fixed-crf-for-upscale 28 --fixed-fps-for-upscale 15

# Dry-run: 2 Videos, nur Upscaling
python scripts/eval_robustness_sweep.py \
    --max-videos 2 --no-video-sweep --no-audio-sweep

# Vollständiger Sweep inkl. Upscaling (CRF × FPS + Audio + Upscale)
python scripts/eval_robustness_sweep.py

# Upscaling-Sweep deaktivieren
python scripts/eval_robustness_sweep.py --no-upscale-sweep
```

**Hintergrund Upscaling-Simulation:** TikTok und WhatsApp re-enkodieren Videos
intern auf 360p und skalieren sie bilinear auf 720p hoch. Der Filter
`scale=640:360,scale=1280:720` bildet genau diesen Artefakt nach.

**Ausgabe:** W&B-Table `sweep_results` mit den Spalten
`modality`, `crf`, `fps`, `audio_bitrate_kbps`, `auc`, `accuracy`,
`fooling_rate`, `mean_fake_prob_delta`.
Mögliche `modality`-Werte: `video`, `audio`, `video_upscale`.

### 7.2 Adversarial-Sweep – Phase 4 (FGSM & PGD über ε-Grid)

```powershell
# Umgebungsvariable setzen (PowerShell)
$env:VIDEOMAE_CKPT_PATH = "checkpoints/epoch=2-step=837.ckpt"
```

```bash
# Dry-run: 2 Videos, ε=0.03, nur FGSM
python scripts/eval_adversarial_sweep.py \
    --max-videos 2 --epsilon-grid 0.03 --methods FGSM

# Vollständiger Sweep (FGSM + PGD, Standard-ε-Grid)
python scripts/eval_adversarial_sweep.py

# Nur FGSM über das vollständige Testset
python scripts/eval_adversarial_sweep.py --methods FGSM

# Eigene Parameter
python scripts/eval_adversarial_sweep.py \
    --epsilon-grid 0.01 0.02 0.03 0.05 0.1 \
    --pgd-steps 20 \
    --methods FGSM PGD \
    --wandb-run-name adversarial-custom
```

**Ausgabe:** W&B-Table `adversarial_sweep_results` mit den Spalten
`method`, `epsilon`, `pgd_steps`, `n_clips`, `auc`, `accuracy`,
`fooling_rate`, `mean_fake_prob_delta`, `mean_attention_shift`.

---

## 8. Gesamtreihenfolge (Kurzübersicht)

| Schritt | Befehl |
|---------|--------|
| 1. Preprocessing (200 Videos) | `python -m src.data_processing.preprocess run.max_videos=200` |
| 2. Checkpoint ablegen | `mkdir checkpoints` → Datei hineinkopieren |
| 3. Evaluation Video | `python src/eval.py experiment=train_video ckpt_path=checkpoints/videomae_colleague.ckpt` |
| 4. Evaluation Audio | `python src/eval.py experiment=train_audio ckpt_path=checkpoints/wav2vec2.ckpt` |
| 5. xAI Video | `python src/explain.py experiment=train_video ckpt_path=checkpoints/videomae_colleague.ckpt` |
| 6. xAI Audio | `python src/explain_audio.py experiment=train_audio ckpt_path=checkpoints/wav2vec2.ckpt` |
| 7. Robustness-Sweep | `python scripts/eval_robustness_sweep.py --no-audio-sweep` |
| 7a. Upscaling-Sweep | `python scripts/eval_robustness_sweep.py --no-video-sweep --no-audio-sweep` |
| 8. Adversarial-Sweep | `python scripts/eval_adversarial_sweep.py` |

---

## 9. Backend-API und Frontend-Devserver starten

Für die interaktive Demo müssen **beide** Prozesse gleichzeitig laufen –
jeweils in einem eigenen Terminal.

### 8.1 FastAPI-Backend

Aus dem **Projektstamm** (`deepfake-detection/`) mit aktivierter Python-Umgebung:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Das Backend läuft auch **ohne** Modell-Checkpoints (diese werden dann nur
übersprungen). Der `/api/clips`-Endpunkt und die Video-Auslieferung unter
`/clips/` funktionieren sofort.

Umgebungsvariablen für Modell-Inferenz (PowerShell):

```powershell
# Wav2Vec2-Checkpoint (Epoch 2, Step 261 — aktuell verfügbar)
$env:WAV2VEC2_CKPT_PATH = "checkpoints/epoch=2-step=261.ckpt"

# VideoMAE-Checkpoint (sobald vorhanden)
$env:VIDEOMAE_CKPT_PATH = "checkpoints/epoch=2-step=837.ckpt"

uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### 8.2 React-Frontend (Vite-Devserver)

In einem **zweiten** Terminal, aus dem `frontend/`-Ordner:

```bash
npm run dev
```

Die App ist dann unter **http://localhost:5173** erreichbar.
Der Vite-Devserver leitet `/api/*` und `/clips/*` automatisch an das
Backend auf Port 8000 weiter.

**Mock-Modus deaktivieren** (Real-Backend verwenden, inkl. Audio-xAI):

Die Datei `frontend/.env.local` (bereits angelegt, nicht in Git) enthält:
```env
VITE_USE_MOCK=false
```
Damit ruft `analyzeClip()` das echte Backend auf. Ist `VITE_USE_MOCK` nicht gesetzt
oder `true`, werden synthetische Demo-Daten ohne Backend-Aufruf zurückgegeben.

---

## Weiterführende Recherche

- Hydra Overrides: https://hydra.cc/docs/advanced/override_grammar/basics/
- PyTorch Lightning Checkpoints: https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html
- Konfigurationsstruktur: Siehe `configs/` und `conf/preprocess.yaml`
- xAI-Methoden und Visualisierungsstandards: Siehe `docs/xai.md`
- Modellarchitekturen: Siehe `docs/model.md`
