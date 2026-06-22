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

**Paralleler Lauf (~3× schneller — empfohlen für volle Regenerierungen):**

```bash
python -m src.data_processing.preprocess run.num_workers=3
```

Worker-Prozesse übernehmen die Extraktion (FFmpeg/decord/MediaPipe); alles
HDF5/CSV-Schreiben bleibt im Hauptprozess (Single-Writer). `run.num_workers=0`
(Default) = sequenzieller Pfad. RAM-Budget beachten: ~1–2 GB je Worker auf der
16-GB-Box. Optional beschleunigt `face_extraction.running_mode=video` die
Gesichtserkennung (Tracking statt Per-Frame-Detection) — liefert leicht andere
Crops, daher **nur zusammen mit einer vollen Regenerierung** aktivieren und mit
`python -m scripts.validate_processed` prüfen (s. `docs/performance_roadmap.md` §1.6/§1.7).

**Ausgaben:**
- `data/processed/train.h5`, `val.h5`, `test.h5`
- `data/processed/train_metadata.csv`, `val_metadata.csv`, `test_metadata.csv`
- `data/normalized/{video_id}.mp4` — jede verarbeitete Quelle wird hier
  materialisiert (on-fps verlustfrei stream-kopiert, off-fps CRF-18-re-encodiert).
  Die Sweeps (`scripts/eval_*`) und die Demo-API lösen darüber die flache Quell-MP4 auf.

---

### `data/normalized/` nachfüllen (Datensätze vor der Stream-Copy-Policy)

Wurde ein Datensatz verarbeitet, als das Preprocessing 25-fps-Quellen noch *direkt*
las (leeres `data/normalized/`, Audit §1.1-Nachtrag), die flachen `{video_id}.mp4`
einmalig aus den Rohvideos nachziehen — kein Re-Processing, verlustfreie Copies:

```bash
python -m scripts.backfill_normalized                 # alle Splits (~10 GB Rohdaten)
python -m scripts.backfill_normalized --splits test   # nur das Testset (genügt für die Sweeps)
python -m scripts.backfill_normalized --dry-run       # nur berichten, nichts schreiben
```

Nicht auflösbare `video_id`s (kein passendes Rohvideo) führen zu Exit-Code ≠ 0.

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

### 4.0 Attention-Modus (SDPA ↔ Eager) — der gesamte Prozess

Das Projekt nutzt **zwei Attention-Implementierungen für zwei Aufgaben**. Der
Prozess von Training bis xAI-Erklärung läuft so ab:

1. **Training mit SDPA (Default).** Alle drei Modell-Configs setzen
   `attn_implementation: sdpa` (`configs/model/*.yaml`): fused Kernels, ~2,8×
   Durchsatz, die `O(N²)`-Attention-Matrix wird nie materialisiert — dadurch
   passt Phase-2-Video-Finetuning mit Batch 6 statt 2 (s. `docs/model.md` §6.4).

2. **Der Checkpoint ist implementierungs-unabhängig.** SDPA und Eager berechnen
   dieselbe Funktion mit denselben Gewichten; nur der Rechenweg unterscheidet
   sich. Ein SDPA-trainierter Checkpoint lädt unverändert in ein Eager-Modell
   (Paritätstest: `tests/test_attn_implementation.py`).

3. **Erklären immer mit Eager — automatisch.** AttnLRP patcht
   `eager_attention_forward` auf Modulebene; unter SDPA würde der Patch still
   umgangen und die Heatmaps wären falsch. Deshalb laden `explain.py`,
   `explain_audio.py`, `explain_multimodal.py` und alle drei API-Loader
   (`src/api/inference.py`) Checkpoints **immer** mit dem Override
   `attn_implementation="eager"` — hier ist nichts umzuschalten.

4. **Guard als letzte Sicherung.** Erreicht doch ein Nicht-Eager-Modell
   `explain()`, wirft `BaseDeepfakeModule._require_eager_attention` einen
   `RuntimeError` mit Reload-Anleitung, statt still unbrauchbare Relevanzen zu
   liefern.

**Modus pro Trainings-Lauf umschalten** (z. B. Repro alter Eager-Läufe; dann
gilt wieder die Eager-Batch-Obergrenze, Video Phase 2 = bs 2):

```bash
python src/train.py experiment=train_video model.attn_implementation=eager
```

Details und Messwerte: `docs/performance_roadmap.md` §1.8, `docs/model.md` §6.4.

### 4.1 Trainings-Läufe

> **Durchsatz/DataLoader-Profiling:** wie man misst, ob GPU oder DataLoader der
> Engpass ist (inkl. repräsentativem GPU-Lauf), steht in §4.2. `prefetch_factor`
> ist nach Messung bereits auf 4 gesetzt (Video/Multimodal) — Werte in
> `docs/performance_roadmap.md` §1.9.

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
> (`data.batch_size=6` für Video unter SDPA; unter `model.attn_implementation=eager` nur 2,
> s. §4.0. Audio ist klein und verträgt mehr). Einfacher: das fertige Experiment
> `experiment=train_video_phase2` nutzen (setzt Batch/LLRD/LR bereits korrekt).

```bash
python src/train.py experiment=train_video_phase2          # empfohlen (bs 6, LLRD, lr 1e-5)
# oder manuell:
python src/train.py experiment=train_video \
    model.freeze_backbone=false warmstart_ckpt=checkpoints/videomae.ckpt \
    data.batch_size=6
# analog: experiment=train_audio ... warmstart_ckpt=checkpoints/wav2vec2.ckpt
```

**Phase 2 mit LoRA statt Full-Finetuning (PEFT-Alternative):**

> Low-Rank-Adapter auf den Attention-Q/V-Projektionen; Basisgewichte bleiben
> eingefroren. Optimizer-States ~94M → <1M Parameter, weniger Overfitting. Der
> Checkpoint-Export **merged die Adapter automatisch zurück** — der exportierte
> Checkpoint ist ein plain Modell, API und `explain()` bleiben unverändert
> (s. `docs/performance_roadmap.md` §1.4).

```bash
python src/train.py experiment=train_video_phase2_lora
```

**Trainings-Ablationen / SOTA-Features (alle config-gated, Defaults unverändert):**

> Vergleichsmetrik jeweils `val/auc_video` gegen die Baseline `train_video`.
> Hintergründe: `docs/performance_roadmap.md` §1.

```bash
python src/train.py experiment=train_video_balanced   # Balanced Sampling statt CE-Gewicht 8,7
python src/train.py experiment=train_video_mixup      # Mixup + Label Smoothing (+ Sampler)
python src/train.py experiment=train_video_robust     # Social-Media-Augmentation (zahlt auf Phase 3 ein)
python src/train.py experiment=train_video callbacks=swa trainer.max_epochs=15  # SWA (ohne Early Stopping)
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

> Alle drei sind volle Phase-1-Läufe. `test/auc` + `test/ap` gegen die Cross-Attention-Baseline
> (`train_multimodal`) vergleichen. Mechanismus: `model.fusion_mode` (siehe `docs/model.md` §4/§7.10;
> 3. Lauf: Cross-Attention ≈ Concat in Phase 1 — eigentlicher Test in Phase 2). Per-CLI auch
> `experiment=train_multimodal model.fusion_mode=concat`.

```bash
python src/train.py experiment=train_multimodal_concat        # ohne Cross-Attention (nur Concat)
python src/train.py experiment=train_multimodal_video_only    # Audio genullt
python src/train.py experiment=train_multimodal_audio_only    # Video genullt
```

**Adversariale Varianten (Phase 4.2 – PGD-augmentiertes Training):**

> **Diese Experimente sind Phase 2** (`freeze_backbone=false`) — adversariales Finetuning auf einem
> *eingefrorenen* Backbone würde nur den Kopf härten. Daher **vom sauberen Phase-1-Checkpoint
> warm-starten** (`warmstart_ckpt`), nicht bei null beginnen. Die Configs setzen den Batch selbst
> herunter (Video bs 2, Multimodal bs 1; siehe `docs/model.md` §7.7). `train.py` testet nach dem
> Training automatisch auf dem aktuellen Test-Split.

```bash
# Warm-Start von den sauberen Phase-1-Baselines (3. Lauf); NICHT die alten *_adv.ckpt wiederverwenden.
python src/train.py experiment=train_video_adversarial `
    warmstart_ckpt=checkpoints/videomae.ckpt
python src/train.py experiment=train_multimodal_adversarial `
    warmstart_ckpt=checkpoints/multimodal.ckpt
```

> **Checkpoints prüfen:** `checkpoints/videomae.ckpt` / `multimodal.ckpt` müssen die aktuellen,
> leakage-bereinigten Phase-1-Modelle sein (Datei-Datum prüfen) — sonst auf den konkreten
> `logs/train/runs/<timestamp>/checkpoints/*.ckpt` zeigen.

**Robustheit messen (der eigentliche Phase-4.2-„Test"):** Nach dem adversarialen Training die
Fooling-Rate eines Baselines gegen das gehärtete Modell vergleichen — Adversarial-Sweep (§7.2):

```powershell
$env:VIDEOMAE_CKPT_PATH = "checkpoints/videomae_adv.ckpt"
python scripts/eval_adversarial_sweep.py
```

**Lauf pausieren und später fortsetzen (echtes Resume statt Warm-Start):**

Checkpoints werden automatisch in `logs/train/runs/<timestamp>/checkpoints/` gespeichert;
`save_last: true` (`configs/callbacks/default.yaml`) hält dort zusätzlich ein `last.ckpt`
aktuell. **Wichtig:** Geschrieben wird nur am **Epochen-Ende** (`check_val_every_n_epoch: 1`,
`every_n_train_steps`/`every_n_epochs`/`save_on_train_epoch_end` = `null` — kein Mid-Epoch-Save).
Lightning legt bei `Ctrl+C` **keinen** zusätzlichen Checkpoint an. Ein Abbruch mitten in der
Epoche setzt also beim letzten **abgeschlossenen** Epochen-Checkpoint wieder auf — der Fortschritt
der laufenden Epoche geht verloren. Bei langen Epochen daher möglichst bis zum nächsten
Epochen-Checkpoint warten (neue `last.ckpt`-Schreibzeit beobachten) oder vorab häufiger
checkpointen: `callbacks.model_checkpoint.every_n_train_steps=<N>`.

Sauber pausieren: **einmal** `Ctrl+C` (SIGINT) im Terminal des Laufs und das geordnete
Herunterfahren abwarten — **nicht** mehrfach `Ctrl+C` und **kein** `taskkill /F`, sonst kann ein
Hard-Kill während des Checkpoint-Schreibens `last.ckpt` beschädigen.

```bash
# Stellt Gewichte + Optimizer + LR-Scheduler + Epoch-Zähler wieder her.
# Gleiches experiment= wie beim Originallauf verwenden.
python src/train.py experiment=train_multimodal ckpt_path=logs/train/runs/<timestamp>/checkpoints/last.ckpt
```

> **Phase-2-Configs (`*_phase2`, `*_adversarial`) setzen `warmstart_ckpt`** — das kollidiert mit
> `ckpt_path` (Guard in `src/train.py`: „Set either warmstart_ckpt … or ckpt_path — not both"
> wirft). Beim Resume daher `warmstart_ckpt=null` mitgeben:
>
> ```bash
> python src/train.py experiment=train_multimodal_phase2 \
>     warmstart_ckpt=null \
>     ckpt_path=logs/train/runs/<timestamp>/checkpoints/last.ckpt
> ```

**Automatischer Export für die API/Frontend:** Nach jedem Training wird der beste
Checkpoint (höchste `val/auc_video`, `mode: max` — s. `configs/callbacks/default.yaml`)
zusätzlich an einen stabilen Pfad kopiert –
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

### 4.2 Durchsatz / DataLoader messen (Profiling)

Misst, ob die GPU oder der DataLoader der Engpass ist. Vergleiche in der
Profiler-Tabelle `[_TrainingEpochLoop].train_dataloader_next` (Warten auf Daten)
gegen `run_training_batch` (GPU-Compute). Dominiert das Warten → I/O-bound →
`data.prefetch_factor` (oder `data.num_workers`) anpassen (RAM-Regel + Messwerte:
`docs/performance_roadmap.md` §1.9).

> **Achtung Worker-Achse:** Mehr `num_workers` hilft NUR bei teurer Per-Item-
> Dekodierung (Video). Bei billiger Dekodierung + großem Batch (Audio: 128
> kleine Items) ist `num_workers=0` ~9× schneller als 4 — der Windows-Spawn-/
> IPC-Overhead übersteigt die Dekodierarbeit (§1.9). Daher beim Profiling immer
> auch `data.num_workers=0` gegentesten, nicht nur `prefetch_factor`.

**Schnell, aber NICHT repräsentativ** (`debug=profiler` erzwingt CPU +
`num_workers=0` — Compute dominiert dann immer, kein Prefetch messbar):

```bash
python src/train.py experiment=train_video debug=profiler
```

**Repräsentativ** (echter GPU-/SDPA-Pfad, konfigurierte `num_workers`/`prefetch_factor`,
auf wenige Batches begrenzt statt voller Epoche):

```bash
python src/train.py experiment=train_video \
    +trainer.profiler=simple \
    +trainer.limit_train_batches=40 +trainer.limit_val_batches=0 \
    trainer.max_epochs=1 test=false export_ckpt=false \
    callbacks=none logger=csv extras.enforce_tags=false
```

**Einen anderen `prefetch_factor` gegentesten** (gleiche Batch-Zahl für einen
fairen Vergleich; so wurde die §1.9-Entscheidung 2→4 getroffen):

```bash
python src/train.py experiment=train_video data.prefetch_factor=4 \
    +trainer.profiler=simple \
    +trainer.limit_train_batches=40 +trainer.limit_val_batches=0 \
    trainer.max_epochs=1 test=false export_ckpt=false \
    callbacks=none logger=csv extras.enforce_tags=false
```

> Hinweis: Beide Läufe tragen denselben einmaligen Worker-Spawn-Warmup (erster
> Batch), das **Total-Delta** von `train_dataloader_next` isoliert daher den
> reinen Prefetch-Effekt. Für Multimodal `experiment=train_multimodal` einsetzen.

**Worker-Achse gegentesten** (so wurde für Audio `num_workers=0` gefunden — bei
nw>0 viele Batches fahren, damit der Spawn amortisiert; nw=0 braucht nur wenige):

```bash
# synchron (kein Spawn/IPC) — wenige Batches reichen
python src/train.py experiment=train_audio data.num_workers=0 \
    +trainer.profiler=simple +trainer.limit_train_batches=20 +trainer.limit_val_batches=0 \
    trainer.max_epochs=1 test=false export_ckpt=false callbacks=none logger=csv extras.enforce_tags=false

# mit Workern — viele Batches, sonst dominiert der Spawn-Warmup die Mittelung
python src/train.py experiment=train_audio data.num_workers=2 \
    +trainer.profiler=simple +trainer.limit_train_batches=150 +trainer.limit_val_batches=0 \
    trainer.max_epochs=1 test=false export_ckpt=false callbacks=none logger=csv extras.enforce_tags=false
```

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

> **Attention-Modus:** Alle explain-Skripte laden den Checkpoint automatisch mit
> `attn_implementation="eager"` (AttnLRP-Voraussetzung) — auch SDPA-trainierte
> Checkpoints funktionieren unverändert (s. §4.0). **Hinweis für Skript-/CI-Läufe:**
> ohne Tags fragt das `extras`-Utility interaktiv nach — `extras.enforce_tags=false`
> anhängen, sonst hängt der Lauf.

### 6.1 Video – AttnLRP-Heatmap (VideoMAE)

```bash
python src/explain.py ckpt_path=checkpoints/videomae.ckpt extras.enforce_tags=false
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

```bash
# Multimodaler Sweep: fusionierter Detektor unter JOINT Video+Audio-Degradation
# (benötigt MULTIMODAL_CKPT_PATH). Standardmäßig läuft er zusätzlich zu den
# unimodalen Sweeps; mit --no-video-sweep/--no-audio-sweep isoliert ausführbar.
python scripts/eval_robustness_sweep.py \
    --multimodal --no-video-sweep --no-audio-sweep --no-upscale-sweep

# Eigene Audio-Bitrate für den multimodalen Sweep (Default 64 kbps)
python scripts/eval_robustness_sweep.py \
    --multimodal --no-video-sweep --no-audio-sweep --no-upscale-sweep \
    --fixed-audio-bitrate-for-mm 32
```

**Ausgabe:** W&B-Table `sweep_results` mit den Spalten
`modality`, `crf`, `fps`, `audio_bitrate_kbps`, `auc`, `accuracy`,
`fooling_rate`, `mean_fake_prob_delta`.
Mögliche `modality`-Werte: `video`, `audio`, `video_upscale`, `multimodal`.
Die `multimodal`-Zeilen degradieren Video (CRF/FPS) **und** Audio (AAC) in einem
Durchgang und bewerten den fusionierten Cross-Attention-Detektor.

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

```bash
# Multimodaler Angriff auf den fusionierten Detektor (benötigt MULTIMODAL_CKPT_PATH).
# --attack-modalities wählt, welche Modalität perturbiert wird: video | audio | both.
python scripts/eval_adversarial_sweep.py --multimodal --attack-modalities both
python scripts/eval_adversarial_sweep.py --multimodal --attack-modalities audio
python scripts/eval_adversarial_sweep.py --multimodal --attack-modalities video

# Eigenes Audio-Budget (sonst spiegelt es den jeweiligen ε-Wert des Grids)
python scripts/eval_adversarial_sweep.py \
    --multimodal --attack-modalities both --audio-epsilon 0.02
```

**Ausgabe:** W&B-Table `adversarial_sweep_results` mit den Spalten
`method`, `attack_modalities`, `epsilon`, `pgd_steps`, `n_clips`, `auc`,
`accuracy`, `fooling_rate`, `mean_fake_prob_delta`, `mean_attention_shift`.
Im video-only-Modus ist `attack_modalities` = `video`; im `--multimodal`-Modus
entspricht es der gewählten `--attack-modalities`. Ground-Truth ist `label_audio`
bei reinen Audio-Angriffen, sonst das kombinierte `label`.

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
| 7b. Robustness-Sweep multimodal | `python scripts/eval_robustness_sweep.py --multimodal --no-video-sweep --no-audio-sweep --no-upscale-sweep` |
| 8. Adversarial-Sweep | `python scripts/eval_adversarial_sweep.py` |
| 8a. Adversarial-Sweep multimodal | `python scripts/eval_adversarial_sweep.py --multimodal --attack-modalities both` |

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
- Performance-Features (SDPA, LoRA, Sampler, Augmentation, paralleles
  Preprocessing): Siehe `docs/performance_roadmap.md`
