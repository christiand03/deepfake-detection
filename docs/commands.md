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

**Video-Modell (VideoMAE):**

```bash
python src/train.py experiment=train_video
```

**Audio-Modell (Wav2Vec2):**

```bash
python src/train.py experiment=train_audio
```

Checkpoints werden automatisch in `logs/train/runs/<timestamp>/checkpoints/` gespeichert.

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

## 7. Gesamtreihenfolge (Kurzübersicht)

| Schritt | Befehl |
|---------|--------|
| 1. Preprocessing (200 Videos) | `python -m src.data_processing.preprocess run.max_videos=200` |
| 2. Checkpoint ablegen | `mkdir checkpoints` → Datei hineinkopieren |
| 3. Evaluation Video | `python src/eval.py experiment=train_video ckpt_path=checkpoints/videomae_colleague.ckpt` |
| 4. Evaluation Audio | `python src/eval.py experiment=train_audio ckpt_path=checkpoints/wav2vec2.ckpt` |
| 5. xAI Video | `python src/explain.py experiment=train_video ckpt_path=checkpoints/videomae_colleague.ckpt` |
| 6. xAI Audio | `python src/explain_audio.py experiment=train_audio ckpt_path=checkpoints/wav2vec2.ckpt` |

---

## Weiterführende Recherche

- Hydra Overrides: https://hydra.cc/docs/advanced/override_grammar/basics/
- PyTorch Lightning Checkpoints: https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html
- Konfigurationsstruktur: Siehe `configs/` und `conf/preprocess.yaml`
- xAI-Methoden und Visualisierungsstandards: Siehe `docs/xai.md`
- Modellarchitekturen: Siehe `docs/model.md`
