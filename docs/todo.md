# 🎯 Die SOTA Aktions- und To-Do Liste

Diese To-Do-Liste bietet einen konkreten Zeitplan für das Projekt "Unmasking Deception" (900 h).

## Meilenstein 1: Projekt-Fundament & Tools (Tag 1-7)
- [x] **Repository aufsetzen:** Git-Repo mit *Cookiecutter Data Science* Ordnerstruktur generieren (Phase 1).
- [x] **Environment sichern:** Dockerfile oder VS Code DevContainer inkl. lokaler GPU-Treiber-Schnittstellen konfigurieren und verifizieren.
- [x] **Formatting (Ruff) forcieren:** Linter & automatische Formatierung in IDE integrieren und `.pre-commit-config.yaml` einrichten.
- [x] **Paper-Setup:** Projekt in *Overleaf* einrichten (IEEE Standard) und *Zotero* für automatische Quellen einbinden.
- [ ] DVC einrichten

## Meilenstein 2: Datasets & Daten Pipeline (Woche 2-4)
- [ ] **Download:** Beschaffung der Rohmaterialien (FakeAVCeleb, WLD, PDD Testset).
- [ ] **Modulare Dataloader (Pytest):** Eine robuste Video/Audio-Klasse programmieren.
- [ ] **FFmpeg Synchronisation:** Audio auf Mono, 16kHz resampeln, Video fps fixieren.
- [ ] **Kontext-Aware Cropping (MediaPipe):** Gesichtsextraktion (Faktor 1.4x), Splitten zu 16-Frame Chunks.
- [ ] **Speicherung:** Alles als performante `.h5` oder `.pt` Archive serialisieren – DVC für diese großen Dateien nutzen.
- [ ] **Sanity-Check Test:** Rekonstruktion eines `.h5` Tensors zu `.mp4` inkl. Audio – Visueller Check. *(Niemals ohne Pass starten!)*

## Meilenstein 3: Architektur & MLOps Framework (Monat 2)
- [ ] **Konfiguration (Hydra):** Strukturieren einer sauberen `config.yaml` um Hardcoding zu stoppen.
- [ ] **Weights & Biases init:** WandB Account verlinken, Project anlegen, `WandBLogger` in PyTorch Lightning einfangen.
- [ ] **Baseline (ISTVT / Wav2Vec 2.0):** Integration der frozen Feature-Extractor Backbones (Achtung auf `einops` und `jaxtyping`).
- [ ] **Overfit-Test:** Prüfen ob der ISTVT auf 1 Batch konvergieren kann.
- [ ] DVC Pipeline einrichten

## Meilenstein 4: Fusion & xAI Visualisierung (Monat 3)
- [ ] **Cross-Attention Head (Pair Programming):** Beide Personen programmieren den Fusions-Head, der Features verknüpft. Training auf Gesamt-Dataset.
- [ ] **Ablation Studies implementieren:** Was passiert, wenn Fusion nur "Concatenate" ist?
- [ ] **LRP & Attention Rollout:** Einhaken der xAI-Bibliotheken in den Attention Layer.
- [ ] **Plot_Style verfeinern:** SciencePlots design, Grid-Builder für die Ausgabe in W&B integrieren.

## Meilenstein 5: Stretch-Goals (Robustness & Frontend) (Monat 4)
- [ ] **Robustness Skripte:** Rausch/Framedrop Tests implementieren & ausführen.
- [ ] **Adversarial Attacks:** Foolbox einbinden, White-Box PGD nutzen, LRP Verschiebungen festhalten.
- [ ] *(Optional)* **FastAPI+React Start:** Kleinen Demo-Prototyp online bringen, in dem der Professor live ein Video evaluieren kann.

## Meilenstein 6: Akademische Schreibphase (Parallel ab Woche 1!)
- [ ] **Monat 1:** Einleitung, Problemstellung, Related Work.
- [ ] **Monat 2:** Methode Datensätze (Warum Cropping-Faktor? Welches Preprocessing? HDF5-Begründung).
- [ ] **Monat 3:** Methodik LRP, Transformer-Erklärung in LaTeX.
- [ ] **Monat 4:** Ergebnisse, Analyse der xAI-Attacken und Abstract finalisieren.
