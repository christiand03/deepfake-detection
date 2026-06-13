# 🎯 Die SOTA Aktions- und To-Do Liste

Diese To-Do-Liste bietet einen konkreten Zeitplan für das Projekt "Unmasking Deception" (900 h).

## Meilenstein 1: Projekt-Fundament & Tools (Tag 1-7)
- [x] **Repository aufsetzen:** Git-Repo mit *Cookiecutter Data Science* Ordnerstruktur generieren (Phase 1).
- [x] **Environment sichern:** Dockerfile oder VS Code DevContainer inkl. lokaler GPU-Treiber-Schnittstellen konfigurieren und verifizieren.
- [x] **Formatting (Ruff) forcieren:** Linter & automatische Formatierung in IDE integrieren und `.pre-commit-config.yaml` einrichten.
- [x] **Paper-Setup:** Projekt in *Overleaf* einrichten (IEEE Standard) und *Zotero* für automatische Quellen einbinden.
- [x] DVC einrichten

## Meilenstein 2: Datasets & Daten Pipeline (Woche 2-4)
- [x] **Download:** Beschaffung der Rohmaterialien (FakeAVCeleb, WLD, PDD Testset).
- [ ] **Modulare Dataloader (Pytest):** Eine robuste Video/Audio-Klasse programmieren.
- [ ] **FFmpeg Synchronisation:** Audio auf Mono, 16kHz resampeln, Video fps fixieren.
- [ ] **Kontext-Aware Cropping (MediaPipe):** Gesichtsextraktion (Faktor 1.4x), Splitten zu 16-Frame Chunks.
- [ ] **Speicherung:** Alles als performante `.h5` oder `.pt` Archive serialisieren – DVC für diese großen Dateien nutzen.
- [ ] **Sanity-Check Test:** Rekonstruktion eines `.h5` Tensors zu `.mp4` inkl. Audio – Visueller Check. *(Niemals ohne Pass starten!)*

## Meilenstein 3: Architektur & MLOps Framework (Monat 2)
- [x] **Konfiguration (Hydra):** Strukturieren einer sauberen `config.yaml` um Hardcoding zu stoppen.
- [x] **Weights & Biases init:** WandB Account verlinken, Project anlegen, `WandBLogger` in PyTorch Lightning einfangen.
- [ ] **Baseline (ISTVT / Wav2Vec 2.0):** Integration der frozen Feature-Extractor Backbones (Achtung auf `einops` und `jaxtyping`).
- [ ] **Overfit-Test:** Prüfen ob der ISTVT auf 1 Batch konvergieren kann.
- [ ] DVC Pipeline einrichten

## Meilenstein 4: Fusion & xAI Visualisierung (Monat 3)
- [ ] **Cross-Attention Head (Pair Programming):** Beide Personen programmieren den Fusions-Head, der Features verknüpft. Training auf Gesamt-Dataset.
- [ ] **Ablation Studies implementieren:** Was passiert, wenn Fusion nur "Concatenate" ist?
- [ ] **LRP & Attention Rollout:** Einhaken der xAI-Bibliotheken in den Attention Layer.
- [ ] **Plot_Style verfeinern:** SciencePlots design, Grid-Builder für die Ausgabe in W&B integrieren.

## Meilenstein 5: Phase 3 & 4 – Erweiterungen (Monat 4)

> Phase 1 & 2 sind abgeschlossen. Das Frontend ist vollständig implementiert (ursprünglich optional).
> Da wir dem Zeitplan voraus sind, werden Phase 3 & 4 über die ursprünglichen Ziele hinaus erweitert.

### Phase 3 – Robustness (Social Media Pipeline)

- [x] **Interaktives Robustness-Lab (Frontend):** H.264-CRF, FPS-Reduktion, Gaussian Noise via FFmpeg; FastAPI `/robustness`; RobustnessPanel mit Confidence-Delta und Breaking-Point-Anzeige.
- [ ] **Systematischer Robustness-Sweep → W&B:** Offline-Eval-Skript `scripts/eval_robustness_sweep.py`, das das gesamte Testset über ein Parameter-Grid auswertet (CRF ∈ {18,23,28,35,40,45,51} × FPS ∈ {25,15,10,5}) und AUC/Accuracy-Kurven in W&B loggt. Beantwortet direkt die Forschungsfrage nach dem Breaking Point.
- [ ] **Audio-Kompressions-Robustheit:** AAC/MP3-Reencoding bei niedrigen Bitraten (32 kbps) in `run_robustness_inference()` ergänzen. Testet Wav2Vec 2.0 unter realen Social-Media-Audiokompressionen (aktuell: `acodec=copy`).
- [ ] **xAI Attention-Shift unter Degradation:** Phase-3-Ergebnis um `attentionShift`-Liste (Anomalie-Region-Scores vor/nach Degradation) ergänzen – analog zu Phase 4. Beweist quantitativ die „trügerische Merkmale"-Hypothese.
- [ ] **Upscaling-Artefakt-Simulation:** FFmpeg-Filter `scale=640:360,scale=1280:720` (TikTok/WhatsApp-Reencoding) als vierten Degradations-Modus ergänzen.
- [ ] Ablation: crop_scale=1.0 vs 1.4 — Attention Maps vergleichen, Hypothese testen ob Kinn/Halsbereich diskriminativ ist.

### Phase 4 – Adversarial Attacks

- [x] **FGSM/PGD (L∞) implementiert:** Native PyTorch-Implementierung in `src/api/inference.py`; FastAPI `/adversarial`; AdversarialPanel mit Frame-Triptych und Attention-Shift-Tabelle.
- [ ] **Batch-Level Fooling Rate → W&B:** Offline-Eval-Skript `scripts/eval_adversarial_sweep.py`, das den Angriff bei ε ∈ {0.01, 0.02, 0.03, 0.05, 0.1} über das gesamte Testset ausführt und Fooling Rate + mittleren Confidence-Drop in W&B loggt (= „Adversarial Robustness Curve").
- [ ] **Multimodaler Adversarial Attack:** `run_adversarial_inference()` auf `MultimodalDeepfakeModule` erweitern – isolierter Angriff auf den Audio-Branch (Perturbation der Wellenform, für das Ohr unsichtbar) und gemeinsamer Audio+Video-Angriff. Testet ob der Audio-Branch anfälliger ist.
- [ ] **Adversarial Fine-Tuning als Verteidigung:** Kurzes PGD-augmentiertes Fine-Tuning (PGD-Beispiele on-the-fly, mit sauberem Batch gemischt). Messung von Clean-Accuracy vs. Adversarial-Accuracy vor/nach Training.
- [ ] **Universal Adversarial Perturbation (UAP):** Clip-unabhängige Perturbation δ*, die die Fooling Rate über alle Clips maximiert. Zeigt systematische Schwächen in den spatio-temporalen Features.

## Meilenstein 6: Akademische Schreibphase (Parallel ab Woche 1!)
- [ ] **Monat 1:** Einleitung, Problemstellung, Related Work.
- [ ] **Monat 2:** Methode Datensätze (Warum Cropping-Faktor? Welches Preprocessing? HDF5-Begründung).
- [ ] **Monat 3:** Methodik LRP, Transformer-Erklärung in LaTeX.
- [ ] **Monat 4:** Ergebnisse, Analyse der xAI-Attacken und Abstract finalisieren.
