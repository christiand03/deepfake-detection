---
title: Diskussion & Limitationen (Belegarbeit) — Deutsch
type: writing/discussion
status: draft-grounded
language: de
created: 2026-07-05
updated: 2026-07-05
tags: [Writing, Discussion, Limitations, Deutsch, Belegarbeit]
---

# Diskussion & Limitationen

> [!info] Status — quellgestützter Entwurf
> Diskussion der abgeschlossenen Phasen 1–2 und der Diagnostik; die Phase-3/4-Diskussion bleibt offen, bis die Ergebnisse vorliegen. Grounded auf [[results-de]], dem Handout §8–§9, `docs/audit_2026-06.md` und [[research-question-card]]. Aussagen folgen der [[Claim Map]]; Überclaims sind vermieden. `\cite{}`-Schlüssel: [references.bib](../references.bib).

## 1. Diskussion der Phase-1/2-Befunde

**Unfreezing ist für Video entscheidend, für Audio nachrangig.** Der eingefrorene VideoMAE-Probe bleibt bei `auc_video` 0,730 und fittet nicht einmal die Trainingsdaten — die selbstüberwachten Kinetics-Features sind nicht linear separierbar für Video-Forgery. Erst das End-to-End-Feintuning hebt das Modell auf 0,999. Für Audio genügt bereits der eingefrorene Wav2Vec-2.0-Backbone (0,976), das Entfrieren bringt nur +0,021 — die vortrainierten Sprachrepräsentationen übertragen also direkt, die Video-Repräsentationen nicht. Das stützt die Design-Entscheidung, self-supervised vortrainierte Backbones einzusetzen (Gap G3), zeigt aber auch ihre modalitätsabhängige Grenze.

**Die clean-data-Aufgabe ist gesättigt — das eigentliche Signal liegt in Phase 3/4.** Mit `auc_video` 0,999 (Video) und 0,997 (Audio) ist die In-Distribution-Erkennung auf sauberen Daten nahe der Decke. Eine nahezu perfekte clean-AUC ist jedoch ein klassisches Anzeichen dafür, dass das Modell auf Generierungs-/Encoding-Artefakte statt auf robuste semantische Hinweise setzen könnte. Die clean-AUC ist damit nicht mehr diskriminativ; die belastbare Aussage über Vertrauenswürdigkeit entsteht erst unter Degradierung (Phase 3) und Angriff (Phase 4).

**Fusion ist über Manipulationstypen balanciert — der Cross-Attention-Mechanismus ist aber nicht robust belegt.** Die multimodale Fusion erkennt alle drei Manipulationstypen ausgewogen (visual 0,932 / audio 0,957 / both 0,988) und schlägt auf den aktuellen Daten die Concat-Variante auf allen acht Metriken. Diese Aussage ist jedoch **directional, nicht robust**: Die Ablation ist nicht parameter-matched (die inerten ~2,1 M Attention-Parameter der `concat`-Config), beruht auf einem einzelnen Seed mit kleinem Eval (4 val / 6 test Identitäten), der Fusions-Lauf überfittet (train 0,214 < val 0,343), und der frühere leakage-behaftete Lauf zeigte einen Gleichstand. Die zentrale Thesen-Aussage „Cross-Modal-Synchronisation ist notwendig" ist damit **noch nicht haltbar**; sie verlangt einen Phase-2-Fusionslauf (der Fusion gegen die *unfrozen* unimodalen 0,999 misst), eine parameter-matched Ablation und ≥3 Seeds.

## 2. Erklärbarkeit

**Der Detektor nutzt intra-chunk-temporale Ordnung, nicht nur räumliche Artefakte.** Die Frame-Perturbations-Diagnostik senkt die video-level AUROC des eingefrorenen Probes von 0,745 auf 0,597–0,691, sobald die Frame-Reihenfolge innerhalb der Chunks gemischt wird. Das ist ein positives Zeichen (das Modell liest Bewegung/Ordnung), aber auf den frozenen Probe beschränkt — für das entfrorene Phase-2-Modell steht die Diagnostik aus.

**Die bivariate Heatmap ist eine bewusste Rekombination, keine fundamental neue Methode.** Sie macht die treue Fake-vs-Real-Entscheidung durch die Entkopplung von Magnitude und kontrastiver Richtung legibel und ist wahrnehmungspsychologisch fundiert \cite{schloss2019colormap, schoenlein2026saturated}; der Neuheitsanspruch bleibt zurückhaltend (vgl. Oh \& Noh 2025, Walter 2025; [[AttnLRP Bivariate Heatmap]]). Entscheidend: Die **Treue** der Heatmaps ist bislang **nicht per Perturbationstest belegt** — plausibel ist nicht faithful \cite{adebayo2018sanity}. Genau diese Treue unter Angriff ist der Gegenstand der ausstehenden Phase 4 (RQ4b, Gap G4).

## 3. Limitationen

| # | Limitation | Ehrliche Einordnung |
|---|---|---|
| 1 | Nur 32 Identitäten, 4 in Val | Hochvariante Checkpoint-Selektion; Proof-of-Concept, keine Populationsschätzung. Limiter: `max_videos=12000`-Plattencap |
| 2 | Generator-Leakage möglich (gleiche TTS/Face-Swap in allen Splits) | In-Dataset-AUC ≠ Generalisierung; dafür ist die SWAN-DF-Cross-Dataset-Probe \cite{korshunov2023swandf} vorgesehen (fake-only → Cross-Dataset-*Recall*) |
| 3 | `auc_video_audio` (Video-Modell) degeneriert (5 Positive) | Nicht zitiert; `visual`/`both` sind gültig |
| 4 | Phase-1-Backbones frozen → keine forgery-spezifischen Features | Günstiger Baseline; Phase 2 (unfreeze/LoRA) lernt diese |
| 5 | Backbones domänenfremd vortrainiert (Kinetics / Sprache) | Transfer Learning unter Domain-Shift, durch Datenknappheit gerechtfertigt |
| 6 | Intra-Chunk-Labelrauschen (16-Frame-Chunk > ~9-Frame-Fake) | Per Min-Overlap-Relabel gemildert (Fake-Rate ~7 % → ~5 %); Rest erwartet |
| 7 | Bivariate Heatmap nicht fundamental neu | Rekombination; bescheiden claimen |
| 8 | Heatmap-Faithfulness (noch) nicht per Perturbation belegt | Als geplante Validierung (Phase 4) benannt |
| 9 | Concat-Param-Count im Log um ~2,1 M überschätzt | Echten Wert ~1,32 M berichten; Ablation ist Mechanismus-an/aus, nicht parameter-matched |
| 10 | Cross-Attention-vs-Concat einzelner Seed, kleines Eval | Directionaler Vorteil ohne Fehlerbalken; ≥3 Seeds + mehr Identitäten nötig |
| 11 | Clean-data-Sättigung (auc_video 0,997–0,999) | Clean-AUC nicht mehr diskriminativ; mögliche Artefakt-Abhängigkeit erst unter Phase 3/4 prüfbar |
| 12 | **Phasen 3 & 4 code-fertig, Ergebnisse ausstehend** | Die Kern-xAI-unter-Angriff-Aussage (RQ4b/Gap G4) ist damit noch nicht belegt |

## 4. Bedrohungen der Validität

Die dominierende Einschränkung ist die **Datenknappheit** (≈8 Train-Identitäten, 6 Test-Identitäten): Sie macht die Modellauswahl auf `val/auc_video` hochvariant, verhindert belastbare Ablations-Fehlerbalken und begrenzt jede Generalisierungsaussage. Zwei Silent-Failure-Klassen sind adressiert (Identity Leakage per deterministischem Split, Chunk-Labelrauschen per Min-Overlap-Relabel; `docs/audit_2026-06.md`), die **Generator-Leakage** bleibt jedoch offen und wird erst durch die ausstehende SWAN-DF-Cross-Dataset-Evaluation beantwortet. Bis die Phase-3/4-Sweeps laufen, ist der Beitrag als Apparat und Fragestellung formuliert, nicht als belegter Befund.

---

> [!note] Quellen
> Limitationstabelle aus Handout §9 (erweitert um die Ablations-/Phase-Punkte 10–12); Befunde aus [[results-de]] und den zugrunde liegenden Ergebnis-Notizen; Silent-Failure-Kontext aus `docs/audit_2026-06.md`. Erlaubte Formulierung: [[Claim Map]]; Positionierung: [[related-work-de]] §7.
