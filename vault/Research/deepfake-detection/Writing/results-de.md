---
title: Ergebnisse (Belegarbeit) — Deutsch
type: writing/results
status: draft-partial
language: de
created: 2026-07-05
updated: 2026-07-05
tags: [Writing, Results, Deutsch, Belegarbeit]
---

# Ergebnisse

> [!warning] Status — Phasen 1–2 vollständig, Phasen 3–4 ausstehend
> Dieser Abschnitt berichtet die abgeschlossenen Ergebnisse der Phasen 1–2 sowie die Diagnostikläufe (alle auf den leakage-bereinigten post-2026-06-11-Daten, 12k Videos, identitätsdisjunkt). Die Phase-3- (Robustheit) und Phase-4-Ergebnisse (Adversarial) **stehen aus** (`docs/project.md` §7.14–§7.15) und sind als `[ERGEBNISSE AUSSTEHEND]` markiert. Alle Zahlen sind wörtlich aus den Ergebnis-Notizen unter `Results/` übernommen. Metrikdefinitionen: [[experimental-setup-de]] §3.

## 1. Evaluationsprotokoll (Kurzfassung)

Primärmetrik ist `test/auc_video` (ROC-AUC auf max-gepoolten Per-Video-Scores). Zu beachten: Die unimodalen Modelle optimieren `label_video` bzw. `label_audio`, das multimodale Modell den kombinierten `label` — ihre `auc_video`-Werte messen **verschiedene Label-Definitionen** und sind daher **nicht direkt vergleichbar**. Innerhalb der Per-Kategorie-Diagnose sind für ein unimodales Modell nur die eigenmodale und die `both`-Zelle valide; kreuzmodale Zellen sind degeneriert (zu wenige positive Videos) und werden weggelassen.

## 2. Phase 1 — Unimodale Baselines (eingefrorene Backbones)

**Video (VideoMAE, `label_video`).** Der eingefrorene Phase-1-Probe (nur `fc_norm`+`classifier`, 3\,074 trainierbare Parameter) erreicht `test/auc_video` **0,730** und fittet die Trainingsdaten nicht (train loss $\approx \ln 2$): Die eingefrorenen Kinetics-Features sind für die Video-Forgery-Erkennung nicht linear separierbar.

**Audio (Wav2Vec 2.0, `label_audio`).** Der eingefrorene Backbone + trainierter Kopf (197\,378 / 94\,569\,090 Parameter, ~0,2 %) liefert dagegen einen **starken** Audio-Baseline:

| Metrik (test, video-level) | Wert |
|---|---|
| `auc_video` | **0,976** |
| `ap_video` | 0,976 |
| `acc_video` | 0,787 |
| `f1_video` | 0,815 |

Valide Per-Kategorie-AUC (eigenmodal + `both`): audio-manipuliert **0,982** (272 pos), both **0,984** (277 pos). Die Zelle visual-only ist unter `label_audio` degeneriert (4 positive Videos) und wird **nicht** berichtet — der korrekte visual-only-Baseline ist das Video-Modell. Der Lauf ist nicht überangepasst (`val/loss 0,174 < train/loss 0,229`; frozener Backbone → nahezu keine Memorierungskapazität).

## 3. Phase 2 — End-to-End-Feintuning und multimodale Fusion

**Unfreezing der unimodalen Backbones.** Das Entfrieren hebt beide Baselines auf clean-data-Niveau nahe der Decke:

| Modell (test, video-level) | Phase 1 (frozen) | Phase 2 (unfrozen) |
|---|---|---|
| VideoMAE `auc_video` | 0,730 | **0,999** |
| Wav2Vec2 `auc_video` | 0,976 | **0,997** |

Für Video ist das Unfreezing **entscheidend** (0,730 → 0,999); für Audio nur ein moderater Sprung (+0,021), weil die eingefrorenen Features bereits stark waren — der Hauptgewinn liegt dort im Betriebspunkt (`f1_video` 0,815 → 0,983). **Auf sauberen Daten ist die In-Distribution-Aufgabe damit für beide Modalitäten gesättigt** — die clean-AUC ist nicht mehr diskriminativ, das eigentliche Signal verlagert sich auf Phase 3/4.

**Multimodale Fusion (kombinierter `label`, Phase-1-frozen).** Die Cross-Attention-Fusion erreicht `test/auc_video` **0,960** und ist über alle drei Manipulationstypen balanciert (audio 0,957 / both 0,988 / visual-only 0,932, alle valide, je 272–277 pos). Die Mechanismus-Ablation gegen `concat`:

| Metrik (test, video-level) | Concat | Cross-Attention |
|---|---|---|
| `auc_video` | 0,934 | **0,960** |
| `ap_video` | 0,966 | **0,979** |
| `acc_video` | 0,877 | **0,908** |
| `f1_video` | 0,913 | **0,934** |

Cross-Attention gewinnt auf allen acht Testmetriken (visual-only 0,932 vs. 0,868).

> [!warning] Einordnung — kein robuster „Cross-Attention ist notwendig"-Beleg
> Drei Konfundierungen begrenzen diese Aussage: (1) Die Ablation ist **Mechanismus-an/aus, nicht parameter-matched** — die `concat`-Config baut die ~2,1 M Attention-Parameter, nutzt sie aber nicht (effektive Kapazität ~1,3 M), sodass „Cross-Attention hat mehr nutzbare Kapazität" ein offener Confound bleibt (der geloggte Trainable-Count 3,42 M ist irreführend). (2) **Einzelner Seed**, kleines Eval (4 val / 6 test Identitäten) — die Differenz braucht Fehlerbalken. (3) Der Fusions-Lauf **überfittet** (train loss 0,214 < val 0,343). Der frühere leakage-behaftete 4000-Video-Lauf zeigte zudem einen Gleichstand (0,651 vs. 0,654). Die Aussage lautet daher: Cross-Attention ist auf den aktuellen Daten **directional überlegen**, aber **nicht robust als notwendig belegt**; ein Phase-2-Fusionslauf (entfrorene Backbones, Vergleich gegen die unimodalen 0,999) sowie ≥3 Seeds stehen aus.

## 4. Diagnostik

**Frame-Perturbation (temporale vs. räumliche Dominanz).** Auf dem eingefrorenen Video-Probe (clean `auc_video` 0,745 — der clean-Referenzlauf der Perturbations-Evaluation; der Phase-1-Baseline-Lauf in §2 misst 0,730, dasselbe eingefrorene Modell in zwei Läufen) senkt das Mischen der Frame-Reihenfolge *innerhalb* jedes 16-Frame-Chunks die video-level AUROC auf 0,597 (`tubelet_shuffle`) bzw. 0,691 (`frame_shuffle`, voll). Beide Perturbationen degradieren die AUROC ⇒ der Probe nutzt **intra-chunk-temporale Ordnung**, ist also nicht rein räumlich. Details und Vorbehalte (Nicht-Monotonie, frozener Probe): [[videomae-frame-perturbation-temporal]].

**Datensatz-Ablation (Diversität vs. Paarung) — in Arbeit.** Nur der `keep_pairs`-Arm ist trainiert (`val/auc_video` 0,769); der `decouple`-Kontrollarm ist vorprozessiert, aber nicht trainiert, und es liegt keine Cross-Dataset-Auswertung vor. **Kein Paarungs-/Diversitätseffekt ist bislang belegt** ([[dataset-ablation-pairing-diversity]]).

## 5. Phase 3 — Robustheit (Social-Media-Simulation) `[ERGEBNISSE AUSSTEHEND]`

Die Sweep-Infrastruktur ist vollständig (uni- und multimodal); die folgende Tabelle wird nach den Läufen auf den post-2026-06-11-Daten mit den W&B-Zahlen gefüllt. Forschungsfragen (RQ3a–c): quantitativer Breaking Point (CRF/FPS), Attention-Shift unter Degradierung, Kompressionsanfälligkeit des Audio- vs. Video-Zweigs.

| Branch | CRF | FPS | Audio-kbps | AUC | Accuracy | Fooling Rate | Δfake |
|---|---|---|---|---|---|---|---|
| Baseline (clean) | — | — | — | — | — | — | — |
| video | … | … | — | — | — | — | — |
| audio | — | — | … | — | — | — | — |
| multimodal (joint) | … | … | … | — | — | — | — |

## 6. Phase 4 — Adversariale Angriffe (White-Box) `[ERGEBNISSE AUSSTEHEND]`

FGSM/PGD (uni- & multimodal), UAP und adversariales Fine-Tuning sind implementiert. Kernfrage (RQ4b, Beitrag G4): Verschiebt ein erfolgreicher Angriff die AttnLRP-Relevanz von semantischen Regionen (Mund/Augen) auf den Hintergrund, und senkt adversariales Training die Fooling Rate ohne Clean-Accuracy-Verlust?

| Setup | Methode | Modalität | ε | AUC | Accuracy | Fooling Rate | Δfake | Attn-Shift |
|---|---|---|---|---|---|---|---|---|
| Baseline-Ckpt | FGSM | video | … | — | — | — | — | — |
| Baseline-Ckpt | PGD | both | … | — | — | — | — | — |
| Baseline-Ckpt | PGD | audio | … | — | — | — | — | — |
| Adv-trainiert | PGD | both | … | — | — | — | — | — |

---

> [!note] Werte und Quellen
> Alle Zahlen aus den Ergebnis-Notizen: [[videomae-unimodal-video-baseline]], [[wav2vec2-phase1-audio-baseline]], [[wav2vec2-phase2-audio-end-to-end]], [[multimodal-fusion-phase1-baseline]], [[multimodal-concat-phase1-ablation]], [[videomae-frame-perturbation-temporal]], [[dataset-ablation-pairing-diversity]]. Degenerierte Per-Kategorie-Zellen (Audio-Modell visual-only, 4 pos; Video-Modell audio-only, 5 pos) sind bewusst weggelassen. Phase-3/4-Tabellen aus `docs/model.md` §7.14–§7.15 (Platzhalter).
