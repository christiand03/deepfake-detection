---
title: Ergebnisse (Belegarbeit) — Deutsch
type: writing/results
status: draft-partial
language: de
created: 2026-07-05
updated: 2026-07-15
tags: [Writing, Results, Deutsch, Belegarbeit]
---

# Ergebnisse

> [!warning] Status — Phasen 1–3 vollständig, Phase 4 ausstehend
> Dieser Abschnitt berichtet die abgeschlossenen Ergebnisse der Phasen 1–3 sowie die Diagnostikläufe (alle leakage-bereinigt, identitätsdisjunkt). Die **Phase-1/2-Zahlen** stammen von der 32-Identitäten-Trainingsstufe (Test 1\,169 Videos); die **Phase-3-Auswertung** nutzt dieselben Checkpoints auf dem ausgebauten 165-Identitäten-Testsplit (1\,471 Videos, leakage-frei — siehe [[experimental-setup-de]] §1). Die Phase-4-Ergebnisse (Adversarial) **stehen aus** und sind als `[ERGEBNISSE AUSSTEHEND]` markiert. Alle Zahlen sind wörtlich aus den Ergebnis-Notizen unter `Results/` übernommen. Metrikdefinitionen: [[experimental-setup-de]] §3.

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

## 5. Phase 3 — Robustheit (Social-Media-Simulation)

Phase 3 misst, wie die Phase-1/2-Detektoren unter Social-Media-typischer Re-Enkodierung zerfallen. Je Degradierungsstufe wird jedes der 1\,471 Test-Videos per FFmpeg neu kodiert und auf Video-Ebene gegen den Clean-Durchlauf verglichen. Video- und Multimodal-Zweig laufen über ein **CRF×FPS-Gitter** (CRF $\in\{18,23,28,35,40,45,51\}$, FPS $\in\{25,15,10,5\}$), der Audio-Zweig über die **Audio-Bitrate** ($\{128,64,32,16\}$ kbps) und der Video-Zweig zusätzlich über eine **Upscale**-Stufe ($640{\times}360 \to 1280{\times}720$); das Multimodal-Gitter fixiert die Audio-Bitrate auf 64 kbps. Vollständige Analyse: [[phase3-robustness-social-media-sweep]].

> [!warning] Vergleichbarkeit der Clean-Baselines
> Die hier berichteten Clean-AUC-Werte sind **nicht** mit den Phase-1/2-Zahlen (§2–§3) identisch: Sie stammen vom größeren 165-Identitäten-Testsplit, und der Video- sowie der Multimodal-Zweig werden gegen den **kombinierten** `label` (fake, wenn Bild *oder* Ton manipuliert) bewertet, nicht gegen `label_video`. Der Video-Zweig verliert dadurch die reinen Audio-Fakes (echtes Bild) und startet bei `auc_video` 0,857 statt der gesättigten 0,999 aus Phase 2. Die Werte sind *innerhalb* von Phase 3 konsistent (gleicher Referenzlauf), phasenübergreifend aber nur eingeschränkt vergleichbar.

**Clean-Baseline, Breaking Point (schlechteste Gitterzelle) und Upscale je Zweig** (test, video-level; ΔAUC = Abfall gegenüber clean):

| Zweig | Bedingung | AUC | ΔAUC | Accuracy | Fooling Rate |
|---|---|---|---|---|---|
| Video | clean | 0,857 | — | 0,767 | — |
| Video | CRF 51 / 10 FPS | **0,527** | −0,330 | 0,345 | 0,562 |
| Multimodal | clean | 0,929 | — | 0,779 | — |
| Multimodal | CRF 51 / 25 FPS | **0,741** | −0,188 | 0,716 | 0,244 |
| Audio | clean (128 kbps) | 0,975 | — | 0,957 | 0,004 |
| Audio | 16 kbps | **0,921** | −0,054 | 0,844 | 0,142 |
| Video (Upscale) | 360p → 720p | 0,777 | −0,080 | 0,701 | 0,096 |

**Der Video-Zweig ist fragil, die Fusion nicht.** Kompression und Framerate-Verlust brechen den reinen Video-Detektor unabhängig voneinander bis nahe an den Zufall (`auc_video` 0,857 → 0,527; Kompression allein bei 25 FPS 0,844 → 0,549, Framerate allein bei CRF 23 0,844 → 0,596). Das multimodale Modell bleibt über das gesamte Gitter nutzbar (schlechteste Zelle 0,741) und **schlägt den Video-Zweig in allen 28 Gitterzellen** (mittlerer AUC-Abstand +0,195, in der härtesten Ecke CRF 51/5 FPS +0,252).

![[Results/assets/phase3-robustness/figure-01-auc-heatmap.png]]

![[Results/assets/phase3-robustness/figure-02-degradation-curves.png]]

**Die Robustheit trägt der Audio-Zweig.** Wav2Vec2 ist gegenüber Audio-Kompression nahezu invariant (`auc_video` 0,975 → 0,977 bis 64 kbps) und hält selbst bei 16 kbps noch 0,921. Da CRF/FPS das Audio nicht berühren, bleibt im Fusionsmodell unter Bild-Degradierung eine intakte Evidenzquelle erhalten — die Fusions-Robustheit ist also primär **Audio-Präsenz**, keine Eigenschaft des Fusions-*Mechanismus* (das Multimodal-Gitter hält Audio bei 64 kbps fest, isoliert also keine reine Bild-Degradierung).

**Mechanismus: gegenläufige Verzerrung.** Die beiden Degradierungsachsen verschieben den Video-Detektor in *entgegengesetzte* Richtungen: Starke Kompression drückt die Vorhersage **Richtung REAL** (Δfake bis +0,41), Framerate-Verlust **Richtung FAKE** (Δfake bis −0,49). Weil der Testsplit zu 71 % aus Fakes besteht, ist die Kompressions-Verzerrung die gefährliche: Bei CRF 51/25 FPS fällt die Accuracy auf 0,358 — das Modell hält unter starker Kompression fast alles für echt und übersieht die Fakes. In der härtesten Ecke (CRF 51/5 FPS) heben sich beide Effekte teils auf (Δfake +0,33). Dies motiviert die AttnLRP-Folgeanalyse auf degradierten Eingaben ([[AttnLRP Bivariate Heatmap]]).

![[Results/assets/phase3-robustness/figure-03-directional-bias.png]]

> [!note] Statistik und Grenzen
> Jede Gitterzelle ist ein einzelner Durchlauf über 1\,471 Videos (**keine Seeds**, keine gespeicherten Per-Video-Scores); seed-übergreifende Signifikanztests sind daher nicht möglich. Die 95 %-Konfidenzbänder in der Kurven-Abbildung sind Hanley-McNeil-Intervalle für eine einzelne AUC (Halbbreite ≈ ±0,02 für Video/Multimodal, ±0,01 für Audio) und erfassen nur die Stichprobenstreuung des Testsatzes, nicht die Lauf-zu-Lauf-Varianz. Da die Clean→degradiert-Abfälle (0,19–0,33 AUC) die KI-Halbbreiten um eine Größenordnung übersteigen, ist die Ordnung *Audio ≫ Multimodal > Video* robust; einzelne Gitterzellen-Differenzen tragen jedoch keine p-Werte.

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
> Alle Zahlen aus den Ergebnis-Notizen: [[videomae-unimodal-video-baseline]], [[wav2vec2-phase1-audio-baseline]], [[wav2vec2-phase2-audio-end-to-end]], [[multimodal-fusion-phase1-baseline]], [[multimodal-concat-phase1-ablation]], [[videomae-frame-perturbation-temporal]], [[dataset-ablation-pairing-diversity]], [[phase3-robustness-social-media-sweep]] (Phase 3). Degenerierte Per-Kategorie-Zellen (Audio-Modell visual-only, 4 pos; Video-Modell audio-only, 5 pos) sind bewusst weggelassen. Die Phase-4-Tabelle ist weiterhin Platzhalter.
