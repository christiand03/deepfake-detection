# 99 — Abgleichmatrix Code → Beleg

**Das Arbeitswerkzeug.** Jede Zeile ist ein implementierter Mechanismus. Die Spalte *Status*
ist gegen `docs/kapitel/*.tex` (Stand 2026-08-01) **gefüllt**:

| Kürzel | Bedeutung |
|---|---|
| `✓` | im Beleg beschrieben, Beschreibung stimmt mit dem Code überein |
| `~` | erwähnt, aber unvollständig oder ungenau |
| `○` | **noch nicht im Fließtext, aber in der Kapitelskizze als To-do vermerkt** — der Punkt ist erkannt und eingeplant, nur noch nicht ausgeschrieben |
| `✗` | **Lücke** — implementiert, aber weder im Fließtext noch in einer Skizze |
| `–` | bewusst weggelassen (Begründung in die Notizspalte) |
| `!` | **Widerspruch** — Beleg beschreibt etwas anderes als der Code tut |

> **Zum Unterschied `○` ↔ `✗`.** Die Kapitel 05–09 bestehen zum Stand 2026-08-01 fast
> vollständig aus Skizzen (`% SKIZZE`-Blöcke bzw. Stichpunktlisten), die Kapitel 00–04 haben
> geschriebenen Text **plus** einen Skizzenblock mit den fehlenden Inhalten. `○` heißt: die
> Skizze nennt den Punkt, das Risiko ihn zu vergessen ist gering. `✗` heißt: der Punkt taucht
> **nirgends** auf — weder im Text noch in einer Skizze — und geht ohne dieses Register
> verloren. **Für die Lückensuche sind die `✗`-Zeilen der eigentliche Ertrag.**

Die Spalte *Kap.* nennt das Kapitel, in dem der Punkt erwartet wird. Die Spalte *Reg.*
verweist auf das Registerdokument mit der Detailbeschreibung.

> **Stand: nachgezogen gegen [01](01_datenpipeline.md)–[12](12_dokumentation_vault.md).**
> Die Zeilen-IDs sind **stabil** — neue Punkte hängen hinten an ihrer Sektion an, bestehende
> wurden nicht umnummeriert (andere Registerdokumente verweisen auf sie, z. B.
> [12 §1.2](12_dokumentation_vault.md) auf F25). Neun Zeilen wurden dabei **inhaltlich
> korrigiert**, weil sie den Fachdokumenten widersprachen: A20, B10, D2, E5, G1, G4, G8,
> H2, H9. Die betroffenen Aussagen sind unten jeweils in der Zeile selbst begründet.

---

## Ergebnis des Abgleichs

Abgeglichen wurden alle 270 Zeilen gegen die zehn Kapiteldateien in `docs/kapitel/`
(Stand 2026-08-01: geschriebener Text in 00–04, Skizzen in 05–09). **Der Bestand liegt
inzwischen bei 290 Zeilen** — S36 bis S38 sind am 2026-08-05 nachträglich angelegt worden
(siehe die Fortschreibung zum Registerabschnitt 07), D36 bis D40 sowie Q22 und Q23 am
selben Tag (Fortschreibung zum Registerabschnitt 10), A41 bis A46 und B17 bis B20 am
2026-08-06 (Fortschreibung *Vollständigkeitskontrolle*, Registerdokument
[01](01_datenpipeline.md)) sowie F49 bis F56 am selben Tag (Registerdokument
[04](04_xai.md)) und S39 bis S47 am selben Tag (Registerdokument
[08](08_frontend.md)), F57 und A47 aus der Archivprüfung sowie S48 bis S55 aus dem
Registerdokument [06](06_backend_api.md) sowie Q24 bis Q30 aus
[12](12_dokumentation_vault.md) — sämtlich am 2026-08-06.
sowie Q31 als Nachtrag desselben Tages und G18 sowie H29–H35 aus
[05](05_robustheit_adversarial.md) sowie B21, D41–D44 und Q32–Q34 aus
[03](03_training_evaluation.md), C16–C19 und D45–D49 aus [02](02_modelle.md) sowie
Q35–Q37 aus [00](00_inventar.md).
**Gesamtbestand: 353 Zeilen**, dazu die nicht mitgezählte Checkliste V1–V8.
**Damit ist die Vollständigkeitskontrolle über alle zwölf Registerdokumente
abgeschlossen.**

| Status | Zeilen | Anteil | Bedeutung für die Arbeit |
|---|---:|---:|---|
| `✓` | 42 | 12 % | steht korrekt im Beleg |
| `~` | 0 | 0 % | steht drin, aber unvollständig oder ungenau |
| `○` | **291** | **82 %** | in der Skizze erkannt, noch nicht ausgeschrieben |
| `✗` | **0** | **0 %** | **taucht nirgends auf — auch nicht in einer Skizze** |
| `!` | 7 | 2 % | **Widerspruch zum Code** |
| `–` | 13 | 4 % | geprüft und bewusst weggelassen (C19, D44, F38, F56, H35, S47, S55, Q27, Q29, Q30, A43, A45, A46) |

> **Zusätzlich, außerhalb dieser Bilanz: die Checkliste V1–V8** in §*Zahlen aus dem Vault*.
> Acht Auflagen für das Übertragen von Ergebniszahlen nach `06Results.tex`, mit eigener
> Statusspalte. Sie sind keine implementierten Mechanismen und zählen deshalb nicht mit
> (Registerzeile Q31).

> **Stand der Widersprüche 2026-08-06 — drei von zehn sind erledigt.** **F14**
> (Wortaggregation „aufsummiert" statt gemittelt), **F18** (falsche Regionsliste mit
> Schultern und Hintergrund) und **F57** (Attention Rollout als Baseline) sind im
> Fließtext korrigiert und stehen auf `✓`. F18 und F57 zogen dabei jeweils vier
> Folgestellen nach sich — die Korrektur betraf zusammen **neun** Textstellen in sechs
> Kapiteldateien. Offen bleiben sieben: A11, A14, A15, A22, D2, G1b, H2.

> **Nur 37 Zeilen stehen im Fließtext.** Die 226 `○`-Zeilen sind Stichpunkte in den
> Skizzenblöcken — gesichert, aber nicht geschrieben. Wer den Fortschritt beurteilt, darf
> `○` nicht als „im Beleg" lesen. Geschrieben sind 12 %, gesichert sind 86 %.
>
> **Die `✗`-Zahl steigt seit dem 2026-08-06 wieder**, von 7 über 14 und 21 auf 29. Das ist
> kein Rückschritt: sie steigt, weil die Vollständigkeitskontrolle Mechanismen findet, die
> im Register beschrieben waren, aber **keine Zeile** hatten und deshalb in keiner Zählung
> auftauchten. Jede neue `✗`-Zeile ist eine Lücke, die vorher unsichtbar war.

> **Fortschreibung 2026-08-04 (Modus A, Abschnitt Datenpipeline).** Die 27 offenen Zeilen der
> Abschnitte A und B (20 × `✗`, 7 × `~`) sind in die Skizzenblöcke von 04, 05, 06, 07 und 09
> überführt und stehen jetzt auf `○`. Der Ausgangsstand vom 2026-08-01 war 20 `~`, 74 `○`,
> 130 `✗`. Die vier `!`-Zeilen des Abschnitts (A11, A14, A15, A22) sind bewusst unangetastet —
> sie werden im Fließtext korrigiert, nicht in eine Skizze verschoben.

> **Fortschreibung 2026-08-04 (Modus A, Abschnitt Modelle).** Die neun offenen Zeilen des
> Abschnitts C (6 × `✗`, 3 × `~`) sind in die Skizzenblöcke von 04, 05, 06, 08 und 09
> überführt und stehen jetzt auf `○`: C1, C5, C7, C8, C11, C12, C13, C14, C15. Abschnitt C
> hat keine `!`-Zeile. Zwei Nebenbefunde: der Dropout-Widerspruch aus C5 ist gegen
> `configs/model/multimodal.yaml` aufgelöst (es gilt **0,3**, nicht der Modul-Default 0,1),
> und für den Kapitel-7-Anteil von C8 ist `–` vorgeschlagen (Doppelung mit Limitation 5).
> Die Zählung oben ist entsprechend nachgezogen (vorher 13 `~`, 101 `○`, 110 `✗`).
>
> **Abschnitt C ≠ Registerdokument [02](02_modelle.md).** `base_module.py` und `metrics.py`
> liefern Zeilen, die in anderen Matrixabschnitten stehen. Aus [02](02_modelle.md) beschrieben,
> aber am 2026-08-04 **nicht** überführt und weiterhin offen (Stand desselben Tages; maßgeblich
> bleibt die Statusspalte):
> **D11, D13, D21, D22, D28** (Training — LoRA-Guards, Warm-Start-Schlüsselübersetzung,
> prozesslokale Aggregationspuffer, `unfreeze_backbone` als Paritätshelfer statt Laufzeitschalter,
> PEFT-Hook bei Wav2Vec2), **E5, E8, E12** (Evaluation — Adaption statt Eigenentwicklung von
> `RecallAtFixedFPR`, Sanity-Check-Ausschluss, Chunk-AUC als stiller Ersatz), **H6** (gemeinsamer
> multimodaler Rückwärtspass des PGD). Wer diese Abschnitte bearbeitet, findet die
> Detailbeschreibung in [02 §base\_module.py](02_modelle.md), [§metrics.py](02_modelle.md) und
> [§multimodal\_module.py](02_modelle.md) — das Dokument muss dafür nicht erneut vollständig
> gelesen werden. **Erledigt am 2026-08-04** durch die Überführung der Abschnitte D und E
> (nächster Absatz); offen aus dieser Aufzählung bleibt nur noch **H6**.

> **Fortschreibung 2026-08-04 (Modus A, Abschnitt Training und Evaluation).** Die 20 offenen
> Zeilen der Abschnitte D und E (19 × `✗`, 1 × `~`) sind in die Skizzenblöcke von 05 und 09
> überführt und stehen jetzt auf `○`: D5, D8, D11, D12, D13, D20, D21, D22, D23, D24, D26,
> D27, D28, D32, D33, D34, D35, E5, E8, E12. Abschnitt E hat keine `!`-Zeile; die `!`-Zeile
> D2 (Phase-2-Batchaufteilung Video) ist bewusst unangetastet und gehört in den Fließtext.
> Der gesamte Ertrag landet in Kapitel 5 — Kapitel 4 bekommt aus diesem Registerabschnitt
> **keinen** Punkt, weil Training und Evaluation hier durchweg Laufbedingungen beschreiben
> und keine Designbegründungen. Drei Zeilen wurden gegen den Code verifiziert:
> `ckpt_export_name` steht in genau 27 Dateien unter `configs/experiment/` (D20),
> `min_lr_ratio` kommt in keiner YAML als Schlüssel vor (D26), und `devices: 1` steht in
> `configs/trainer/default.yaml` (D21). Für drei Teilzuordnungen ist `–` vorgeschlagen:
> der Kapitel-7-Anteil von D21, der Kapitel-6-Anteil von E12 und — vorbehaltlich einer
> Prüfung der Läufe — D33 insgesamt. Die Zählung oben ist entsprechend nachgezogen
> (vorher 10 `~`, 110 `○`, 104 `✗`).

> **Fortschreibung 2026-08-04 (Modus A, Abschnitt xAI).** Die 22 offenen Zeilen des
> Abschnitts F (19 × `✗`, 3 × `~`) sind geprüft: **21 sind in die Skizzenblöcke von 04,
> 05, 06, 07 und 09 überführt** und stehen jetzt auf `○` — F5, F13, F19, F24, F26, F27,
> F28, F29, F31, F34, F35, F36, F37, F40, F41, F42, F43, F44, F45, F47, F48 —, **eine
> steht auf `–`** (F38, s. u.). Die beiden `!`-Zeilen
> F14 (Audio-L2 „aufsummiert" statt vorzeichenbehaftetes Mittel) und F18 (Regionen mit
> Schultern und Hintergrund statt sieben Landmark-Regionen) sind bewusst unangetastet —
> sie werden im Fließtext korrigiert. Der Ertrag verteilt sich auf fünf Kapitel: Kapitel 4
> bekommt die Methodenaussagen, Kapitel 5 die Laufbedingungen der Abbildungserzeugung,
> Kapitel 6 die Herkunfts- und Leseregeln der Bildtafeln, Kapitel 7 zwei Eigenschaften
> der Audio-xAI und den Faithfulness-Vorbehalt, Kapitel 9 die Testzeile.
> Fünf Zeilen wurden gegen den Code verifiziert: `per_class=True` kommt in `src/` nur in
> `api/inference.py` vor (F26), `[0:1]` steht in allen drei Erklärskripten (F31),
> `explain.py` zeichnet fest ±1 gegen `hm_vmax` in `explain_multimodal.py` (F28), der
> Wächter `_lxt_patched` steht an zwei Stellen in `attnlrp.py` (F29) — und **F13 ist
> falsch benannt:** Docstring und Matrixzeile sagen „Abs-Max-Pooling", der Code rechnet
> ein Mittel der Beträge mal dem Mehrheitsvorzeichen (`audio_xai.py:204-207`). Für vier
> Teilzuordnungen ist `–` vorgeschlagen: der Kapitel-6-Anteil von F19, der
> Kapitel-4-Anteil von F47, der Kapitel-9-Anteil von F24 und der Kapitel-7-Anteil von
> F36. Die Zählung oben ist entsprechend nachgezogen (vorher 9 `~`, 130 `○`, 85 `✗`);
> `–` erscheint mit F38 erstmals als eigene Statuszeile.
>
> **Fehlerhistorie gehört nicht in den Beleg** (Autorenentscheidung 2026-08-04). Der
> Beleg beschreibt den Endstand. Zwei Zeilen sind davon betroffen:
>
> - **F36** — die Bandaufteilung von etwa 0,43 / 0,56 ist der Messwert eines behobenen
>   Fehlers, kein Ergebnis. Übernommen wird ausschließlich die heutige Formel: der
>   Bandwert ist ein energiegewichtetes Mittel `Σ(E·R)/Σ(E)`, nötig, weil Sprachenergie
>   fast vollständig in Low und Mid liegt. Der Diskussionsanteil in Kapitel 7 entfällt,
>   F36 steht nur noch in Kapitel 4.
> - **F38** — die frühere Wortrelevanz über `argmax(|·|)` trifft auf den heutigen Code
>   nicht mehr zu und ist vollständig auf `–` gesetzt.
>
> Nicht betroffen sind F35 und F37: beide beschreiben Eigenschaften der heutigen
> Fassung, keine überholten Zustände. Ebenfalls unberührt bleibt der `!`-Widerspruch
> **F14** — dass Kapitel 4 die Wortaggregation als „aufsummiert" beschreibt, während der
> Code vorzeichenbehaftet mittelt, ist ein Fehler *im Beleg* und weiterhin in Modus C zu
> korrigieren. Er hängt nicht an F38.

> **Fortschreibung 2026-08-05 (Modus A, Abschnitt Robustheit und Adversarial).** Die 30
> offenen Zeilen der Abschnitte G und H (24 × `✗`, 6 × `~`) sind in die Skizzenblöcke von
> 04, 05, 06, 07 und 09 überführt und stehen jetzt auf `○`: G1, G4, G6, G7, G8, G12, G13,
> G17, H3, H4, H5, H6, H7, H9, H10, H11, H14, H15, H17, H18, H19, H20, H21, H22, H23, H24,
> H25, H26, H27, H28. Die beiden `!`-Zeilen G1b (Gauß-Rauschen als Sweep-Achse) und H2
> (Angriffsziel wahres Label statt eigene Vorhersage) sind bewusst unangetastet — sie
> werden im Fließtext korrigiert. Der Ertrag verteilt sich nach Aussagetyp: Kapitel 4
> bekommt die Angriffs- und Degradationsmechanik (17 IDs), Kapitel 5 die Laufbedingungen
> und Metrikbedingungen (7), Kapitel 6 die Lesehinweise zu Tabellen und Abbildungen (8),
> Kapitel 7 eine Deutungszeile (G6), Kapitel 9 drei Anhangzeilen. Drei Zeilen wurden gegen
> den Code verifiziert: die ~60-h-Schätzung und die je Konfiguration eigene Resume-CSV
> stehen in `scripts/smoke_phase34.ps1:9` und `scripts/run_phase34.ps1:58-60,196-211`
> (H24/H26), und **G17 ist zu schwach formuliert** — auch der Offline-Sweep verzichtet auf
> die Wortebene, weil `eval_robustness_sweep.py` `run_audio_inference_score`
> (`src/api/inference.py:2605`) aufruft. Für vier Teilzuordnungen ist `–` vorgeschlagen:
> der Kapitel-6-Anteil von G6 und von H3, außerdem die Kapitel-6-Verortung von H10/H20/H21
> (Metrikdefinitionen sind Setup-Aussagen, Begründung wie bei E5) und von H27 (die
> Messgröße wird in Kapitel 4 definiert). Mit dieser Runde ist `~` **leer**: die
> verbliebenen 42 `✗`-Zeilen liegen vollständig in **S Demonstrator** (30) und
> **I Reproduzierbarkeit** (12). Die Zählung oben ist entsprechend nachgezogen (vorher
> 6 `~`, 151 `○`, 66 `✗`).
>
> ~~**Ein Widerspruch innerhalb des Registers, der beim Ausschreiben zu beachten ist.** Die
> Kennzahlentabelle in [05](05_robustheit_adversarial.md) definiert den Breaking Point als
> „erster Gitterpunkt, an dem das Urteil kippt" und verweist dafür auf
> `RobustnessPanel.tsx:188`.~~ — **behoben 2026-08-06.** Die Kennzahlenzeile trennt jetzt
> den auswertungsseitigen Kipppunkt aus den Sweep-Tabellen von der gleichnamigen
> Frontend-Komponente; ein eigener Kasten in [05](05_robustheit_adversarial.md) hält fest,
> dass `BreakingPoint` (`RobustnessPanel.tsx:188`) keinen Sweep fährt, sondern den
> relativen Konfidenzverlust eines einzelnen Parametersatzes einstuft. Zeile G8 bleibt
> unverändert gültig.

> **Fortschreibung 2026-08-05 (Modus A, Registerabschnitt [06 Backend / FastAPI](06_backend_api.md)).**
> Bearbeitet wurden die 15 Zeilen des Matrixabschnitts **S**, deren Spalte *Reg.* auf
> [06](06_backend_api.md) verweist und die auf `✗` standen: S2, S3, S4, S5, S6, S7, S13, S14,
> S15, S16, S17, S18, S20, S21, S22. Alle 15 sind überführt und stehen jetzt auf `○`. Die
> beiden übrigen 06-Zeilen (S1, S19) standen bereits auf `○`; `!` und `~` kommen in diesem
> Registerabschnitt nicht vor.
>
> Der Ertrag verteilt sich nach Aussagetyp und weicht dabei mehrfach von der Spalte *Kap.* ab,
> die für fast alle Zeilen pauschal Kapitel 4 nennt:
> **Kapitel 4** bekommt die Designentscheidungen und Definitionen (S6+S7 Datenherkunft des
> Demonstrators, S16 Reichweite der Ganzclip-Analyse, S18 Trennung Urteil/Konfidenz, S15
> Bedingung der Reprojektion, S17 zusätzlicher sauberer Durchlauf der Phase 4);
> **Kapitel 5** die Laufbedingungen (S3+S22 Checkpointwahl über Umgebungsvariablen und die
> nicht erzwungene Fusionsart, S4+S13+S14+S20 Cacheverhalten, S21 schema-erzwungene
> Parametergrenzen); **Kapitel 6** zwei Lesehinweise zu den Abbildungen (S18, S17);
> **Kapitel 7** zwei Limitationszeilen (S13 Punkt 21, S15 Punkt 22); **Kapitel 9** zwei
> Anhangzeilen (S2 am Systemdiagramm §F, S5+S3 in §G).
>
> Zwei Zeilen wurden gegen den Code verifiziert: `src/api/app.py:53-73` lädt im Preload **nur**
> VideoMAE und Wav2Vec2 — das multimodale Modell entsteht erst bei der ersten Anfrage (S2);
> und `src/api/inference.py:184-194` bestätigt S22, der abweichende `fusion_mode` erzeugt
> ausschließlich eine Logwarnung, das Modell wird trotzdem zwischengespeichert und benutzt.
> Für eine Teilzuordnung ist `–` vorgeschlagen: S5 (manuell synchron gehaltener
> TypeScript-Vertrag) trifft keine Forschungsfrage und ist der erste Streichkandidat, falls der
> Anhang gekürzt wird.
>
> **Anschlusszeile aus einem anderen Registerabschnitt.** S23 (Reg. `07`) gehört inhaltlich in
> denselben Absatz wie S22 — nur `run_multimodal_inference` reicht den Fusionsmodus durch,
> weshalb alle multimodalen Sweep- und Phase-4-Werte für `cross_attention` gelten. Die Zeile
> stand in dieser Runde weiter auf `✗` und wurde hier **nicht** eingearbeitet
> (nachgeholt am selben Tag, s. u.).
>
> Die Zählung oben ist entsprechend nachgezogen (vorher 181 `○`, 42 `✗`).

> **Fortschreibung 2026-08-05 (Modus A, Registerabschnitt [07 Inferenzpipeline](07_inference_pipeline.md)).**
> Dieser Registerabschnitt hatte genau **eine** offene Zeile: **S23**. Sie ist überführt und steht
> jetzt auf `○`. Alle übrigen Zeilen mit *Reg.* `07` standen bereits auf `○`, `✓` oder `–`; `~`
> kommt in der Matrix nicht mehr vor. Die drei `!`-Zeilen dieses Registerabschnitts — **F14**
> (Audio-L2 „aufsummiert" statt vorzeichenbehaftetem Mittel), **F18** (Regionen mit Schultern und
> Hintergrund) und **G1b** (Gauß-Rauschen als Sweep-Achse) — sind bewusst unangetastet und im
> Fließtext zu korrigieren.
>
> S23 ist **auf drei Kapitel aufgeteilt**, weil die Zeile drei verschiedene Aussagetypen
> enthält: die Reichweite der Phasen 3 und 4 (Kapitel 4, §Methodik zur Evaluierung der
> Robustheit, Vorspann), die Laufbedingung (Kapitel 5, §Laufzeitkonfiguration, als Punkt (d)
> des bereits dafür reservierten Absatzes S3+S22) und die Gültigkeitsangabe zu den berichteten
> Zahlen (Kapitel 6, §Phase 3 und §Phase 4). Der Kapitel-5-Anteil weicht von der Spalte *Kap.*
> ab; die 05-Skizze hatte den Platz dafür ausdrücklich freigehalten.
>
> Gegen den Code verifiziert: `get_multimodal_model` hat den Vorgabewert `cross_attention`
> (`src/api/inference.py:151`); nur `run_multimodal_inference` gibt einen Modus weiter (`:2420`),
> während `run_multimodal_adversarial_inference` (`:3343`), `run_multimodal_inference_score`
> (`:3615`) und `run_multimodal_adversarial_batch` (`:3673`) ohne Argument aufrufen. Damit laufen
> der Robustheitssweep (`scripts/eval_robustness_sweep.py:569,628`), der Adversarialsweep
> (`scripts/eval_adversarial_sweep.py:389,453`) und die UAP-Anpassung
> (`scripts/compute_uap.py:388`) sämtlich auf `cross_attention`.
>
> **Eine Präzisierung gegenüber der Registerzeile.** S23 formuliert pauschal „unabhängig davon,
> was im Frontend umgeschaltet ist". Für Phase 4 und die Offline-Sweeps trifft das zu; die
> **interaktive Robustheitsroute** reicht den Modus dagegen sehr wohl durch
> (`src/api/routers/robustness.py:57` an `run_multimodal_robustness_inference`). Dass die
> Oberfläche ihn dort trotzdem fest auf `cross_attention` setzt, ist die eigenständige Zeile
> **S24** (Reg. `08`) und weiterhin offen — beim Ausschreiben nicht vermischen.
>
> Die Zählung oben ist entsprechend nachgezogen (vorher 196 `○`, 27 `✗`).
>
> **Drei Zeilen sind bei dieser Runde neu entstanden: S36, S37, S38.** Beim vollständigen
> Lesen von [07](07_inference_pipeline.md) fielen Mechanismen auf, die dort beschrieben
> sind, aber **keine Matrixzeile** hatten — sie standen damit weder auf `✗` noch auf `○`,
> tauchten in keiner Zählung auf und wären von keinem Durchgang je erfasst worden. Das ist
> eine Lücke im Register, nicht im Beleg:
>
> - **S36** — die Erzeugung der Crop-Videos für die Vorher-Nachher-Spieler. S10 beschreibt
>   nur die Frontend-Komponente; woher deren Bildmaterial stammt, stand nirgends.
> - **S37** — die MP4-Ausgaben unter `data/phase_media/`. Die zweite Plattenausgabe der
>   Laufzeitpipeline (WhisperX-Cache) war über F14 bereits abgedeckt, diese nicht.
> - **S38** — die Boxverfolgung je Fenster über lange Clips. Sie trägt jede zeitliche
>   Aussage des Demonstrators und schließt an S16 an.
>
> Alle drei stehen auf `✗` und sind **nicht** überführt; die Umfangvorschläge stehen in der
> jeweiligen Zeile. Der Gesamtbestand steigt damit von 270 auf 273 Zeilen, `✗` von 26 auf 29.
>
> **Der Befund ist allgemeiner als diese drei Zeilen.** Geprüft wurde nur das Registerdokument
> [07](07_inference_pipeline.md) gegen die Matrix. Ob dieselbe Art Lücke auch in
> [01](01_datenpipeline.md)–[06](06_backend_api.md) und
> [08](08_frontend.md)–[12](12_dokumentation_vault.md) steckt, ist offen und wäre ein
> eigener Durchgang. **Für [10](10_konfiguration.md) ist die Frage inzwischen beantwortet:
> ja, sieben Zeilen — siehe die Fortschreibung zum Registerabschnitt 10 und Befund 4.**

> **Fortschreibung 2026-08-05 (Modus A, Registerabschnitt [08 Frontend](08_frontend.md)).**
> Bearbeitet wurden die 13 Zeilen des Matrixabschnitts **S** mit *Reg.* `08`, die auf `✗`
> standen: S9, S11, S24, S25, S26, S27, S28, S29, S31, S32, S33, S34, S35. Alle 13 sind
> überführt und stehen jetzt auf `○`. Dieser Registerabschnitt hat **keine** `!`-Zeile; `~`
> kommt in der Matrix nicht mehr vor. Die übrigen 08-Zeilen (S8, S10, S30, G8, F17, F22,
> F23, F33, F34, F37, S5, S17, S18, S19, S21) standen bereits auf `○` oder `✓`.
>
> Der Ertrag ist fast vollständig **Darstellungssemantik**: Kapitel 4 bekommt das
> Erklärsystem als eigenständiges Artefakt (S9+S33 ein Absatz, S32 ein Absatz) und die
> Eigenschaften, die die Abbildungen lesbar machen (S29 sechs von sieben Regionen, S31
> Anzeigeraster, S34 nicht gezeichnete Schemafelder, S25 Achsensemantik und feste Skalen
> der Shift-Tabelle, S26+S27 Wirkbereich der beiden Verstärkungen, S28+S35 zwei
> Farbrampen); Kapitel 6 zwei Leseregeln (S32, S25) und eine Korrektur der
> Abbildungsplanung (S34); Kapitel 7 zwei Vorbehalte (S26 als neuer Punkt 23, S31 als
> Erweiterung von Punkt 13).
>
> **Zwei Zeilen weichen von der Spalte *Kap.* ab, beide nach Kapitel 5.** S24 (Fusionsmodus
> im Robustheitslabor) schließt als Punkt (e) den bereits reservierten Absatz S3+S22+S23
> und löst dessen offene Abgrenzung ein: auch die interaktive Route läuft praktisch auf
> `cross_attention`, weil die Oberfläche nichts anderes anbietet. S11 (Mock-Modus) hatte in
> der Spalte *Kap.* ein `—`, also gar kein Ziel; sie gehört zur Erzeugungsvorschrift der
> Screenshots, dieselbe Begründung wie bei F24/F31.
>
> **Eine Teilzuordnung steht auf `–`, eine ist dazugekommen.** Der Kapitel-7-Anteil von
> S34 entfällt (Autorenentscheidung 2026-08-05): als Limitation wäre er eine Doppelung des
> Kapitel-4-Satzes und betrifft keine berichtete Zahl. An seine Stelle tritt ein
> Kapitel-6-Anteil derselben Zeile, weil beim Lesen von [08](08_frontend.md) §6 eine
> **konkrete Fehlplanung** auffiel: die 06-Skizze sieht für Phase 4 ein Frame-Triptychon
> mit Differenz-Heatmap vor, `differenceFrames` wird aber von keiner Komponente gezeichnet
> — die Abbildung ist so nicht herstellbar und wäre offline zu erzeugen oder auf das Paar
> Clean/Adversarial zu verkürzen. Die Spalte *Kap.* von S34 ist entsprechend von
> „04, 07" auf „04, 06" korrigiert. Besonders heikel ist das in Verbindung mit S11: die
> Mock-Fabrik befüllt genau dieses Feld, ein aus dem Mock-Modus abfotografiertes
> Triptychon zeigte also erfundene Zahlen.
>
> Zwei Zeilen wurden gegen den Code verifiziert: `RobustnessPanel.tsx:362` setzt
> `fusionMode: 'cross_attention'` fest und `:490-498` sperrt den Audiotest bei aktivem
> Multimodalmodus (S24); `FaceSchematic.tsx:36` definiert sechs Regionsflächen, während
> `:112` die Bezugsgröße `totalMag` über **alle** gelieferten Regionen bildet — die
> Prozentanteile summieren sich damit nicht auf 100 % (S29).
>
> Die Zählung oben ist entsprechend nachgezogen (vorher 197 `○`, 29 `✗`). Damit liegen die
> verbliebenen 16 `✗`-Zeilen in **I Reproduzierbarkeit** (12), **S Demonstrator** aus
> Reg. `07` (S36–S38) und **S12** (Reg. `09`).

> **Fortschreibung 2026-08-05 (Modus A, Registerabschnitt [10 Konfiguration](10_konfiguration.md)).**
> Dieser Registerabschnitt war bereits weitgehend abgearbeitet: von den **37 Matrixzeilen mit
> *Reg.* `10`** stand genau **eine** offen — **Q16** (Umgebungsbrücke der Pfade). Sie ist
> überführt und steht jetzt auf `○`. `~` kommt in der Matrix nicht mehr vor; die drei
> `!`-Zeilen dieses Registerabschnitts — **A14** (zwei Datenstände der Splitgrößen), **A22**
> (Statuswiderspruch SWAN-DF) und **D2** (Phase-2-Batchaufteilung Video) — sind bewusst
> unangetastet und im Fließtext zu korrigieren.
>
> **Der eigentliche Ertrag dieser Runde sind sieben neu angelegte Zeilen: D36–D40, Q22, Q23.**
> Beim vollständigen Lesen von [10](10_konfiguration.md) fielen — wie zuvor bei
> [07](07_inference_pipeline.md) — Mechanismen auf, die dort beschrieben sind, aber **keine
> Matrixzeile** hatten. Sie standen damit weder auf `✗` noch auf `○` und wären von keinem
> Durchgang erfasst worden:
>
> - **D36** — multimodales Mixup teilt `lam` und Permutation über beide Modalitäten, damit die
>   A/V-Paarung erhalten bleibt. D14 nennt nur Mixup allgemein.
> - **D37** — die Kopfkommentare der Phase-2-Konfigurationen halten je einen eigenen Anlass
>   fest. Warum es Phase 2 gibt, stand nirgends.
> - **D38** — Vorgeschichte von `max_epochs: 30` und `patience: 5`; D18 führt nur die Werte.
> - **D39** — Checkpoint-Namensvorlage und `save_top_k: 1`; die 05-Skizze zitiert die
>   val-AUC-Werte der Dateinamen, ohne ihre Herkunft zu nennen.
> - **D40** — `train_audio_smoothing.yaml` als einziger Arm, der eine der drei
>   Regularisierungsmaßnahmen allein führt (Gegenprobe zu D30).
> - **Q22** — 27 Trainingskonfigurationen gegen acht dokumentierte Läufe.
> - **Q23** — die Klassengewichte in den YAML sind Momentaufnahmen und widersprechen sich
>   (7,361 gegen „~8,7“); die 05-Skizze führt sie bereits als „zuletzt gemessene Gewichte“.
>
> Anders als S36–S38 sind diese sieben **im selben Zug überführt** und stehen auf `○`. Der
> gesamte Ertrag landet in Kapitel 5, dazu eine Anhangzeile (Q16 in §A) — Kapitel 4 bekommt
> aus diesem Registerabschnitt **keinen** Punkt: Konfigurationen beschreiben Laufbedingungen,
> nicht Designbegründungen. Einzige Grenzentscheidung ist D36, dessen Begründung inhaltlich
> nach Kapitel 4 gehörte; **Kapitel 4 führt Mixup aber an keiner Stelle ein**, weshalb der
> Halbsatz dort bleibt, wo Mixup im Beleg tatsächlich steht.
>
> Sechs weitere Mechanismen sind geprüft und **bewusst ohne Matrixzeile geblieben**
> (Begründungen im 05-Skizzenblock): `log_model: False` in `logger/wandb.yaml`, die vier
> übrigen `debug/`-Konfigurationen neben D34, die nie angesteuerten Trainer-Varianten
> `cpu`/`mps`/`ddp`/`ddp_sim`, der `mnist`-Zweig, `ckpt_path: ???` in `configs/eval.yaml` und
> `extras/default.yaml`.
>
> Gegen den Code verifiziert wurden alle sieben neuen Zeilen und Q16:
> `configs/paths/default.yaml:4-19` (Q16), `multimodal_module.py:524,527` (D36),
> `train_video_phase2.yaml:3-5` (D37), `trainer/default.yaml:6-9` und
> `callbacks/default.yaml:21-23` (D38), `callbacks/default.yaml:10-17` plus
> `model_checkpoint.yaml:10` (D39), `wav2vec2_module.py:177` und
> `train_audio_mixup.yaml:25` (D40), die acht Notizen unter
> `vault/Research/deepfake-detection/Results/` (Q22).
>
> ~~**Ein Widerspruch innerhalb des Registers, der beim Ausschreiben zu beachten ist.**
> [10 §4](10_konfiguration.md) beschreibt die Fusions-Ablationen mit „bei `video_only`/
> `audio_only` wird der Pool-Vektor der jeweils anderen Modalität genullt“.~~ —
> **behoben 2026-08-06.** [10 §4](10_konfiguration.md) sagt jetzt, was der Code tut: der
> Backbone der verworfenen Modalität wird **gar nicht erst ausgeführt**
> (`_extract_features` → `None`, `multimodal_module.py:384-391`), so wie Zeile **C15** es
> festhält. **Offen bleibt der Beleg selbst:** dieselbe Nullungsformulierung steht in
> `docs/kapitel/04Methodology.tex` und ist dort in Modus C zu präzisieren — der
> Klassifikatoreingang ist in beiden Fassungen derselbe, der Rechenweg nicht, und nur die
> korrigierte Fassung trägt das Argument „Beitrag des Signals, nicht einer kleineren
> Architektur“.
>
> Die Zählung oben ist entsprechend nachgezogen (vorher 273 Zeilen, 210 `○`, 16 `✗`).

> **Fortschreibung 2026-08-06 (Modus A, Registerabschnitt [11 Infrastruktur](11_infrastruktur.md)).**
> Bearbeitet wurden die acht Zeilen des Matrixabschnitts **I**, deren Spalte *Reg.* auf
> [11](11_infrastruktur.md) verweist und die auf `✗` standen: Q5, Q6, Q9, Q11, Q12, Q13, Q15,
> Q19. Alle acht sind überführt und stehen jetzt auf `○`. Die übrigen vier 11-Zeilen (Q7, Q8,
> Q14, Q16) standen bereits auf `○`; `!` und `~` kommen in diesem Registerabschnitt nicht vor.
>
> Der Ertrag teilt sich nach Aussagetyp und weicht dafür zweimal von der Spalte *Kap.* ab:
> **Kapitel 5** bekommt die drei Laufbedingungen (Q12 gemischte Versionsbindung an die
> Stack-Tabelle, Q15 der Launch-Shim an den W&B-Punkt, Q13 die stille Wort-Zeitleiste als
> vierte Erzeugungsbedingung der Abbildungen im §Laufzeitkonfiguration);
> **Kapitel 9 §G** bekommt den Qualitätssicherungsapparat (Q5+Q6+Q11 als ein Absatz, Q19 als
> Halbsatz an C12, Q9 als eigener Punkt zum Container).
>
> Vier Zeilen wurden gegen den Code verifiziert: die CI läuft bei Push **und**
> Pull-Request und lintet nur `src/ tests/` (`.github/workflows/ci-pipeline.yml:4,49-50`),
> `fix = true` steht in `pyproject.toml:4` und die `per-file-ignores` für `src/models/*` in
> `:19-23` (Q11/Q19), Pre-Commit pinnt v0.11.0 gegen das ungepinnte `ruff` aus
> `requirements-dev.txt:1` (Q6/Q11/Q12), und der WhisperX-`ImportError` wird in
> `src/api/inference.py:2090-2092` auf `debug` protokolliert (Q13).
>
> Für drei Teilzuordnungen ist `–` vorgeschlagen: die Kapitel-9-Anteile von Q12, Q13 und
> Q15. Alle drei sind Laufbedingungen; eine zweite Fassung im Anhang wäre eine Doppelung.
> Q19 ist zusätzlich als Streichkandidat markiert, aber nur **gemeinsam mit C12** — der
> Halbsatz steht ohne die Laufzeit-Formprüfung ohne Bezug.
>
> **Ein Widerspruch, der beim Ausschreiben zu beachten ist.** `CLAUDE.md` verlangt eine
> Zeilenlänge ≤ 88, `pyproject.toml` setzt `line-length = 120` und schaltet `E501` zusätzlich
> ab. Maßgeblich ist die durchgesetzte Konfiguration. Falls der Beleg Codekonventionen nennt,
> ist `pyproject.toml` die Quelle — nicht `CLAUDE.md`.
>
> **Ein Mechanismus ohne Matrixzeile** (vgl. Befund 4): `.env` und `.env.example` sind beide
> unversioniert. Damit fehlt im Klon die Vorlage für genau die elf Backend-Variablen, deren
> Tabelle S3 für Anhang G vorsieht. Die Zeile ist **nicht** angelegt — Rückfrage, ob daraus
> Q24 werden soll oder ob der Halbsatz in den S3-Eintrag gehört.
>
> Die Zählung oben ist entsprechend nachgezogen (vorher 218 `○`, 15 `✗`).

> **Fortschreibung 2026-08-06 (Modus A, Vollständigkeitskontrolle gegen
> [01](01_datenpipeline.md) und [09](09_tests.md)).** Diese Runde hat **nichts überführt**,
> sondern geprüft: Steht in den Registerdokumenten etwas, das keine Matrixzeile hat? Für
> [01](01_datenpipeline.md) lautet die Antwort **ja, zehnmal** — A41 bis A46 und B17 bis
> B20 sind neu angelegt. Sieben stehen auf `✗`, drei auf `–`. [09](09_tests.md) ist
> dagegen **vollständig abgedeckt**: alle 13 als methodisch markierten Tests haben eine
> Entsprechung (F5, F12, C7, D4, H15, B1, A17, E5, A10) oder laufen über die Sammelzeile
> Q4; die einzigen zwei Kandidaten aus den Abschnitten 2–6 sind A44 und A46 und damit
> bereits erfasst.
>
> **Der größte Einzelfund ist B17.** Die 05-Skizze führt für den `robust`-Arm nur JPEG
> 30--90, Blur 0,5--2 und Downscale 0,5--0,9. Es fehlen **sämtliche
> Ziehungswahrscheinlichkeiten** (p = 0,3 je robuste Korruption, p = 0,5 für Flip,
> Polaritätsumkehr, Rauschen und Zeitmaskierung), der Jitterbereich 0,8--1,2, die
> Crop-Seitenskala 0,9--1,0, der SNR-Bereich 15--40 dB und die Maskenlänge 5--10 %. In
> dieser Form ist der Augmentierungsarm nicht reproduzierbar.
>
> **Vier Punkte wurden geprüft und bewusst *nicht* angelegt**, weil sie bereits in den
> Skizzen stehen: die Reihenfolge Augmentierung-vor-z-Normierung (B1,
> `04Methodology.tex:579`), gzip-4 und die LZF-Umstellung (A24,
> `09Appendix.tex:113-116`), `data/normalized/` als Abspiel- und Sweepgrundlage (A2,
> `04Methodology.tex:542`) und die korrekte Siebener-Regionsliste
> (`04Methodology.tex:366-371`). Ebenfalls verworfen: die DataModule-Vorgabewerte der
> Batchgröße (8/32/4) — jede Experimentkonfiguration überschreibt sie, die Skizze führt
> die gefahrenen Werte, die Vorgabe ist eine nie gelaufene Zahl.
>
> **Die drei `✗`-Stichproben sind gegengeprüft** und zu Recht offen: S38 erscheint in
> `04Methodology.tex:1236` nur als *Quellenangabe* unter G6+G7, nicht als beschriebener
> Mechanismus; Q20 wird in `05Experimental_Setup.tex:1405` nur als Querverweis genannt;
> S12, S36, S37, Q17 und Q21 haben in keiner Kapiteldatei einen Treffer.
>
> **Die drei registerinternen Widersprüche sind in derselben Runde behoben** (Details
> unten unter *Zustandsbefunde*): die Breaking-Point-Definition in
> [05](05_robustheit_adversarial.md), die Nullungsaussage zu den `*_only`-Modi in
> [10 §4](10_konfiguration.md) und der veraltete L3-Rückfallgrund in
> [08 §4](08_frontend.md).
>
> Die Zählung oben ist entsprechend nachgezogen (vorher 280 Zeilen, 7 `✗`, 1 `–`).

> **Fortschreibung 2026-08-06 (Modus A, Vollständigkeitskontrolle gegen
> [04](04_xai.md)).** Wieder keine Überführung, sondern eine Prüfung — und wieder mit
> Ertrag: **acht Mechanismen ohne Matrixzeile**, angelegt als F49 bis F56 (7 × `✗`,
> 1 × `–`). Damit hat der Abschnitt F 58 Zeilen und ist der größte der Matrix.
>
> **Der wichtigste Fund ist F49, weil er drei bekannte Befunde erklärt.** Die Audio-xAI
> existiert zweimal: `src/utils/audio_xai.py` bedient die drei Offline-Erklärskripte,
> `src/api/inference.py` implementiert Wortaggregation und Frequenzbänder **eigenständig
> neu**. Geteilt ist allein der Relevanzkern `attnlrp.py`. F26 (Single-Seed gegen
> bivariat), F27 (überholte L3-Formel) und F28 (verschiedene Farbskalen) sind damit keine
> drei Einzelfälle, sondern drei Symptome einer Codeverdopplung. **Daraus folgt eine
> Prüfaufgabe:** die Zeile nennt auch die Wortaggregation als eigenständig
> reimplementiert — ob die Laufzeit dieselbe Formel wie `aggregate_word_relevance` nutzt,
> ist offen und berührt direkt den `!`-Widerspruch F14.
>
> **Zwei Funde betreffen die Lesbarkeit von Abbildungen** und gehören damit in dieselbe
> Muss-Gruppe wie S18/S26/F26: F50 (die Skriptfiguren erklären die *vorhergesagte* Klasse
> — bei einem REAL-Clip zeigt die Karte Evidenz für REAL, nicht für FAKE) und F52 (die
> gezeigte Wellenform ist das z-normalisierte Modelleingangssignal, keine Lautstärke).
> F53 ergänzt S28/S35 um eine **dritte** Farbkonvention.
>
> **Ein Kandidat wurde geprüft und *nicht* angelegt:** die Reichweite der gemeinsamen
> lxt-Patch-Map (GELU, LayerNorm, Dropout; Softmax und Matmul modellspezifisch) steht
> bereits in `02Tech_Explanations.tex:402-403`. Ebenfalls dort belegt: der gemeinsame
> multimodale Rückwärtspass samt Begründung (`:408-409`, deckt F10).
>
> Die Zählung oben ist entsprechend nachgezogen (vorher 290 Zeilen, 14 `✗`, 4 `–`).

> **Fortschreibung 2026-08-06 (Modus A, Vollständigkeitskontrolle gegen
> [08](08_frontend.md)).** **Neun Mechanismen ohne Matrixzeile**, angelegt als S39 bis
> S47 (8 × `✗`, 1 × `–`). Der Abschnitt S hat damit 47 Zeilen.
>
> **Der wichtigste Fund ist keine neue Zeile, sondern eine Ursache.** Die falsche
> Regionsliste des Belegs (`!`-Zeile **F18**) ist keine Formulierungsschwäche, sondern
> eine **falsche Quelle**. Die Fünferliste „Mouth, Eye, Jaw, Shoulder, Background“ stand
> an zwei Stellen im Projekt, keine davon maßgeblich:
>
> - **[`docs/archive/adversarial.md` §2.1](../archive/adversarial.md)** — wörtlich
>   dieselbe Aufzählung in einem Planungsdokument. Das ist die **wahrscheinlichere**
>   Quelle des Belegsatzes, weil die Reihenfolge exakt übereinstimmt.
> - **`frontend/src/lib/mockData.ts`** — die Attrappenzeilen der Shift-Tabelle erfanden
>   `Background` und `Shoulder` als Regionsnamen.
>
> **Beide sind am 2026-08-06 bereinigt** (Auftrag des Autors): die Mockzeilen nutzen
> jetzt reale Regionsnamen, `bshift` trägt die kanonische Liste im Docstring, und das
> Archivdokument hat einen Korrekturhinweis. Damit erzeugt keine Stelle im Repositorium
> mehr die beiden erfundenen Namen. **Der Belegsatz in `04Methodology.tex:267` bleibt zu
> korrigieren** — das ist Modus C und nicht Teil dieser Runde.
>
> **Folge für S11:** Von den drei genannten Erkennungsmerkmalen einer Mock-Abbildung
> entfallen zwei. Übrig bleibt das verlässlichste — die einfarbigen 224×224-Kacheln ohne
> räumliche Struktur —, dazu die auf fünf verkürzte Regionsliste und die Statusanzeige
> der Kopfzeile (S39). Die 05-Skizze ist entsprechend nachgezogen.
>
> **Vier Funde betreffen unmittelbar die Lesbarkeit von Abbildungen:** S40 (die
> Overlay-Deckkraft ist ein Bedienparameter mit Startwert 0,85, keine feste Eigenschaft),
> S41 (die Relevanzzeitreihe sättigt ab 0,25, und die beiden Zeitreihen dürfen
> unterschiedlich lang sein), S42 (der Vergleichsspieler zeigt den **Crop-Raum**, die
> Hauptansicht die rückprojizierte Karte — nicht dieselbe Darstellung) und S43 (die
> beiden L3-Ansichten sind mit `dirGain 1,4` gegen `4` verschieden stark verstärkt, was
> F37 erst lesbar macht: blass **trotz** dreifacher Verstärkung).
>
> **Zwei Funde sind Ehrlichkeitspunkte:** S44 — das Regionspanel verschwindet lautlos,
> wenn `regionRelevance` leer ist, ein Screenshot ohne Gesichtsschema belegt also nichts;
> und S39 — die Statusanzeige der Kopfzeile ist die Stelle, an der sich die Einhaltung der
> Erzeugungsvorschrift im Bild **nachprüfen** lässt.
>
> **Ein Kandidat wurde geprüft und *nicht* angelegt:** die Erkennungsmerkmale der
> Mock-Attrappe stehen bereits im S11-Eintrag der 05-Skizze
> (`05Experimental_Setup.tex:1141-1144`) — neu war allein die Verbindung zu F18, und die
> hat den Eintrag inzwischen selbst verändert (s. o.).
>
> Die Zählung oben ist entsprechend nachgezogen (vorher 298 Zeilen, 21 `✗`, 5 `–`).

> **Fortschreibung 2026-08-06 (Modus A, Vollständigkeitskontrolle gegen
> [06](06_backend_api.md)).** **Acht Mechanismen ohne Matrixzeile**, angelegt als S48 bis
> S55 (7 × `✗`, 1 × `–`). Der Abschnitt S hat damit 55 Zeilen und ist der zweitgrößte der
> Matrix.
>
> **Zwei Funde betreffen die Vergleichbarkeit von Zahlen und sind die wichtigsten:**
>
> - **S48** — unimodale und multimodale Analyse desselben Clips starten von
>   **verschiedenen Daten**: unimodal aus dem HDF5, multimodal aus dem neu dekodierten
>   Rohvideo (weil die Tonspur gebraucht wird). Zusammen mit F45 (nicht bitgleiche
>   Interpolation) heißt das: dieselbe Clip-ID, nicht dieselben Pixel. Jeder Vergleich der
>   beiden Modi im Demonstrator trägt diesen Unterschied mit.
> - **S51** — die Regionswerte des Demonstrators sind ein **Clipmittel über alle Frames**.
>   Das widerspricht F25a **nicht**: der Clipmittelwert ist das gewichtete Mittel aus
>   Fake-Fenstern und Restclip, 16,7 % liegt zwischen 17,4 % und 16,5 %. Der Punkt ist,
>   dass das Panel den zeitlichen Kontrast **wegmittelt**, auf dem der Kernbefund beruht —
>   ein Screenshot davon kann die Lokalisierungsaussage weder zeigen noch stützen. In der
>   Bildunterschrift gehört deshalb der Zusatz, dass die Anteile für den ganzen Clip gelten.
>
> **Zwei Funde sind Ehrlichkeitspunkte:** S49 (Gesichtsverlust führt unimodal zu einem
> Vollbildrückfall, multimodal zu HTTP 500 **ohne** Rückfall — derselbe Clip kann in einem
> Modus ein Ergebnis liefern und im anderen scheitern) und S50 (ein Phase-3-Ergebnis
> spiegelt `upscale` und `audio_bitrate` **nicht** zurück; aus einem gespeicherten Ergebnis
> ist nicht ablesbar, ob die Skalierungsachse aktiv war).
>
> **Zwei sind Laufbedingungen:** S52 (drei stille Degradationen bei fehlender `clips.json`
> oder fehlendem `data/normalized/`, dazu der fest konstruierte Videopfad) und S53 (je
> Router ein eigener Einzelthread-Executor — Analyse, Phase 3 und Phase 4 können sich auf
> der GPU überlappen, was jede Laufzeitmessung im Demonstrator bedingt).
>
> **S54** schließt S20: abwärtskompatible Vorgabewerte halten alte Cachedateien gültig,
> aber eine fehlgeschlagene Validierung verwirft sie **still**. Beide Hälften gehören in
> denselben Satz.
>
> Die Zählung oben ist entsprechend nachgezogen (vorher 309 Zeilen, 30 `✗`, 6 `–`).

> **Fortschreibung 2026-08-06 (Modus A, Vollständigkeitskontrolle gegen
> [12](12_dokumentation_vault.md)).** **Sieben Mechanismen ohne Matrixzeile**, angelegt als
> Q24 bis Q30 (6 × `✗`, 1 × `–`). Der Abschnitt I hat damit 30 Zeilen.
>
> **Der gefährlichste Fund ist Q25 — `0,976` bezeichnet zwei verschiedene Größen.** Im
> Beleg steht die Zahl an drei Stellen als **Test-`auc_video` des eingefrorenen
> Audiomodells** (`06Results.tex:43`, `08Conclusion.tex:23`, `00Abstract.tex:24`). In
> [12 §3.2](12_dokumentation_vault.md) bezeichnet dieselbe Zahl die **Val-AUC im
> Dateinamen des multimodalen Fusions-Checkpoints**. Dazu kommt, dass für die Fusion zwei
> Werte kursieren: Test 0,960 gegen Val 0,976. Zwei Modelle, zwei Metrikebenen, eine Zahl —
> wer die Checkpointdateinamen als Ergebniszahlen zitiert, berichtet Validierungswerte als
> Testwerte.
>
> **Q24 füllt einen offenen Platzhalter.** Die 05-Skizze hält eine Tabelle der
> Trainingszeiten frei und vermerkt \enquote{noch mit echten Werten füllen}; die Werte
> stehen in der Baseline-Notiz (eingefroren 20 Epochen / ~41\,h, entfroren 86.228.738
> Parameter, 12 Epochen / ~30\,h).
>
> **Drei Funde betreffen dieselbe Fehlerklasse wie die Archivprüfung.** Q26: `vault/Writing/`
> hält 12 Textentwürfe, die in mindestens fünf Skizzenblöcken als `QUELLE` zitiert werden,
> ohne dass geprüft wäre, ob Entwurf oder `.tex` aktueller ist — und sie tragen **keine**
> Warnung. Q27: `docs/kapitel/archiv/` enthält 10 ältere Kapitelfassungen, in denen die am
> selben Tag korrigierte Hintergrund-Erzählung weiterhin steht (`00abstract.txt:24`,
> `06Results.txt:32`). Q28: `superpowers/plans/2026-06-15-gpu-side-normalization.md` ist
> ein 18-KB-Planungsdokument mit ungeprüftem Umsetzungsstand — potenziell ein zweiter
> Fall F57.
>
> **Q29 hält eine Bezeichnerkollision fest**, die beim Zitieren zwischen Register und
> Beleg auffällt: `G1`–`G17` sind hier Phase-3-Zeilen, im Beleg sind `G1`–`G5` die
> Forschungslücken; `H2` ist hier ein PGD-Angriffsziel, in `frontend_roadmap.md` ein
> Oberflächenkürzel.
>
> **Eine Beobachtung ohne eigene Zeile:** Die sieben Einträge der Tabelle
> *\enquote{Zahlen aus dem Vault, die nicht ungeprüft in `06Results.tex` dürfen}* sind
> inhaltlich vollständig, haben aber **keine Zeilen-ID und keinen Status**. Sie lassen sich
> damit nicht als erledigt abhaken. Falls Kapitel 6 ausgeschrieben wird, ist das die
> Gruppe, die eine eigene Nachverfolgung braucht.
>
> Die Zählung oben ist entsprechend nachgezogen (vorher 317 Zeilen, 37 `✗`, 7 `–`).

> **Fortschreibung 2026-08-06 (Modus A, Vollständigkeitskontrolle gegen
> [05](05_robustheit_adversarial.md)).** **Acht Mechanismen ohne Matrixzeile**, angelegt
> als G18 und H29 bis H35 (7 × `✗`, 1 × `–`).
>
> **H29 ist der schwerwiegendste Fund der gesamten Kontrollreihe.**
> `num_adversarial_samples` rechnet `batch_size // 2` und rundet ab; die Mischung ersetzt
> die erste Hälfte des **Mikro**batches in `training_step`, `accumulate_grad_batches`
> gleicht das nicht aus. `configs/experiment/train_multimodal_adversarial.yaml:22` setzt
> `batch_size: 1`. Daraus folgt: **null adversariale Samples je Schritt** — der Lauf
> trüge den Namen \enquote{adversarial} und trainierte vollständig auf sauberen Daten.
> Die beiden anderen Arme sind unauffällig (Video 2 → 1 von 2, Audio 16 → 8 von 16).
> Der Befund berührt **H13**, das auf `✓` steht: die Beschreibung \enquote{1:1-Mischung}
> in `04Methodology.tex` §4.2 stimmt für Video und Audio, für den multimodalen Arm nicht.
> Ob der Lauf je gefahren wurde, ist offen — laut H16 und Q22 existiert dafür keine
> Ergebnisnotiz.
>
> **H31 schließt eine Lücke in der Transferaussage.** H8 nennt den UAP-Transfer auf
> ungesehene Clips als Zweck, H18/H19 beschreiben die Anpassungsmenge. Dass **auf
> `train_metadata.csv` angepasst und auf `test_metadata.csv` ausgewertet** wird, stand
> nirgends — es ist aber genau die Bedingung, unter der \enquote{Transfer} überhaupt eine
> Aussage ist und nicht zirkulär.
>
> **H32 ist eine herstellbare Abbildung für eine Phase ohne Ergebnisse.** `compute_uap.py`
> speichert δ\* nicht nur als Tensor, sondern rendert den Videoanteil als PNG (Mittel über
> Frames und Kanäle, `seismic`, symmetrisch um 0) und protokolliert das **gemessene L∞
> gegen das Budget**. Eine universelle Störung sichtbar zu machen ist anschaulicher als
> jede Fooling-Rate-Tabelle.
>
> **Zwei Funde betreffen die Belastbarkeit der Tabellen:** G18 (die Clipzahl je
> Gitterpunkt ist keine Konstante — Clips mit gescheiterter Inferenz fallen samt
> Baselineeintrag heraus) und H34 (im voreingestellten Adversarialsweep spiegelt das
> Audiobudget den Video-ε-Wert, die getrennten Budgets aus H7 sind dort also gekoppelt).
>
> **Zwei Statuskorrekturen aus der Vorrunde sind in dieser Zählung nachgezogen:** Q27 und
> Q29 stehen jetzt auf `–` statt `✗`. Beide sind erledigt, aber nicht als Belegtext —
> Q27 als Agentenwarnung (`docs/kapitel/archiv/README.md`, `CLAUDE.md`), Q29 als
> Zitierkonvention ([README](README.md)). Dazu kam Q31 (`○`) für die Checkliste V1–V8.
>
> Die Zählung oben ist entsprechend nachgezogen (vorher 325 Zeilen, 41 `✗`, 10 `–`).

> **Fortschreibung 2026-08-06 (Modus A, Vollständigkeitskontrolle gegen
> [03](03_training_evaluation.md)).** **Acht Mechanismen ohne Matrixzeile**, angelegt als
> B21, D41–D44 und Q32–Q34 (7 × `✗`, 1 × `–`). Kein Fund widerspricht dem Beleg; alle acht
> sind **Begründungen und Absicherungen**, die bestehende Zeilen tragfähig machen.
>
> **Drei schließen Lücken in bereits geplanten Absätzen:** D41 (der Lernratenplan skaliert
> **je Parametergruppe** — nur deshalb vertragen sich D7 und D9 überhaupt), D43 (der
> LoRA-Export lädt den besten Checkpoint **frisch**, weil das Live-Modul die Gewichte der
> letzten Epoche hält — ohne das exportierte D12 das falsche Modell) und B21
> (`vision_constants.py` ist die **einzige** Quelle der ImageNet-Statistik, per Suche
> verifiziert; das ist der Mechanismus hinter der Train/Serve-Parität aus B1 — sie ist
> nicht nur getestet, sondern konstruktiv erzwungen).
>
> **D42 ist eine harte Reproduktionsabhängigkeit:** `add_safe_globals([linear_warmup_cosine])`
> am Modulende macht den Scheduler selbst ladbar, weil das Hydra-Partial in den
> Lightning-Hyperparametern und damit in **jedem** Checkpoint landet. Wer die Funktion
> umbenennt oder verschiebt, macht alle bestehenden Checkpoints unladbar. Das ist eine
> andere Registrierung als die sechs Einstiegspunkte aus Q17.
>
> **Q32 sammelt drei Guards, die Zufallszahlen als Messwerte verhindern** — der stärkste
> ist der `ValueError` in `eval.py` bei fehlendem `ckpt_path`: ohne ihn würde ein frisch
> initialisiertes Modell evaluiert und sein Rauschen als Ergebnis ausgewiesen. Q33 und Q34
> erklären zwei Dinge, die sonst unmotiviert wirken: warum `config_tree.log` selbsttragend
> ist (`resolve=True`) und warum an den `explain.py`-Aufrufen `extras.enforce_tags=false`
> steht.
>
> Die Zählung oben ist entsprechend nachgezogen (vorher 333 Zeilen, 48 `✗`, 11 `–`).

> **Fortschreibung 2026-08-06 (Modus A, Vollständigkeitskontrolle gegen
> [02](02_modelle.md)).** **Neun Mechanismen ohne Matrixzeile**, angelegt als C16–C19 und
> D45–D49 (8 × `✗`, 1 × `–`). Wie schon bei [03](03_training_evaluation.md) widerspricht
> kein Fund dem Beleg — es sind durchweg **Wirkbereiche und Bedingungen**, die bestehende
> Zeilen erst überprüfbar machen.
>
> **C16 beantwortet eine Frage, die C1 offenlässt.** C1 nennt `use_mean_pooling=True` als
> Kopfvariante; der Grund ist, dass VideoMAE **kein CLS-Token hat**. `last_hidden_state`
> ist die volle Patch-Token-Sequenz der Länge **1568** = 8 zeitliche × 14 × 14 Patches.
> Dieselbe Zerlegung erklärt das 16×16-Patch-Pooling aus F9 und die zeitliche Auflösung
> der Videorelevanz.
>
> **C17 ergänzt F10 um seine zweite Hälfte:** Für die multimodale Erklärung wird **auch
> der Fusionskopf** gepatcht, nicht nur die beiden Backbones — er ist kein
> HuggingFace-Modul und fiele sonst durch das Patching hindurch.
>
> **Drei Funde beschreiben stille Wirkungslosigkeit** — dieselbe Klasse wie Q20:
> D45 (LLRD wird ohne Meldung übersprungen, wenn der Backbone eingefroren ist; ein
> `llrd_decay` in einer Phase-1-Konfiguration bliebe wirkungslos), D46 (Gradient
> Checkpointing ist in Phase 1 **inert**, weil die Backbones dort im `eval`-Modus laufen —
> gemessene Kosten in Phase 2: ~10 % Schrittzeit) und C18 (`normalize_video` steuert nur
> die Videoseite; die Audiorelevanz wird im Einzelziel-Pfad immer normiert, was die
> Skalengleichheit aus F10 dort einschränkt).
>
> **D47 und D48 sind Interpretationsbedingungen:** der PGD-Angriff des adversarialen
> Trainings läuft im `eval`-Modus, sonst wäre Dropoutrauschen Teil des Gradienten; und
> unter Mixup werden die **Metriken gegen die unpermutierten Labels** berichtet — ohne
> diesen Halbsatz ist die Trainingsgenauigkeit eines Mixup-Laufs nicht mit den übrigen
> Armen vergleichbar.
>
> **D49 schließt an D42 an:** Klassengewichte werden zu einer reinen float-Liste
> normalisiert, weil ein OmegaConf-Objekt in den Hyperparametern das Laden unter
> `weights_only=True` bricht. Beide sind Bedingungen dafür, dass die trainierten
> Checkpoints überhaupt ladbar bleiben, und gehören in denselben Absatz.
>
> Die Zählung oben ist entsprechend nachgezogen (vorher 341 Zeilen, 55 `✗`, 12 `–`).

> **Fortschreibung 2026-08-06 (Modus A, Vollständigkeitskontrolle gegen
> [00](00_inventar.md)) — Abschluss der Kontrollreihe.** **Drei Mechanismen ohne
> Matrixzeile**, angelegt als Q35–Q37. Das Dokument ist überwiegend Zählwerk, liefert aber
> einen Zahlenkonflikt und zwei fehlende Umfangsangaben.
>
> **Q36 ist der eigentliche Fund:** [00 §5](00_inventar.md) beziffert `data/` mit
> **59.894 Dateien / 11,3\,GB**, Zeile **Q14** den DVC-Hash über **59.777 Dateien /
> 10,7\,GB** — eine Differenz von 117 Dateien und 0,6\,GB. Q14 führt den Hash als
> \enquote{zitierfähige Kennung des Datenstands}; deckt er den Bestand nicht vollständig,
> trägt er diese Rolle nur eingeschränkt. Vor jeder Angabe im Setup-Kapitel zu klären.
>
> **Q35 und Q37 liefern Zahlen, die der Beleg bisher nicht hat:** den Umfang der
> Implementierung (485 Projektdateien, 110 Python-Module mit 25.245 Zeilen, 61 TS/TSX mit
> 11.019) und die Verteilung des Codegewichts. Q37 hat dabei einen unmittelbaren Bezug zur
> **Umfangsentscheidung**: Die beiden größten Frontend-Module sind `RobustnessPanel.tsx`
> (780 Z.) und `AdversarialPanel.tsx` (726 Z.) — also genau die Phasen, deren Gewicht im
> Beleg reduziert werden soll. Der gebaute Apparat ist dort groß, die Ergebnisse fehlen;
> genau dafür sieht Befund 2 den Apparat als Berichtsgegenstand vor.
>
> **Ein überholter Zustandsbefund ist bei der Gelegenheit korrigiert:** Der Eintrag zur
> Frontend-Zeilenzahl behauptete, [00 §3](00_inventar.md) nenne 10.994 und sei nicht
> nachgezogen. Das trifft nicht mehr zu — [00 §3](00_inventar.md) führt für TS/TSX
> überhaupt keine Zeilenzahl mehr, nur die Modulzahlen (49 `.tsx` + 12 `.ts` = 61).
>
> Die Zählung oben ist entsprechend nachgezogen (vorher 350 Zeilen, 63 `✗`, 13 `–`).

> ### Umfangsentscheidung des Autors (2026-08-06) — bindend für die Abschnitte G und H
>
> **Der Projektumfang ist verkleinert worden. Robustheit (Phase 3) und Adversarial
> (Phase 4) werden nicht weiter bearbeitet**, und auch im Beleg soll auf ihnen **weniger
> Gewicht** liegen. Konkret bestätigt: der multimodale Adversariallauf wurde nicht
> ausgeführt und wird es voraussichtlich auch nicht (vgl. H29).
>
> **Das ändert die Reihenfolge, nicht den Auftrag.** Zuerst wird **alles einmal
> ausformuliert** — erst danach ist entscheidbar, was im Hauptteil bleibt, was in den
> Anhang wandert und was gestrichen wird. Vorher zu kürzen hieße, ohne Textgrundlage zu
> entscheiden.
>
> **Praktische Folgen für dieses Register:**
>
> - Die `○`-Zeilen der Abschnitte **G** (17) und **H** (24) bleiben gültig und sind
>   auszuschreiben. Sie werden **nicht** vorab gestrichen.
> - Für die Triage danach ist die Spalte *Umfang* je Zeile der Ansatzpunkt: Halbsätze und
>   Tabellenzeilen überleben eine Kürzung eher als eigene Absätze.
> - Zeilen, die ausschließlich eine **nicht gefahrene** Konfiguration beschreiben (H29,
>   Teile von H12/H24/H25/H26), sind die ersten Streichkandidaten — aber erst nach dem
>   Ausformulieren.
> - **Unberührt bleibt der Apparat als Berichtsgegenstand.** Befund 2 (unten) gilt weiter:
>   Für eine Phase ohne Ergebnisse ist der gebaute Apparat genau der berichtbare Teil.
>   H32 (die δ\*-Visualisierung) ist dafür der stärkste Einzelposten.
> - **Unberührt bleiben die `!`-Zeilen G1b und H2.** Eine falsche Beschreibung wird
>   korrigiert oder gestrichen, nicht durch Umfangsreduktion geerbt.

> **Fortschreibung 2026-08-06 (Modus A, Überführung — Runde 1: Kapitel 4).** Die
> Vollständigkeitskontrolle hat 66 `✗`-Zeilen erzeugt; diese Runde überführt die
> **Kapitel-4-Gruppe** in den Skizzenblock von `04Methodology.tex` (neuer Block am
> Dateiende, überschrieben *NACHGETRAGEN … REGISTERÜBERGREIFEND*).
>
> **Zwei Zeilen waren bereits erledigt und wechseln auf `✓`:** **A41** (gespiegelte
> Augenregionen) und **A42** (Gesichtsoval als Partitionsmaske) stehen seit der Korrektur
> des `!`-Widerspruchs F18 im **Fließtext** von `04Methodology.tex:267` — sie hätten nie
> auf `✗` bleiben dürfen.
>
> **20 Zeilen wechseln auf `○`:** A47, B18, B19, B20, B21, C16, C17, C18, D47, F49, F51,
> F55, H31, H33, S36, S38, S41, S42, S48, S51.
>
> **Vier Gruppierungen**, weil die Punkte in denselben Absatz gehören: C16 mit C1, B9 und
> F9 als eine Kette (die Herleitung 8 × 14 × 14 = 1568); S48 mit S42 (zwei Stellen, an
> denen der Demonstrator nicht dasselbe zeigt); S36 mit S38 (Herkunft und Boxverfolgung
> des Vergleichsspielers); B18 mit dem bestehenden B11/B13-Absatz.
>
> **Ein Prüfauftrag ist mitgeschrieben, keine Zahl:** ob die zeitliche Auflösung der
> Videorelevanz 16 oder 8 Positionen je Chunk trägt (bei C16). Beim Audio ist die Zählung
> belegt (F30), im Video nicht.
>
> **Noch offen: 44 `✗`-Zeilen**, verteilt auf Kapitel 5 (Laufbedingungen, größte Gruppe),
> Kapitel 6 (Abbildungsherkunft), Kapitel 7 (Limitationen) und Kapitel 9 (Anhang).
> Die Zielstruktur dafür ist bestimmt und steht in dieser Fortschreibung nicht noch
> einmal — sie ergibt sich aus der Spalte *Zielabschnitt* der jeweiligen Zeilennotiz.
>
> **Nachgezogen in Runde 2** (s. u.) — die Statuszellen stehen jetzt korrekt.

> **Fortschreibung 2026-08-06 (Modus A, Überführung — Runde 2: Kapitel 5, 6, 7, 9).**
> **`✗` ist leer.** Die verbliebenen 44 Zeilen sind in die Skizzenblöcke der vier
> übrigen Kapitel überführt; jede Kapiteldatei hat dafür einen neuen Block
> *NACHGETRAGEN … REGISTERÜBERGREIFEND* am Dateiende.
>
> | Kapitel | IDs | n |
> |---|---|--:|
> | 5 Setup | B17, D41–D43, D45, D46, D48, D49, H30, H31 (Zahlenteil), H34, Q24, Q25, Q26, Q28, Q36, S37, S39, S40, S43, S45, S46, S52, S53, S54 | 25 |
> | 6 Ergebnisse | F50, F52, F53, F54, G18, H32, S50 | 7 |
> | 7 Diskussion | S12, Q21, S44, S49, H29 (bedingt) | 5 |
> | 9 Anhang | A44, Q17, Q20, Q32, Q33, Q34, Q35, Q37 | 8 |
>
> **Ein neuer Abschnitt wird vorgeschlagen:** `\section{Umfang der Implementierung}` in
> Kapitel 9, als einzige Stelle für Q35 und Q37 — die übrigen Kapitel verweisen dorthin,
> statt die Zahlen zu wiederholen (Verfallszahlen N1/N2).
>
> **Sieben Gruppierungen** statt Einzelpunkten: D41+D45 (Komposition und stille
> Abschaltung des LLRD), D42+D49 (Ladbarkeit der Checkpoints), Q26+Q28 (ungeprüfte
> Quellen als Arbeitsanweisung), S52+S46 (Inbetriebnahmebedingungen), S12+Q21
> (Abdeckungslücken), Q32+Q33+A44 (Guards gegen stille Fehlmessungen), Q35+Q37
> (Umfangstabelle).
>
> **Drei Punkte sind bedingt oder ohne Belegtext eingetragen:** H29 nur, falls Phase 4.2
> doch berichtet wird; Q26/Q28 als Prüfauftrag ohne eigenen Belegtext; Q36 erst nach
> Klärung des Zahlenkonflikts — bis dahin ist nur **eine** der beiden Dateizahlen zu
> nennen, nicht beide.
>
> **Damit ist der Ertrag der Vollständigkeitskontrolle vollständig gesichert:** alle 353
> Zeilen stehen auf `✓`, `○`, `!` oder `–`. Was jetzt fehlt, ist kein Wissen mehr,
> sondern Fließtext.

> **Fortschreibung 2026-08-06 (Verifikationslauf — Gegenprobe der Überführung).**
> Geprüft wurde nicht die Matrix, sondern ihr **Anspruch**: Für jede Zeile, deren Notiz
> eine Überführung behauptet, muss die ID in einem `% [SKIZZE]`-Block auftauchen. Dazu
> sind alle ID-Nennungen aus `docs/kapitel/*.tex` extrahiert und gegen den Sollbestand
> gestellt worden.
>
> **Alle 44 Zeilen der beiden Überführungsrunden sind nachweisbar eingetragen.** Ebenso
> die rund 60 älteren `○`-Zeilen, deren Notizen auf Inhalt statt auf IDs verweisen
> (\enquote{05-Skizze §Hyperparameter} und ähnlich) — sie stammen aus der Zeit vor der
> ID-Konvention und sind legitim gedeckt.
>
> **Sechs Zeilen waren es nicht.** Sie standen auf `○`, während ihre **eigene Notiz** den
> fehlenden Teil benannte — faktisch `~`-Fälle in einer Spalte, die `~` seit dem
> 2026-08-05 als leer meldet:
>
> | ID | Was die Notiz selbst sagte |
> |---|---|
> | **D16** | \enquote{der Konflikt mit Early Stopping fehlt} |
> | **F39** | \enquote{`language="en"` fehlt} |
> | **H12** | \enquote{die Wiederaufnahme fehlt} |
> | **Q3** | \enquote{die übrigen drei Stellen fehlen} |
> | **G16** | Notizspalte **leer** — kein Ziel angegeben |
> | **A19** | Notiz verwies auf das **Registerdokument** 07, nicht auf ein Belegkapitel |
>
> **Alle sechs sind am selben Tag nachgetragen:** F39 → 04-Skizze (Ebene 2 der
> akustischen xAI), D16 + H12 + Q3 → 05-Skizze, G16 → 06-Skizze, A19 → 09-Skizze §G.
> Die Notizen sind entsprechend korrigiert; die durchgestrichenen Passagen bleiben als
> Beleg stehen, warum die Zeile aufgefallen ist.
>
> **Zwei Lehren für die Statusspalte:**
>
> 1. `○` bedeutet \enquote{der Punkt steht in einer Skizze} — nicht \enquote{er ist
>    erkannt}. Eine Notiz, die \enquote{… fehlt} enthält, ist mit `○` unvereinbar.
> 2. Notizen, die auf \enquote{05-Skizze §…} statt auf eine ID verweisen, lassen sich
>    **nicht per Suche prüfen**. Sie sind unkritisch, aber bei ihnen fällt eine Lücke
>    erst beim Ausschreiben auf. Wer eine solche Zeile anfasst, trägt die ID nach.
>
> **Ein Restzweifel bleibt benannt:** **H1** (FGSM = PGD mit `steps=1`) verweist auf die
> 02-Skizze, die ID ist dort nicht auffindbar. Die Deckung durch Inhalt ist plausibel,
> aber nicht nachweisbar — beim Ausschreiben von Kapitel 2 zu prüfen.

**Die Verteilung je Abschnitt zeigt, wo die Arbeit steht:**

| Abschnitt | ✓ | ~ | ○ | ✗ | ! | Lesart |
|---|--:|--:|--:|--:|--:|---|
| A Daten/Preprocessing | **15** | 0 | **24** | **1** | 4 | 2026-08-04 vollständig in die Skizzen überführt; **2026-08-06 um A41–A46 erweitert** (Vollständigkeitskontrolle gegen [01](01_datenpipeline.md)), davon 3 × `✗` und 3 × `–`. Offen: A41, A42, A44 und die vier `!`-Zeilen |
| B Laden/Augmentierung | 1 | 0 | **19** | **1** | 0 | 2026-08-04 vollständig in die Skizzen überführt; **2026-08-06 um B17–B20 erweitert**, alle vier offen |
| C Architekturen | 5 | 0 | **13** | 0 | 0 | 2026-08-04 vollständig in die Skizzen überführt |
| D Training | 4 | 0 | **36** | **7** | 1 | 2026-08-04 vollständig in die 05-Skizze überführt; offen bleibt nur die `!`-Zeile D2. **2026-08-05 um D36–D40 erweitert** (aus [10](10_konfiguration.md), direkt mit überführt) |
| E Evaluation | 3 | 0 | 9 | 0 | 0 | 2026-08-04 vollständig in die Skizzen überführt |
| F xAI | 10 | 0 | **43** | **4** | 0 | 2026-08-04 vollständig in die Skizzen überführt (dazu 1 × `–`: F38); **2026-08-06 um F49–F56 erweitert** (Vollständigkeitskontrolle gegen [04](04_xai.md)), davon 7 × `✗` und 1 × `–`. Offen: F49–F55 und die `!`-Zeilen F14 und F18 |
| G Phase 3 | 1 | 0 | 16 | **1** | 1 | 2026-08-05 vollständig in die Skizzen überführt; offen bleibt nur die `!`-Zeile G1b |
| H Phase 4 | 3 | 0 | **26** | **4** | 1 | 2026-08-05 vollständig in die Skizzen überführt; offen bleibt nur die `!`-Zeile H2 |
| S Demonstrator | 0 | 0 | **40** | **13** | 0 | 2026-08-05 sind die 15 Zeilen mit Reg. `06`, die Zeile S23 (Reg. `07`) und die 13 Zeilen mit Reg. `08` überführt. **2026-08-06 um S39–S47 erweitert** (Vollständigkeitskontrolle gegen [08](08_frontend.md)), davon 8 × `✗` und 1 × `–`. Offen: S12, S36–S38, S39–S46 |
| I Reproduzierbarkeit | 0 | 0 | **21** | **13** | 0 | **2026-08-06:** die acht Zeilen mit Reg. `11` (Q5, Q6, Q9, Q11, Q12, Q13, Q15, Q19) sind in die 05- und 09-Skizze überführt. Offen bleiben Q17 und Q20 (Reg. [03](03_training_evaluation.md)) sowie Q21 (Reg. [09](09_tests.md)) |

**Drei Befunde daraus:**

1. **Der Demonstrator war praktisch unbeschrieben** (30 von 35 Zeilen `✗` zum Stand
   2026-08-01). Die 04-Skizze fordert zwar einen Abschnitt mit den Visuals V1–V10, und 08
   nennt ihn als Nebenprodukt — aber sämtliche Eigenschaften, die die Abbildungen *lesbar*
   machen (richtungslose Konfidenz, Darstellungsverstärkungen, Farbrampen, unmarkierte
   Rückfälle), fehlten. Das betrifft direkt die Bildunterschriften jedes Screenshots.
   **Erledigt am 2026-08-05** in drei Runden: die 15 Backend-Zeilen (Reg. `06`), darunter
   die richtungslose Konfidenz (S18); die einzelne Zeile aus Reg. `07` (S23, Gültigkeit
   aller multimodalen Phase-3/4-Zahlen); und die 13 Frontend-Zeilen (Reg. `08`) mit den
   Darstellungsverstärkungen (S26, S27), den abweichenden Farbrampen (S28, S35) und dem
   Erklärsystem (S9, S32, S33). Offen bleiben nur noch S12 (Reg. `09`, kein Frontend-Test)
   und die drei am 2026-08-05 neu angelegten Zeilen S36–S38 aus Reg. `07`.
2. **Phase 4 war als Apparat unbeschrieben** (19 von 28 `✗` zum Stand 2026-08-01). Dass die
   Ergebnisse ausstehen, ist sauber dokumentiert; der *gebaute* Apparat — Log-Scraper,
   Wiederaufnahme, Runbooks, UAP-Anpassungslogik, die zwei Fooling-Rate-Definitionen — kam
   nirgends vor. Für eine Arbeit, deren Phase 4 keine Ergebnisse liefert, ist genau das der
   berichtbare Teil. **Erledigt am 2026-08-05** durch die Überführung der Abschnitte G
   und H.
3. **Die `○`-Zeilen sind unkritisch, die `✗`-Zeilen sind der Ertrag.** Zum Stand 2026-08-01
   standen 74 Punkte bereits in den Skizzen und gingen nicht verloren; die damals 130
   `✗`-Zeilen wären ohne dieses Register nicht wieder aufgetaucht. Nach der Überführung der
   Abschnitte Datenpipeline, Modelle, Training und Evaluation sowie xAI (2026-08-04) waren
   es 151 `○` und 66 `✗`; nach Robustheit und Adversarial (2026-08-05) 181 `○` und 42 `✗`;
   nach dem Backend-Registerabschnitt (2026-08-05) 196 `○` und 27 `✗`; nach der
   Inferenzpipeline (2026-08-05) sind es 197 `○` und 29 `✗` — letztere Zahl steigt gegenüber
   der Vorrunde, weil in derselben Runde drei bislang unerfasste Mechanismen als S36–S38
   angelegt wurden; nach dem Frontend-Registerabschnitt (2026-08-05) 210 `○` und 16 `✗`;
   nach dem Konfigurations-Registerabschnitt (2026-08-05) sind es 218 `○` und 15 `✗` bei
   280 Zeilen — dort kamen sieben neue Zeilen hinzu, die im selben Zug überführt wurden.
   nach dem Infrastruktur-Registerabschnitt (2026-08-06) 226 `○` und 7 `✗`.
   Die verbliebenen `✗`-Zeilen liegen in zwei Abschnitten:
   **I Reproduzierbarkeit** (3 — Q17 und Q20 aus Reg. `03`, Q21 aus Reg. `09`) und
   **S Demonstrator** (4 — S12 aus Reg. `09`, S36–S38 aus Reg. `07`). `~` ist leer.

4. **Das Register hat selbst Lücken, und sie sind reproduzierbar.** Viermal ergab das
   vollständige Lesen eines Registerdokuments gegen die Matrix Mechanismen ohne Zeile:
   drei bei [07](07_inference_pipeline.md) (S36–S38), sieben bei
   [10](10_konfiguration.md) (D36–D40, Q22, Q23), einen bei
   [11](11_infrastruktur.md) (`.env`/`.env.example` beide unversioniert, `.gitignore`
   L64/L65 — die Vorlage für die elf Backend-Variablen aus S3 liegt damit in keinem Klon;
   im Beleg mit Platzhaltern für die maschinenabhängigen Werte auszuschreiben, die
   Variablen*namen* stehen im Register) und **zehn bei
   [01](01_datenpipeline.md)** (A41–A46, B17–B20, angelegt 2026-08-06). Jedes Mal lag die
   Ursache nicht im Beleg, sondern in der Matrix.

   **Der Prüfstand je Registerdokument:**

   | Dokument | geprüft | Mechanismen ohne Matrixzeile |
   |---|---|---:|
   | [01](01_datenpipeline.md) | 2026-08-06 | 10 (A41–A46, B17–B20) |
   | [02](02_modelle.md) | 2026-08-06 | 9 (C16–C19, D45–D49) |
   | [03](03_training_evaluation.md) | 2026-08-06 | 8 (B21, D41–D44, Q32–Q34) |
   | [04](04_xai.md) | 2026-08-06 | 8 (F49–F56) |
   | [05](05_robustheit_adversarial.md) | 2026-08-06 | 8 (G18, H29–H35) |
   | [06](06_backend_api.md) | 2026-08-06 | 8 (S48–S55) |
   | [07](07_inference_pipeline.md) | 2026-08-05 | 3 (S36–S38) |
   | [08](08_frontend.md) | 2026-08-06 | 9 (S39–S47) |
   | [09](09_tests.md) | 2026-08-06 | 0 |
   | [10](10_konfiguration.md) | 2026-08-05 | 7 (D36–D40, Q22, Q23) |
   | [11](11_infrastruktur.md) | 2026-08-06 | 1 (`.env`, keine Zeile angelegt) |
   | [12](12_dokumentation_vault.md) | 2026-08-06 | 7 (Q24–Q30) |
   | [00](00_inventar.md) | 2026-08-06 | 3 (Q35–Q37) |

   **ABGESCHLOSSEN 2026-08-06: alle zwölf Registerdokumente sind geprüft, 81 Zeilen
   fehlten.** Die Quote lag über alle Runden bemerkenswert stabil bei rund acht je
   Dokument — ein Hinweis darauf, dass die Lücke nicht an einzelnen Dokumenten hing,
   sondern an der Art, wie die Matrix ursprünglich entstanden ist. Zwei Ausreißer nach
   unten haben strukturelle Gründe: [09](09_tests.md) war lückenlos, weil Tests
   Mechanismen prüfen, die anderswo schon eine Zeile haben; [00](00_inventar.md) und
   [11](11_infrastruktur.md) sind Inventare mit wenigen eigenen Mechanismen.

   **Was die Kontrolle nicht geleistet hat und als Nächstes ansteht:** Die 66 `✗`-Zeilen
   sind angelegt, aber **nicht in die Kapitelskizzen überführt**. Das ist der nächste
   Modus-A-Durchgang und die Voraussetzung dafür, dass sie beim Ausschreiben auftauchen.

### Die zehn Widersprüche

Vorrangig zu klären, weil eine falsche Beschreibung schlimmer ist als eine fehlende.
**F57 ist der schwerwiegendste**, weil er als einziger bis ins Fazit reicht und eine
Abbildung fordert, die nicht herstellbar ist.

| # | Stelle im Beleg | Beleg sagt | Code tut |
|---|---|---|---|
| A11 | 04 §Labels | Fake-Anteil 5–7 % | ~6 % (`label_video`), ~7 % (`label_audio`), ~10 % (kombiniert) — der eigene `$$`-Kommentar vermutet die Verwechslung zu Recht |
| A14 | 05-Skizze | 9.482 / 1.382 / 1.471 bei 165 Identitäten | Register führt 9.959 / 861 / 1.180 bei ~30 — **zwei Datenstände**, vor Übernahme festlegen |
| A15 | 04 §Speicherung | HDF5 speichert `float32` | `uint8`; Normalisierung erst im DataLoader (~4× kleiner) |
| A22 | 03 ↔ 05/07 | Testset angekündigt ↔ „Zugang ausstehend“ | `conf/datasets/swan.yaml` + Loose-Video-Pfad existieren — Statusfrage klären |
| D2 | 05-Skizze | Phase 2 Video `2×3` | `6×1` unter SDPA; `2×3` nur noch adversarial |
| F14 | 04 §Audio-L2 | ~~Relevanz je Wort **aufsummiert**~~ | vorzeichenbehaftetes **Mittel** (Längennormierung) — **korrigiert 2026-08-06**, beide Codepfade verifiziert |
| F57 | 04 §Rollout, 06, 08, 09 | ~~Attention Rollout als **Vergleichsbasis/Referenz**, zwei geplante Vergleichstafeln~~ | **nicht implementiert** — repositoriumsweit kein Treffer für `rollout`. **Alle vier Stellen korrigiert 2026-08-06**; Rollout bleibt nur in Kapitel 2 als Erklärungsgrundlage |
| F18 | 04 §Regionen | ~~„Mund, Augen, Kiefer, **Schultern**, **Hintergrund**“~~ | sieben Landmark-Regionen, oval-maskiert, **kein Hintergrundwert**. **Korrigiert 2026-08-06** in `04Methodology.tex` samt der vier Folgestellen in `00Abstract.tex`, `01Einleitung.tex`, `06Results.tex` und `07Discussion_Limitations.tex` |
| F18 | 04 §Regionen | „Mund, Augen, Kiefer, **Schultern**, **Hintergrund**“ | sieben Landmark-Regionen (Forehead, Left/Right Eye, Nose, Mouth, Jaw, Chin), ohne Schultern und ohne Hintergrund — vom Autor selbst als falsch markiert. **Herkunft geklärt:** dieselbe Fünferliste stand im archivierten Planungsdokument und in den Mockdaten; beide Quellen sind am 2026-08-06 bereinigt, der Belegsatz noch nicht (s. Zeile F18) |
| G1b | 04 §Phase 3 | Gauß-Rauschen als Sweep-Achse | `eval_robustness_sweep.py` kennt **keinen** Rauschparameter; der Filter existiert nur interaktiv und ist gleichverteilt. **Auch die 04-Skizze irrt hier** |
| H2 | 04 §4.1 | Angriff maximiert CE gegen das **wahre Label** | Sweep und Demonstrator greifen gegen die **eigene saubere Vorhersage** an; gegen das wahre Label läuft nur das Training (4.2) |

---

## A — Datensatz und Preprocessing

| # | Mechanismus | Reg. | Kap. | Status | Notiz |
|---|---|---|---|---|---|
| A1 | AV-Deepfake1M als Primärdatensatz, Baumstruktur `{identity}/{clip}/{segment}/{variant}` | 01 | 04 | ○ | Datensatz benannt (Kap. 3/4), Baumstruktur nicht beschrieben · nachgetragen 2026-08-04 in die 04-Skizze |
| A2 | FFmpeg-Normalisierung auf 25 fps CFR + 16 kHz mono in **einem** Aufruf | 01 | 04 | ○ | Normierung beschrieben, aber nicht als **ein** FFmpeg-Aufruf · nachgetragen 2026-08-04 in die 04-Skizze |
| A3 | `reencode_crf: 18` statt libx264-Default 23 — Begründung: Default zerstört hochfrequente Fälschungsspuren | 01 / 10 | 04 | ✓ | |
| A4 | Stream-Copy (`remux_copy`) statt Re-Encode bei bereits passender Bildrate — vermeidet Generationsverlust | 01 | 04 | ✓ | |
| A5 | 16-Frame-Chunks, unvollständiger Restblock verworfen | 01 | 04 | ○ | Chunking ✓; verworfener **Restblock** fehlt (04 nennt nur Videos < 16 Frames) · nachgetragen 2026-08-04 in die 04-Skizze (gemeinsam mit A27) |
| A6 | MediaPipe FaceLandmarker, **Ablehnung des ganzen Chunks** bei einem fehlgeschlagenen Frame | 01 | 04 | ○ | Skip beschrieben; die Regel *ein* Frame ohne Landmarks ⇒ ganzer Chunk verworfen fehlt · nachgetragen 2026-08-04 in die 04-Skizze |
| A7 | **Zeitliche Box-Glättung**: Mittelung der 16 Boxen vor der Cropbestimmung (gegen Box-Jitter als Scheinsignal) | 01 | 04 | ✓ | |
| A8 | `crop_scale: 1.4`, Quadratisierung vor Resize auf 224×224 | 01 | 04 | ✓ | |
| A9 | FaceMesh-Landmarks `(16, 468, 2)` int16 im HDF5 gespeichert | 01 | 04 | ○ | Landmarks im HDF5 nirgends erwähnt — sie sind die Grundlage von F18/F20 · nachgetragen 2026-08-04 in die 04-Skizze |
| A10 | **Segmentgenaue Chunk-Labels** mit Überlappungsschwelle (0,1 s ODER 50 % der Segmentdauer) | 01 / 10 | 04 | ✓ | |
| A11 | Folge daraus: Fake-Klasse macht auf Chunk-Ebene nur ~7–10 % aus | 01 / 02 | 04, 05 | ! | 04 §Labels nennt 5–7 %; Register/Configs: label_video ~6 %, label_audio ~7 %, kombiniert ~10 %. Der eigene $$-Kommentar (04:70) vermutet die Verwechslung Video-/Chunk-Ebene zu Recht |
| A12 | Getrennte Labels je Modalität (`label`, `label_video`, `label_audio`) | 01 | 04 | ✓ | |
| A13 | **Identitätsdisjunkte Splits** über stabilen Hash, `split_seed: 11` | 01 | 04 | ✓ | |
| A14 | Konkrete Splitgrößen 9959 / 861 / 1180 Videos bei ~30 Identitäten | 10 | 05 | ! | **Zahlen weichen ab:** 05-Skizze nennt 9.482/1.382/1.471 Videos bei 165 Identitäten (gemessen), diese Zeile 9.959/861/1.180 bei ~30 (aus `conf/preprocess.yaml`). Vor Übernahme klären, welcher Datenstand gemeint ist |
| A15 | HDF5-Layout, uint8-Speicherung (Normalisierung erst im DataLoader, ~4× kleiner) | 01 | 04 | ! | **04 §Speicherung sagt `float32`** — gespeichert wird `uint8`, die Normalisierung passiert erst im DataLoader (~4× kleiner). Faktisch falsch |
| A16 | Wiederaufnehmbares Preprocessing (`skip_existing`) | 01 | 05 | ○ | nachgetragen 2026-08-04 in die 05-Skizze (gemeinsam mit A17) |
| A17 | **Paralleles Preprocessing** mit Worker-eigenem FaceExtractor; Schreiben bleibt im Hauptprozess | 01 | 05 | ○ | nachgetragen 2026-08-04 in die 05-Skizze (gemeinsam mit A16; RAM-Budget in §Hardware) |
| A18 | `validate_processed.py`: Struktur, CSV-Konsistenz, Labelverteilung, Crop-Geometrie, Pixel-/Audiostatistik, **Identitätsdisjunktheit** | 01 | 04/09 | ✓ | |
| A19 | `relabel_chunks.py` — In-Place-Neulabelung ohne Neu-Preprocessing | 01 | 09 | ○ | ~~07 nennt das Min-Overlap-Relabeling als adressierte Silent-Failure-Klasse~~ — **diese Notiz verwies auf das Registerdokument 07, nicht auf ein Belegkapitel; der Punkt war damit nirgends im Beleg verankert** · **nachgetragen 2026-08-06** (Verifikationslauf) in die 09-Skizze §G. **Warum es zählt:** die Min-Overlap-Regel (A10) wurde nachträglich eingeführt, der Datenbestand aber nicht neu erzeugt — A19 ist der Weg, auf dem der Bestand umgelabelt wurde, und damit Teil der Entstehungsgeschichte der berichteten Fake-Anteile |
| A20 | **Ablationsdatensatz** `keep_pairs` vs. `decouple_variant` (Frame-Zwillinge als Störgröße). **Beide Arme variieren zugleich die Identitätsdiversität**: ~12,5 k Videos über **alle 165 Identitäten**, gegenüber ~30 alphabetisch ersten der 12k-Baseline | 01 / 10 | 04, 06 | ○ | Zwei Variablen — im Beleg trennen · 05-Skizze §Ablationen; Status dort ehrlich als „nur keep_pairs trainiert“ |
| A21 | `ablation_stats.py` — Decoupling-Dosis quantifiziert | 01 | 06 | ○ | Die gemessene Decoupling-Dosis fehlt — ohne sie ist der Kontrollarm nicht quantifiziert · nachgetragen 2026-08-04 in die 06-Skizze; **Zahlenwert fehlt noch** (Bericht von `ablation_stats.py`) |
| A22 | **Cross-Dataset**: SWAN-DF über `preprocess_loose_videos.py` + `conf/datasets/swan.yaml` | 01 / 10 | 04, 06 | ! | **Statuswiderspruch:** 03 kündigt SWAN-DF als Testset an, 05/07 sagen „Zugang ausstehend/nicht gesichert“ — im Repo existieren `conf/datasets/swan.yaml` und der Loose-Video-Pfad. Klären, ob Daten vorliegen |
| A23 | Stratifizierte, geseedete Sweep-Stichprobe mit Fake-Anreicherung | 01 | 05 | ○ | nachgetragen 2026-08-04 in die 05-Skizze; offen: Verhältnis zu „alle 1.471 Testvideos" der Phase-3-Skizze |
| A24 | LZF- statt gzip-Kompression, mit Lesebenchmark als Entscheidungsgrundlage | 01 | 09 | ○ | nachgetragen 2026-08-04 in die 09-Skizze (§G); Kandidat für `–`, falls der Anhang gekürzt wird |
| A25 | **Stille-Ausfall-Bilanz am Laufende**: Face-Skip-Rate **getrennt je `modify_type`** (läge sie bei Fakes höher, wäre die Fake-Klasse still unterrepräsentiert); ab **5 %** unwiederbringlichem Ausfall wird die Meldung von `WARNING` auf `ERROR` hochgestuft | 01 | 04, 09 | ✓ | 04 §Gesichtsextraktion + 09-Skizze B |
| A26 | Audio wird aus der **Quelldatei** extrahiert, nicht aus dem normalisierten Zwischenprodukt — sonst MP4 → AAC → WAV als zweite Lossy-Stufe vor Wav2Vec2 | 01 | 04 | ✓ | |
| A27 | **Ausrichtungsgrenze**: die Chunkschleife bricht bei `chunk_idx ≥ n_audio_chunks` ab; kein Chunk bekommt ein aufgefülltes Audiofenster, Bild und Ton stammen immer aus demselben Zeitraum | 01 | 04 | ○ | Das Zuordnungsintervall steht in 04; die Abbruchbedingung am Clipende fehlt · nachgetragen 2026-08-04 in die 04-Skizze (gemeinsam mit A5) |
| A28 | `num_faces=1` — genau **ein** Gesicht je Frame; bei mehreren Personen wird nur das erstplatzierte verarbeitet | 01 | 04, 07 | ○ | Limitation (Talking-Head-Zuschnitt) · Einschränkung auf genau ein Gesicht je Frame nirgends benannt · nachgetragen 2026-08-04 in die 04-Skizze (Methodik) und die 07-Skizze (Limitation 19) |
| A29 | `_expand_to_square` **verschiebt das Quadrat nach innen statt zu klemmen** — Klemmen führte die Seitenverhältnisverzerrung wieder ein, die die Quadratisierung gerade vermeidet | 01 | 04 | ✓ | |
| A30 | `probe_video` liest `avg_frame_rate` statt `r_frame_rate` (letzteres ist die Codec-Zeitbasis und liefert bei VFR-Quellen `90000/1`); gebrochene Bildraten werden als Bruch geparst | 01 | 04 | ○ | nachgetragen 2026-08-04 in die 04-Skizze |
| A31 | **Schema-Schutz des `H5Writer`**: bestehende CSV-Kopfzeile wird gegen `_CSV_FIELDNAMES` geprüft (sonst still inkonsistente Altdateien); Mischen von Audio-mit/ohne bzw. Landmarks-mit/ohne in **einer** Datei löst `ValueError` aus | 01 | 04, 09 | ○ | nachgetragen 2026-08-04 in die 04-Skizze (§Datenspeicherung) |
| A32 | **Der Split-Leak-Vorfall und seine Korrektur**: der Vorgänger (Mischen + `df.head(max_videos)`) partitionierte bei jedem inkrementellen Lauf neu und leakte Identitäten über alle drei Splits — ein realer, dokumentierter Vorfall; der Hash-Split ist die Korrektur dazu | 01 | 04, 07 | ✓ | Gehört als Vorfall in den Beleg, nicht nur als Entwurfsentscheidung · 04 §Split nennt den Vorfall samt −0,12 AUC; 07 greift ihn als methodische Stärke auf |
| A33 | Preis des Hash-Splits: bei wenigen Identitäten sind die Verhältnisse nur ungefähr getroffen und ein Split kann leer bleiben — der Lauf protokolliert die Splitgrößen und warnt mit Seed-Hinweis | 01 | 04 | ✓ | |
| A34 | Ablation über **Hardlinks statt Symlinks** (Symlinks brauchen unter Windows erhöhte Rechte); Pfadstruktur bleibt erhalten, damit die JSON-Metadatenschlüssel gültig bleiben — kein zusätzlicher Rohdatenspeicher | 01 / 10 | 04 | ○ | 05-Skizze §Ablationen nennt Hardlinks |
| A35 | Die beiden Ablationsarme haben **verschiedene Brauchbarkeitskriterien** (Arm A: *eine* Variante mit allen vier Typen; Arm B: alle vier Typen *irgendwo* im Szenario) — sie können über unterschiedlich viele Szenarien laufen | 01 | 04, 06 | ○ | Manifest-Zählwerte in den Ergebnisvergleich aufnehmen · Die ungleichen Brauchbarkeitskriterien der Arme fehlen — sie relativieren den Armvergleich · nachgetragen 2026-08-04 in die 04-Skizze (gemeinsam mit A36, ein Absatz), Zählwerte über A21 in die 06-Skizze |
| A36 | Arm B ist bei Szenarien mit < 4 Varianten **unvollständig entkoppelt** (Wiederverwendung) — genau deshalb misst `ablation_stats.py` die erreichte Dosis, statt sie anzunehmen | 01 | 04, 06 | ○ | nachgetragen 2026-08-04 in die 04-Skizze (gemeinsam mit A35, ein Absatz) |
| A37 | **Abweichende Labelsemantik externer Datensätze**: ohne Segmentannotation setzt `preprocess_loose_videos.py` ein Sentinel-Fake-Segment `[0, 10⁶]`, sodass **jeder** Chunk das Konfigurationslabel erbt — anders als die segmentgenauen AV-Deepfake1M-Labels | 01 | 04, 06 | ○ | Fake-Anteile beider Datensätze nicht direkt vergleichen · Betrifft jede künftige SWAN-Zahl: dort ist **jeder** Chunk fake-gelabelt · nachgetragen 2026-08-04 in die 04-Skizze (§Min-Overlap) und die 07-Skizze (Limitation 14); Kap. 06 erst, wenn SWAN-Zahlen vorliegen |
| A38 | **Gültigkeitsgrenze der Cross-Dataset-Zahlen**: `max_videos: 400` von 5760 SWAN-DF-Clips ≈ **7 % des Datensatzes** (Speichergrenze, ~11 GB statt ~150 GB) | 10 | 05, 06, 07 | ○ | Limitation · Falls SWAN doch ausgewertet wird, ist die 7-%-Grenze zwingend zu nennen · nachgetragen 2026-08-04 in die 05-Skizze (§Datensatz, mit B7) und die 07-Skizze (Limitation 14); gegen `conf/datasets/swan.yaml` verifiziert |
| A39 | `running_mode: image` (Erkennung je Einzelbild) ist der Stand des Datensatzes; `video` (MediaPipe-Tracking) ergäbe andere Crops und ist nur mit **vollständiger Neugenerierung** umschaltbar | 01 / 10 | 04 | ○ | nachgetragen 2026-08-04 in die 04-Skizze (§Gesichtsextraktion) |
| A40 | Registry- und Demodaten sind generiert, nicht handgepflegt: `build_clips_json.py` → `conf/clips.json` (45 Einträge, vierstufige Hierarchie), `build_demo_subset.py` → identitätsdiverser Demo-Teilsatz | 01 / 10 | 04 | ○ | nachgetragen 2026-08-04 in die 04-Skizze (neuer §Demonstrator) |
| A41 | **Die Augenregionen sind gespiegelt gemappt:** \enquote{Left Eye} ist das Auge links *im Bild*, nicht das anatomisch linke Auge des Subjekts. MediaPipes eigene Benennung ist subjektanatomisch; die Umbenennung sorgt dafür, dass Beschriftung und Betrachtersicht übereinstimmen | 01 | 04, 06 | ✓ | **Neu angelegt 2026-08-06, am selben Tag erledigt:** steht im Fließtext `04Methodology.tex:267` (Korrektur des Widerspruchs F18). Ursprüngliche Notiz: (in [01 §face\_extractor.py](01_datenpipeline.md), `_REGION_LANDMARKS`, beschrieben; hatte keine Matrixzeile) · Betrifft jede Bildunterschrift, die eine Augenregion benennt — ohne den Satz liest eine Prüferin die Seitenangabe anatomisch. Umfang: 1 Satz |
| A42 | `FACE_OVAL_INDICES` (36 Silhouettenpunkte) ist die **Maske** der Pixelpartition: alles außerhalb des Ovals gehört zu keiner Region. Das ist der Mechanismus hinter der Aussage \enquote{Hintergrund ist keine aggregierte Region} | 01 | 04 | ✓ | **Neu angelegt 2026-08-06, am selben Tag erledigt:** steht im Fließtext `04Methodology.tex:267` (Korrektur des Widerspruchs F18). Ursprüngliche Notiz: · Die 04-Skizze hält bereits fest, dass Hintergrund keine Region ist (04Methodology.tex:366-371); *warum* — die Ovalmaske — steht nirgends. Gehört in denselben Absatz wie die Korrektur des `!`-Widerspruchs F18. Umfang: Halbsatz |
| A43 | Die Saatpunkttabelle der Regionspartition **dedupliziert nach First-Wins**: ein Landmark kann in mehreren Gruppen vorkommen (Index 8 liegt in `Forehead` und `Nose`), die Partition braucht je Saatpunkt aber genau eine Region — es gewinnt die in `REGION_NAMES` zuerst genannte | 01 | — | – | **Neu angelegt 2026-08-06**, zugleich **bewusst weggelassen**: Implementierungsdetail der Partition, das keine berichtete Zahl und keine Forschungsfrage berührt. Die Nichtüberlappungsgarantie selbst steht bereits in F44 |
| A44 | **Guards gegen Überschreiben der Primärdaten:** `_RESERVED_OUTPUTS` in `preprocess_loose_videos.py` und `_resolve_outputs` in `build_demo_subset.py` verhindern, dass ein externer oder Demo-Lauf die Primärsplits überschreibt; beides ist getestet | 01 / 09 | 09 | ○ | **Neu angelegt 2026-08-06** · gehört in denselben Anhangsabsatz wie die übrigen Silent-Failure-Guards (A31, Q11); Kandidat für `–`, falls der Anhang gekürzt wird. Umfang: Halbsatz |
| A45 | `normalize_video()` (Re-Encode ohne Tonspur) wird von der Pipeline **nicht aufgerufen** — nur von Tests. Ursprünglich als ISTVT-Eingang vorgesehen | 01 | — | – | **Neu angelegt 2026-08-06**, zugleich **bewusst weggelassen**: Doppelung mit C8 (ISTVT nicht implementiert), die den Sachverhalt bereits für den Ausblick trägt. Eine zweite Erwähnung auf Funktionsebene brächte keine eigene Aussage |
| A47 | **Die verworfene Hintergrund-Ablation.** `crop_scale: 1.4` (`conf/preprocess.yaml:31`) sollte ursprünglich Hals- und Schulterpartie im Bild halten, damit der Bildhintergrund als eigene Größe untersucht werden kann. Die Studie wurde verworfen; die Regionspartition ist heute auf das Gesichtsoval maskiert und berechnet keinen Hintergrundwert | 01 / 10 | 04 | ○ | **Neu angelegt 2026-08-06** · Erklärt, warum `crop_scale: 1.4` im Code steht, obwohl der Beleg keinen Hintergrund auswertet — ohne den Satz wirkt der Faktor unmotiviert. **Begründung vom Autor bestätigt (2026-08-06):** Die Datengrundlage ist bereits mit Faktor 1,0 als Bounding Box normalisiert; 1,4 und 1,0 liefern deshalb dasselbe Ergebnis, weil `_scale_bbox` an die Bildgrenzen klemmt. Damit war die Hintergrund-Ablation gegenstandslos und wurde verworfen. **Evidenzstufe:** Autorenaussage, nicht aus dem Code ableitbar — im Beleg entsprechend als Entwurfsgeschichte formulieren, nicht als Messung. Umfang: Halbsatz bis 1 Satz |
| A46 | `backfill_normalized.py` füllt `data/normalized/{video_id}.mp4` für bereits verarbeitete Videos nach (Stream-Copy bei passender Bildrate, sonst Re-Encode) — nötig, weil Frontend und Sweeps aus diesem Bestand lesen | 01 | — | – | **Neu angelegt 2026-08-06**, zugleich **bewusst weggelassen**: reine Nachpflege eines Artefakts, dessen Rolle bereits über A2 (Kopplung Bild/Ton) und A40 (Registry aus `data/normalized/`) im Beleg steht |

## B — Datenladung, Augmentierung, Sampling

| # | Mechanismus | Reg. | Kap. | Status | Notiz |
|---|---|---|---|---|---|
| B1 | ImageNet-z-Score (Video) / Zero-Mean-Unit-Var je Sample (Audio) — **eine** Implementierung für Training und API | 01 | 04 | ○ | Normalisierung beschrieben; die Train/Serve-Identität nur in der 04-Skizze (Punkt 3) · als eigener Punkt 2026-08-04 in die 04-Skizze nachgetragen |
| B2 | Zwei Augmentierungsstufen: `standard` und `robust` | 01 | 04 | ○ | 05-Skizze §Hyperparameter |
| B3 | `robust` = Standard + JPEG-Roundtrip + Gaußblur (DFDC-Gewinner-Rezept) | 01 | 04 | ○ | 05-Skizze nennt JPEG/Blur/Downscale-Upscale samt Wertebereichen |
| B4 | Audio-`robust` = Standard + Zeitmaskierung (SpecAugment-artig auf der Wellenform) | 01 | 04 | ○ | 05-Skizze nennt Audio-Time-Masking |
| B5 | **Balanced Sampling** via `WeightedRandomSampler` (Alternative zur Verlustgewichtung) | 01 | 04 | ○ | 05-Skizze §Klassenungleichgewicht |
| B6 | Klassengewichte `auto` — inverse Frequenz zur Fit-Zeit aus dem Trainsplit | 02 | 04 | ○ | 04 nennt gewichtete Cross-Entropy; `auto` (Fit-Zeit-Berechnung) nur in der 05-Skizze · nachgetragen 2026-08-04 in die 04-Skizze (§Verlustfunktion) |
| B7 | Stage-bewusstes `setup()` (Evaluation ohne vollständigen Datenbestand möglich) | 01 | 05 | ○ | nachgetragen 2026-08-04 in die 05-Skizze (gemeinsam mit A38, Cross-Dataset-Punkt) |
| B8 | Lazy HDF5-Handle je DataLoader-Worker | 01 | 05 | ○ | nachgetragen 2026-08-04 in die 05-Skizze (§Trainingskonfiguration, DataLoader) |
| B9 | **Frame-Perturbation** `tubelet_shuffle` / `frame_shuffle` als Eval-Diagnostik | 01 | 04, 06 | ○ | 05- und 06-Skizze; **Achtung:** die Konfiguration heißt `frame_shuffle`, setzt aber `tubelet_shuffle` |
| B10 | **Spatial-Dominance-Test** — *Hypothese*: AUROC unverändert ⇒ Modell ignoriert die chunkinterne Zeitordnung. **Gemessen wurde das Gegenteil**: 0,745 → 0,597 (tubelet-erhaltend) bzw. 0,691 (voll), die Video-Probe nutzt die Bildreihenfolge also sehr wohl | 01 / 10 / 12 | 06, 07 | ○ | Lief nur auf dem **eingefrorenen** Phase-1-Checkpoint, nicht auf Phase 2 (0,999) · 06-Skizze nennt die Zahlen **und** den widerlegten Hypothesenausgang |

| B11 | **Eine Ziehung je Chunk, nicht je Frame**: alle Zufallsparameter der Videoaugmentierung gelten für **alle 16 Frames identisch** — je Frame neu gezogen entstünde ein künstliches, labelunkorreliertes Flackern genau in der Zeitdimension, die der Transformer auswertet | 01 | 04 | ○ | Ohne diesen Punkt wirkt die Augmentierung wie framweise gezogen · nachgetragen 2026-08-04 in die 04-Skizze (neuer §Augmentierung, gemeinsam mit B13) |
| B12 | Augmentierung ist **trainingsexklusiv** (`augment and split == "train"`); die Frame-Perturbation ist bewusst **nicht** so abgeriegelt, weil sie als Diagnostik den Testsplit erreichen muss | 01 | 04, 06 | ○ | „Augmentierung nur im Train-Split“ steht in der 05-Skizze; die bewusste Nicht-Abriegelung der Perturbation fehlt |
| B13 | Gegenläufige Absicht der beiden Stufen: `standard` soll Identitäts-/Aufnahme-Shortcuts brechen, **ohne** die Fälschungsartefakte zu beschädigen; `robust` greift sie **absichtlich** an, damit sich das Modell nicht auf fragile Hochfrequenzspuren stützt | 01 | 04 | ○ | nachgetragen 2026-08-04 in die 04-Skizze (neuer §Augmentierung, gemeinsam mit B11) |
| B14 | `drop_last=True` **nur** im Trainloader — ein angebrochener Schlussbatch verfälschte bei `accumulate_grad_batches` die effektive Batchgröße eines Gradientenschritts | 01 | 05 | ○ | 05-Skizze begründet `drop_last` bereits |
| B15 | `_load_eval_metadata` **degradiert kontrolliert**: fehlende oder zeilenzahl-inkonsistente CSV ⇒ Warnung und Rückfall auf Chunk-Metriken, statt still falsch zu aggregieren | 01 | 05 | ○ | nachgetragen 2026-08-04 in die 05-Skizze (§Evaluationsmetriken); verwandt mit E12 (dort noch ✗) |
| B16 | Die Wahl der Labelspalte ist in **beide** Richtungen begründet: `label_audio` (kein Fake-Label ohne Tonevidenz) **und** `label_video` (der kombinierte Anteil ist aus dem Bild teils prinzipiell nicht lernbar, das Training kollabierte auf die Mehrheitsklasse) | 01 / 10 | 04 | ✓ | 04 §Labels begründet beide Richtungen ausführlich |
| B17 | **Die konkreten Augmentierungsparameter:** Standard Video — Horizontalspiegelung p = 0,5, Helligkeits-/Kontrast-/Sättigungsjitter mit Faktoren 0,8--1,2, Random-Resized-Crop mit Seitenskala 0,9--1,0. Robust Video — zusätzlich je mit p = 0,3 JPEG-Qualität 30--90, Gauß-σ 0,5--2,0, Downscale-Upscale 0,5--0,9. Standard Audio — Polaritätsumkehr p = 0,5, additives Gaußrauschen bei SNR 15--40 dB (p = 0,5). Robust Audio — zusätzlich Zeitmaskierung von 5--10 % der Chunklänge (p = 0,5) | 01 / 10 | 05 | ○ | **Neu angelegt 2026-08-06** · Die 05-Skizze führt nur JPEG 30--90, Blur 0,5--2, Downscale 0,5--0,9 und \enquote{Audio-Time-Masking} — **die Ziehungswahrscheinlichkeiten, die Jitter- und Cropbereiche, der SNR-Bereich und die Maskenlänge fehlen alle**. Ohne sie ist der `robust`-Arm nicht reproduzierbar. Verifiziert gegen [10 §2](10_konfiguration.md) (Zeitmaskierung 5--10 %, p = 0,5). Umfang: zwei Tabellenzeilen |
| B18 | **Warum gerade Polaritätsumkehr und additives Rauschen:** eine reine Pegeländerung wäre durch die nachfolgende Zero-Mean-Unit-Variance-Standardisierung wegnormiert und wird deshalb gar nicht erst verwendet; die Polaritätsumkehr ist für die Aufgabe phaseninvariant und nimmt dem Modell die absolute Wellenformpolarität als Merkmal | 01 | 04 | ○ | **Neu angelegt 2026-08-06** · Gegenstück zu B13 (gegenläufige Absicht der beiden Stufen), aber für die Audioseite: der Beleg nennt die Augmentierungen, nicht das Auswahlkriterium. Umfang: 1 Satz |
| B19 | Die Frame-Perturbation greift **nach** der Normierung — zulässig, weil Mischen und framweise Normierung kommutieren; zusätzlich muss die Framezahl durch `tubelet_size` teilbar sein | 01 | 04, 05 | ○ | **Neu angelegt 2026-08-06** · Ergänzung zu B9/B12. Ohne das Kommutationsargument ist die Diagnostik angreifbar (\enquote{gemessen wurde eine anders normierte Eingabe}), und die Teilbarkeitsbedingung ist die Voraussetzung des tubelet-erhaltenden Arms. Umfang: Halbsatz |
| B20 | **Zwei Konsistenzprüfungen gegen stille Fehlausrichtung:** `MultimodalHDF5Dataset` prüft, dass `video` und `audio` gleich lang sind (eine Längendifferenz verschöbe die Modalitäten stillschweigend gegeneinander); `_train_labels()` liest die Labelspalte aus derselben Quelle, die auch das Dataset nutzt (sonst kämen Klassengewichte aus `label` und Training aus `label_audio`) | 01 | 04, 05 | ○ | **Neu angelegt 2026-08-06** · Beides sind Silent-Failure-Guards derselben Klasse wie A31 und B15 und stützen die Aussage, dass Audio- und Videofenster eines Chunks garantiert zusammengehören. Umfang: 1 Satz |

| B21 | **`vision_constants.py` ist die einzige Quelle der ImageNet-Statistik.** `IMAGENET_MEAN` und `IMAGENET_STD` stehen an genau einer Stelle; Training (`base_hdf5_dataset.py`), API-Inferenz (`api/inference.py`) und die drei Erklärskripte ziehen dieselben Werte — eine Suche nach den Zahlwerten findet im Projekt keine zweite Stelle. `inverse_normalize_frame` ist die Gegenoperation für jede Bildausgabe | 03 | 04 | ○ | **Neu angelegt 2026-08-06** · Das ist der **Mechanismus hinter B1**: Die Train/Serve-Parität ist nicht nur getestet, sondern konstruktiv erzwungen, weil es die Konstanten nur einmal gibt. B1 behauptet die Parität, B21 erklärt, warum sie nicht auseinanderlaufen kann. `inverse_normalize_frame` ist zugleich die Funktion, die S36 (`_encode_crop_video`) für die Vergleichsspieler nutzt. Umfang: Halbsatz an B1 |

## C — Modellarchitekturen

| # | Mechanismus | Reg. | Kap. | Status | Notiz |
|---|---|---|---|---|---|
| C1 | VideoMAE-base + Klassifikationskopf, `use_mean_pooling=True` | 02 | 02, 04 | ○ | Architektur ✓; `use_mean_pooling` (statt CLS-Token) für den Kopf nicht benannt · nachgetragen 2026-08-04 in die 04-Skizze (nur Kap. 4: Kapitel 2 trägt keine eigenen Konfigurationswerte) · **Begründung nachgeliefert 2026-08-06: siehe C16** — es gibt kein CLS-Token, Mittelung ist daher keine Wahl, sondern die einzige Möglichkeit. **Nicht ohne C16 ausschreiben**, sonst wirkt `use_mean_pooling` wie eine Geschmacksentscheidung |
| C2 | Wav2Vec2-base + Projektor/Kopf | 02 | 02, 04 | ✓ | |
| C3 | **Bidirektionale Cross-Attention-Fusion**, parallel und pre-norm | 02 | 04 | ✓ | |
| C4 | Beide Blöcke nutzen die **ursprünglichen** Projektionen als K/V — Begründung: saubere xAI-Interpretierbarkeit | 02 | 04 | ✓ | 04 hebt die unveränderten K/V und die xAI-Begründung explizit hervor |
| C5 | `fusion_dim: 512`, 8 Köpfe, Dropout 0,1, Mean-Pool → Konkatenation → 2-Schicht-MLP | 02 | 04 | ○ | 512/8 Köpfe/Mean-Pool/MLP ✓; **Dropout-Widerspruch aufgelöst:** `configs/model/multimodal.yaml:9` setzt 0,3 (Begründung dort: Overfitting des Fusionskopfes), der Modul-Default 0,1 wird von keiner Konfiguration gefahren — es gilt 0,3 · nachgetragen 2026-08-04 in die 04-Skizze (Ansatzpunkt `proj_dropout`, wirkt auf **alle** Modi; Wert bleibt in Kap. 5) |
| C6 | Vier `fusion_mode`s: `cross_attention`, `concat`, `video_only`, `audio_only` | 02 | 04, 06 | ✓ | 04 nennt alle vier Modi und den Verzicht auf `*_only` aus Zeitgründen |
| C7 | Nachweis, dass `video_only` das Audio *tatsächlich* ignoriert (Test) | 02 / 09 | 06 | ○ | Der Gültigkeitsnachweis der Ablation fehlt · nachgetragen 2026-08-04 in die 04-Skizze (gemeinsam mit C15, ein Absatz) und die 09-Skizze §G (Testzeile); **nicht** in Kap. 6 — die `*_only`-Modi wurden nicht trainiert, dort gibt es kein Ergebnis |
| C8 | **ISTVT ist NICHT implementiert** (`configs/model/istvt.yaml` ist leer) | 10 | 07, 08 | ○ | Als Ausblick führen · Zusätzlich: `vault/Archive/istvt-2023.md` trägt „do not cite“ · nachgetragen 2026-08-04 in die 08-Skizze (§Ausblick) samt Zitiersperre; für den Kap.-7-Anteil Vorschlag `–` (Doppelung mit Limitation 5, domänenfremd vortrainierte Backbones) |
| C9 | Gemeinsame Basisklasse `BaseDeepfakeModule` — Vergleichbarkeit per Konstruktion | 02 | 04 | ✓ | |
| C10 | **Parameterzählungs-Fallstrick:** beide Attention-Blöcke werden **unbedingt** gebaut (2.101.248 Parameter), im Forward aber nur in `cross_attention` ausgeführt — `model/params/trainable` überschätzt `concat` und die `*_only`-Modi um genau diesen Betrag; der `concat`-Kopf trainiert real ~1,32 M statt 3,42 M | 02 | 04, 05 | ○ | In der Parametertabelle die kleinere Zahl angeben · 05-Skizze §Ablationen und 07-Limitation 9 nennen den Caveat bereits |
| C11 | Der Verlust wird **im Modul** berechnet, nicht über die interne CE von HuggingFace — nur so greifen die `class_weights`, die bei ~7 % Fake-Anteil nötig sind | 02 | 04 | ○ | nachgetragen 2026-08-04 in die 04-Skizze (§Verlustfunktion, gemeinsam mit B6) · die Fake-Anteil-Zahl dieser Zeile ist Gegenstand des Widerspruchs A11 und beim Ausschreiben wegzulassen |
| C12 | `wav2vec2_module.py` ist als **einziges** der drei Modelle durchgängig `@beartype` + jaxtyping annotiert — **Laufzeitprüfung** der Tensorformen auf `__init__`, `forward`, `model_step`, allen Steps und `explain` | 02 / 11 | 04, 09 | ○ | nachgetragen 2026-08-04 in die 09-Skizze §G; **nicht** in Kap. 4 — Implementierungshygiene, keine Designentscheidung. Kandidat für `–`, falls der Anhang gekürzt wird |
| C13 | **Empirischer Befund:** kaltes vollständiges Finetuning des Wav2Vec2-Encoders **konvergiert nicht** (Verlust bleibt bei ln 2, AUC auf Zufallsniveau). Frozen-Backbone ist damit Konvergenzvoraussetzung, kein Rechenzeitkompromiss | 02 / 10 | 05, 06 | ○ | Der Konvergenzbefund begründet die Frozen-Baseline — er fehlt als Begründung · nachgetragen 2026-08-04 in die 06-Skizze (§Diagnostik, mit den Diagnosewerten aus `docs/model.md` §7.2) und als Begründungshalbsatz in die 05-Skizze (§Hyperparameter) |
| C14 | Der multimodale Pfad friert den Wav2Vec2-CNN **ohne Abschaltmöglichkeit** und über die private API `_freeze_parameters()` ein — zwei Unterschiede zum unimodalen Audiomodul, die beim Laufvergleich zu beachten sind | 02 | 04 | ○ | nachgetragen 2026-08-04 in die 04-Skizze (§Trainingsstrategie, Phase-2-Punkt) |
| C15 | In den `*_only`-Modi wird der Backbone der verworfenen Modalität **gar nicht erst ausgeführt** (`_extract_features` → `None`) — die Ablation misst den Beitrag des Signals, nicht den einer kleineren Architektur | 02 / 10 | 04, 06 | ○ | 04 sagt „durch Nullvektoren substituiert“; der Code überspringt den Backbone ganz (verifiziert: `multimodal_module.py:384-391`) · nachgetragen 2026-08-04 in die 04-Skizze (§Ablationsstudien, gemeinsam mit C7) — beim Ausschreiben ist der bestehende Satz zu präzisieren, nicht zu ergänzen |

| C19 | Sieben Kleinmechanismen der Modellschicht: Hyperparameterzahl je Modell (VideoMAE 18, Wav2Vec2 18, Multimodal 25); `configure_optimizers` erkennt **per Signaturinspektion**, ob der Scheduler `num_training_steps` annimmt, und plant dann schrittweise statt epochenweise; für `ReduceLROnPlateau` wird der Monitor auf `val/auc_video` gesetzt (dieselbe Größe wie Checkpointing und Early Stopping); `adv_steps < 1` wird bei aktivem `adv_train` als Fehler abgewiesen; Gradient Checkpointing wird bei Wav2Vec2 bewusst **vor** `_wrap_lora` aktiviert, damit die LoRA-Sonde es bei Bedarf wieder abschalten kann (D28); `_MODIFY_CATEGORIES` bezieht seine Indizes aus `MODIFY_TYPE_TO_IDX` der Datenschicht; die Querschnittstabelle in [02](02_modelle.md) listet elf Mechanismen, die **einmal** in `base_module.py` liegen und im Beleg nicht als modellspezifisch dargestellt werden dürfen | 02 | — | – | **Neu angelegt 2026-08-06**, zugleich **bewusst weggelassen**: Verdrahtungs- und Hygienedetails ohne Bezug zu einer Forschungsfrage. **Ausnahme mit Belegwert:** der letzte Punkt ist ein Schreibhinweis, kein Mechanismus — er stützt C9 (gemeinsame Basisklasse = Vergleichbarkeit per Konstruktion) und ist beim Ausschreiben von Kapitel 4 mitzuführen |
| C16 | **VideoMAE liefert kein CLS-Token.** `last_hidden_state` ist die volle Patch-Token-Sequenz der Länge **1568** = 8 zeitliche × 14 × 14 Patches (bei 16 Frames à 224×224). Das ist zugleich die Eingabelänge der Cross-Attention und die Gitterbasis der Heatmap-Nachverarbeitung | 02 | 04 | ○ | **Neu angelegt 2026-08-06** · **Wirkungskette B9 → C16 → C1 → F9, in dieser Richtung zu lesen und im Beleg einmal zusammenhängend zu schreiben:** (1) VideoMAE-base bettet mit einem Tubelet der Tiefe 2 und Patches von 16×16 Pixeln ein — 16 Frames à 224×224 ergeben daher **8 × 14 × 14 = 1568** Tokens (14 = 224/16); die Tubelet-Tiefe 2 ist dieselbe, auf der die Diagnostik `tubelet_shuffle` aus **B9** operiert. (2) Weil es **kein CLS-Token** gibt, ist Mittelung über die Patch-Tokens keine Variante, sondern die einzige Möglichkeit — das ist die Begründung von **C1**. (3) Dieselbe 14×14-Gitterweite bei 16 Pixeln Schrittweite ist der Grund, warum die Relevanznachverarbeitung in **F9** genau auf 16×16 poolt: Das Pooling bildet das Tokengitter ab, das Upsampling macht daraus wieder Pixel. Ohne (1) wirkt die 16 in F9 wie eine frei gewählte Glättungsbreite. **Offen und vor dem Ausschreiben zu prüfen:** ob die *zeitliche* Auflösung der Videorelevanz 16 oder 8 Positionen je Chunk trägt. Die Karte hat die Form `(B, 16, H, W)`, weil Input×Gradient am Eingabetensor hängt; der Informationsgehalt ist aber durch 8 Tokenpositionen begrenzt. Anders als beim Audio (**F30**, dort ist die Zählung 31 Werte auf 64 Bins ausdrücklich belegt) ist das hier **nicht** nachgerechnet — bis dahin keine Aussage über frameweise zeitliche Lokalisierung im Video treffen. Umfang: 1 Absatz (die Kette), 1 Prüfauftrag |
| C17 | **Für die multimodale Erklärung wird auch der Fusionskopf gepatcht**, nicht nur die beiden Backbones: `monkey_patch(self.fusion, build_common_patch_map())` deckt dessen LayerNorm, GELU und Dropout ab — ohne diesen zweiten Teil liefe die Relevanz durch den Fusionsgraphen nicht korrekt. Zusätzlich hält **jedes** der drei Modelle eine eigene modulweite Wächtervariable (`_VIDEOMAE_LRP_PATCHED`, `_WAV2VEC2_LRP_PATCHED`, `_MULTIMODAL_LRP_PATCHED`) | 02 / 04 | 04 | ○ | **Neu angelegt 2026-08-06** · F10 beschreibt den gemeinsamen Rückwärtspass, F29 die Idempotenz **in `attnlrp.py`**. Dies ist die zweite Hälfte: der Fusionskopf ist kein HuggingFace-Modul und fiele sonst durch das Patching hindurch. Die drei Wächter sind eine **zweite** Idempotenzschicht auf Modellebene. Umfang: Halbsatz an F10 |
| C18 | **Normierungsasymmetrie in der multimodalen Erklärung:** `normalize_video` steuert **nur** die Videoseite — die Audiorelevanz wird im Einzelziel-Pfad *immer* normiert. Im `per_class`-Pfad wird `normalize_video` bewusst ignoriert und beide Seiten kommen roh zurück, damit der Aufrufer Magnitude und Richtung clipglobal bildet | 02 | 04 | ○ | **Neu angelegt 2026-08-06** · F10 begründet den gemeinsamen Rückwärtspass damit, dass Video- und Audiorelevanz \enquote{auf gleicher Skala} liegen. Im Einzelziel-Pfad stimmt das nur eingeschränkt, weil die Audioseite zusätzlich normiert wird. Beim bivariaten Pfad — dem im Demonstrator gefahrenen — trifft die Aussage zu. Umfang: Halbsatz |

## D — Training

| # | Mechanismus | Reg. | Kap. | Status | Notiz |
|---|---|---|---|---|---|
| D1 | **Phase 1** = eingefrorener Backbone, nur Kopf | 02 / 10 | 04, 05 | ✓ | |
| D2 | **Phase 2** = End-to-End, `lr=1e-5` (LoRA `1e-4`); **effektive Batchgröße überall 6**, nur unterschiedlich aufgeteilt: Video 6×1 (unter SDPA), Audio 32×1, Multimodal 1×6. Die alte Aufteilung 2×3 steht heute **nur noch** in `train_video_adversarial.yaml` | 10 | 05 | ! | **05-Skizze nennt für Phase 2 Video weiterhin 2×3** — das ist der alte Eager-Wert; aktuell 6×1 bei gleicher effektiver Batchgröße |
| D3 | Eingefrorener Backbone bleibt im `eval`-Modus (`train()` überschrieben) | 02 | 04 | ✓ | |
| D4 | Wav2Vec2-CNN-Feature-Extractor bleibt **auch in Phase 2** gefroren | 02 | 04 | ✓ | |
| D5 | **Warm-Start vs. Resume** — `warmstart_ckpt` lädt nur Gewichte, frischer Optimierer/LR | 03 | 05 | ○ | Warm-Start ✓; die Abgrenzung zu `ckpt_path` (Resume mit altem Optimierer) fehlte · nachgetragen 2026-08-04 in die 05-Skizze (§Trainingskonfiguration, gemeinsam mit D22) |
| D6 | Gradient Checkpointing | 02 | 05 | ○ | 05-Skizze §Hardware |
| D7 | `linear_warmup_cosine`, schrittbasiert, `warmup_ratio: 0.05`, `horizon_epochs: 15` entkoppelt von `max_epochs` | 03 / 10 | 05 | ○ | 05-Skizze inkl. Begründung von `horizon_epochs` |
| D8 | Begründung: `ReduceLROnPlateau` war bei 10 Epochen / `patience 3` wirkungslos | 10 | 05 | ○ | Die Vorgeschichte (ReduceLROnPlateau wirkungslos) fehlte als Begründung des Schedulerwechsels · nachgetragen 2026-08-04 in die 05-Skizze (§Trainingskonfiguration, gemeinsam mit D26) |
| D9 | **Layer-wise LR Decay** (`llrd_decay`) für Phase 2 | 02 | 04, 05 | ○ | 05-Skizze nennt LLRD 0,75 |
| D10 | **LoRA** auf Attention-Q/V; Optimizer-States ~94 M → < 1 M | 02 | 04, 05 | ○ | 05-Skizze führt LoRA als Phase-2-Alternative |
| D11 | LoRA-Guards: verlangt entfrorenen Backbone, unverträglich mit LLRD | 02 | 05 | ○ | nachgetragen 2026-08-04 in die 05-Skizze (neuer Unterpunkt „LoRA als Phase-2-Alternative“, gemeinsam mit D12, D13, D28) |
| D12 | **LoRA-Merge beim Export** — der Checkpoint ist wieder ein gewöhnliches Modell | 03 | 05 | ○ | Ohne den Merge wirkt LoRA wie ein abweichendes Modellformat · nachgetragen 2026-08-04 in die 05-Skizze (LoRA-Absatz) |
| D13 | Warm-Start-Schlüsselübersetzung für LoRA-Module (sonst würden Backbone-Gewichte still übersprungen) | 02 / 03 | 05 | ○ | nachgetragen 2026-08-04 in die 05-Skizze (LoRA-Absatz) |
| D14 | **Mixup** (Beta(α,α) auf Eingaben und Zielen); bei adversarialem Training übersprungen | 02 | 04 | ○ | 05-Skizze |
| D15 | **Label Smoothing** | 02 | 04 | ○ | 05-Skizze |
| D16 | **SWA** (opt-in); Konflikt mit Early Stopping dokumentiert | 10 | 04 | ○ | ~~SWA als Arm genannt; der Konflikt mit Early Stopping fehlt~~ · **nachgetragen 2026-08-06** (Verifikationslauf) in die 05-Skizze §Ablations- und Diagnostikläufe. **Beim Ausschreiben:** die genaue Formulierung des Konflikts aus `configs/callbacks/swa.yaml` zitieren, nicht paraphrasieren |
| D17 | `gradient_clip_val: 1.0` gegen Gradientenspitzen im bf16-Phase-2-Training | 10 | 05 | ○ | 05-Skizze §Hyperparameter |
| D18 | `max_epochs: 30`, Early Stopping `patience: 5` auf `val/auc_video` | 10 | 05 | ○ | 05-Skizze §Hyperparameter |
| D19 | `seed: 42` fest; **Einzelläufe, keine Multi-Seed-Varianz** | 03 | 05, 07 | ○ | Limitation · 05/06/07 benennen den Einzelseed mehrfach als Limitation |
| D20 | Checkpoint-Export auf stabilen Pfad für API-Wiederverwendung; `ckpt_export_name` ist in **27 der 29** Experimentkonfigurationen gesetzt — ohne eigenen Namen schriebe jeder Ablationsarm auf denselben klassenabgeleiteten Pfad und überschriebe die Baseline | 03 / 10 | 05 | ○ | nachgetragen 2026-08-04 in die 05-Skizze (§Reproduzierbarkeit, gemeinsam mit D34); gegen `configs/experiment/` verifiziert: 27 Dateien setzen den Schlüssel |
| D21 | **Prozesslokale Puffer der videoweisen Aggregation** — korrekt nur bei `devices=1`; ein Mehr-GPU-Lauf bräuchte `all_gather`, sonst aggregierte jeder Rang nur seinen Ausschnitt. Im Code als Kommentar festgehalten | 02 | 05, 07 | ○ | Bekannte Einschränkung, nicht übersehen · nachgetragen 2026-08-04 in die 05-Skizze (§Evaluationsmetriken); für den Kap.-7-Anteil Vorschlag `–`: alle Läufe fahren `devices: 1`, die Bedingung schränkt keine berichtete Zahl ein |
| D22 | `unfreeze_backbone()` ist **kein Laufzeitschalter**: der Optimierer wird je `fit` einmal über die dann trainierbaren Parameter gebaut, ein Auftauen mitten im Lauf erreicht ihn nicht. Der unterstützte Weg ist ein **frischer Lauf** mit `freeze_backbone=false` + `warmstart_ckpt` | 02 / 03 | 05 | ○ | Darf im Beleg nicht fehlen · 04 formuliert den Warm-Start als „idealerweise“ — tatsächlich ist er der einzige unterstützte Weg · nachgetragen 2026-08-04 in die 05-Skizze (gemeinsam mit D5); die Präzisierung des 04-Satzes bleibt Modus C |
| D23 | **Testmetriken stammen nicht garantiert vom besten Checkpoint** — ohne `checkpoint_callback` oder bei leerem `best_model_path` fällt `train.py` still auf die *letzten* Gewichte zurück und loggt nur eine Warnung | 03 | 05, 06 | ○ | Nur mit „Best ckpt path: …" im Lauf-Log belastbar · Betrifft die Belastbarkeit **jeder** berichteten Testzahl · nachgetragen 2026-08-04 in die 05-Skizze (§Modellauswahl und Monitoring, gemeinsam mit E8); der Kap.-6-Anteil bleibt offen, bis die Lauf-Logs geprüft sind |
| D24 | `eval.py` instanziiert **keine Callbacks** und ruft **kein `seed_everything`** — die Reproduzierbarkeit des Perturbationstests hängt am datensatzseitigen `frame_perturbation_seed`, nicht am globalen Seed | 03 | 05 | ○ | nachgetragen 2026-08-04 in die 05-Skizze (§Reproduzierbarkeit) |
| D25 | Parameterzahlen (gesamt / trainierbar / eingefroren) werden je Lauf über `log_hyperparameters` mitgeschrieben — **die Quelle der Modellgrößen im Beleg** | 03 | 05 | ✓ | Die Parameterzahlen stehen in 04 §Trainingsstrategie |
| D26 | `min_lr_ratio` wird von **keiner** Konfiguration gesetzt ⇒ ab Ende des Horizonts Lernrate **exakt 0**, und weil AdamW sein Weight Decay mit `lr` multipliziert, ein vollständiger Stillstand. Praktisch doppelt abgesichert (Early Stopping, SWA-Override) | 03 / 10 | 05 | ○ | nachgetragen 2026-08-04 in die 05-Skizze (§Trainingskonfiguration, gemeinsam mit D8); verifiziert: `min_lr_ratio` kommt in keiner YAML als Schlüssel vor |
| D27 | LoRA spart **keinen** Aktivierungsspeicher — die Gradienten fließen weiter durch alle Schichten zu den Adaptern; gespart werden Optimizer-States und Basisgradienten. Deshalb identische Batchgrößen wie beim Full-Finetuning | 10 | 05 | ○ | Häufiges Missverständnis · Verbreitetes Missverständnis — gehört zur LoRA-Beschreibung · nachgetragen 2026-08-04 in die 05-Skizze (Batchgrößentabelle) |
| D28 | LoRA + Wav2Vec2: PEFT registriert bei aktivem Gradient Checkpointing einen Hook, der `get_input_embeddings()` braucht — Wav2Vec2 hat keine; das Modul **schaltet das Checkpointing mit Warnung ab** | 02 / 10 | 05 | ○ | nachgetragen 2026-08-04 in die 05-Skizze (LoRA-Absatz) |
| D29 | `bf16-mixed` kommt aus `trainer/gpu.yaml`, **nicht** aus den Experimentkonfigurationen — wer „bf16 Mixed Precision" schreibt, zitiert diese Datei | 10 | 05 | ○ | 05-Skizze §Hardware nennt `bf16-mixed` |
| D30 | Die `*_mixup`-Ablationen sind **kein isolierter Mixup-Test**, sondern das vollständige ViT-Rezept: Mixup `Beta(0.2,0.2)` **plus** Label Smoothing 0,1 **plus** Balanced Sampling; Vergleichsmaßstab ist deshalb doppelt (gegen Baseline **und** gegen `*_balanced`) | 10 | 05, 06 | ○ | 05-Skizze sagt „nur in dedizierten Ablations-Bündeln aktiv“ — die Dreifachkopplung ist damit angedeutet, aber nicht benannt |
| D31 | Alle Balanced-Varianten setzen `class_weights: null` — Sampler und Verlustgewichtung korrigieren dieselbe Schieflage und dürfen nicht doppelt greifen | 01 / 10 | 04 | ○ | 05-Skizze warnt explizit vor doppelter Korrektur |
| D32 | **Phase 1 ist der unveränderte Modell-Default**, keine Experiment-Überschreibung: die drei Phase-1-Dateien setzen weder `freeze_backbone` noch `freeze_feature_extractor` | 10 | 05 | ○ | nachgetragen 2026-08-04 in die 05-Skizze (§Trainingskonfiguration, Einleitung der Tabellen) |
| D33 | **Optuna-Suchraum existiert** (`deepfake_optuna.yaml`: `val/auc_video` maximieren, 10 Trials, TPESampler `seed 42`, Suchraum lr / batch_size / weight_decay) | 10 | 05 | ○ | Läufe nicht dokumentiert — durchgeführt? · nachgetragen 2026-08-04 in die 05-Skizze (§Ablations- und Diagnostikläufe); bleibt Kandidat für `–`, falls die Prüfung von W&B/`logs/multiruns/` ergibt, dass kein Trial gelaufen ist |
| D34 | `debug/default.yaml` setzt `export_ckpt: false`, damit Debugläufe die echten trainierten Checkpoints auf den stabilen Pfaden nicht überschreiben | 10 | 05 | ○ | nachgetragen 2026-08-04 in die 05-Skizze (§Reproduzierbarkeit, gemeinsam mit D20) |
| D35 | SWA-Feinheit: Lightning tauscht die Lernrate nur **epochenweise** aus und passt damit nicht zum schrittbasierten `linear_warmup_cosine`; die Gewichtsmittelung funktioniert trotzdem | 10 | 04, 05 | ○ | nachgetragen 2026-08-04 in die 05-Skizze (§Ablations- und Diagnostikläufe); **nicht** in Kap. 4 — Laufzeitbedingung, keine Designbegründung |
| D36 | **Multimodales Mixup mischt beide Modalitäten mit demselben `lam` und derselben Permutation**, damit die A/V-Paarung eines Beispiels erhalten bleibt; auf adversarialen Batches wird Mixup in allen drei Modulen übersprungen | 10 | 05 | ○ | **Neu angelegt 2026-08-05** (in 10 §2 beschrieben, hatte keine Matrixzeile) · nachgetragen in die 05-Skizze (§Klassenungleichgewicht); Kap.-4-Verortung verworfen, weil Kapitel 4 Mixup nirgends einführt · verifiziert: `multimodal_module.py:527` und `:524` |
| D37 | **Motivation der drei Phase-2-Läufe steht je im Kopfkommentar:** Video = Ausbruch aus dem Linear-Probe auf Kinetics-Features (Kommentar nennt ~0,56 AUC), Multimodal = eigentlicher Test der Cross-Modal-Sync-Hypothese, Audio-LoRA = Ersatz für das kollabierte Cold-Full-Finetuning | 10 | 05 | ○ | **Neu angelegt 2026-08-05** · nachgetragen in die 05-Skizze (§Trainingskonfiguration, an den D5+D22-Absatz) · **Die ~0,56 sind eine Konfigurationsnotiz, keine Ergebnisnotiz** — vor Übernahme gegen Kapitel 6 abgleichen, sonst ohne Zahl schreiben (vgl. Q23) · verifiziert gegen `train_video_phase2.yaml:3-5` |
| D38 | Vorgeschichte der Abbruchparameter: `max_epochs: 30` hoch angesetzt, weil ein VideoMAE-Phase-2-Lauf bei Epoche 10 noch underfittete (train acc 0,66); `patience: 5` muss kleiner als `max_epochs` sein — vorher patience 15 bei max_epochs 10, der Callback konnte nie feuern | 10 | 05 | ○ | **Neu angelegt 2026-08-05** · Ergänzung zu D18 (dort stehen nur die Werte) und dieselbe Fehlerklasse wie D8 · nachgetragen in die 05-Skizze (§Trainingskonfiguration, Punkt „Zwei Entscheidungen“) · verifiziert gegen `trainer/default.yaml:6-9` und `callbacks/default.yaml:21-23` |
| D39 | Checkpoint-Namensvorlage `epoch_{epoch:03d}-val_auc_video_{val/auc_video:.3f}` mit `auto_insert_metric_name: False`, `save_last: True` und `save_top_k: 1` — pro Lauf genau **ein** bester Checkpoint plus `last.ckpt`, Val-AUC im Dateinamen | 10 | 05 | ○ | **Neu angelegt 2026-08-05** · Die 05-Skizze zitiert die Checkpoints „mit den val-AUC-Werten der Dateinamen“, ohne die Herkunft zu nennen · nachgetragen in die 05-Skizze (§Reproduzierbarkeit, an D20+D34) · verifiziert gegen `callbacks/default.yaml:10-17`, `model_checkpoint.yaml:10` und die vier Dateien unter `checkpoints/` |
| D40 | `train_audio_smoothing.yaml` ist der **einzige Arm, der eine der drei Regularisierungsmaßnahmen allein führt** (Label Smoothing 0,1 ohne Mixup) — Gegenprobe zu D30. Der Kopfkommentar begründet ihn falsch mit „Wav2Vec2 unterstützt kein Mixup“ | 10 | 05 | ○ | **Neu angelegt 2026-08-05** · gehört in denselben Absatz wie D30 · **Kommentar überholt, Code gilt:** `wav2vec2_module.py:177` ruft `_mixup_training_loss`, `train_audio_mixup.yaml:25` setzt `mixup_alpha: 0.2` · nachgetragen in die 05-Skizze (§Ablations- und Diagnostikläufe) |

| D41 | **Der Lernratenplan komponiert mit dem schichtweisen Decay:** `linear_warmup_cosine` skaliert die Basis-Lernrate **je Parametergruppe** statt eine globale Rate zu setzen. Nur deshalb lassen sich `llrd_decay` (D9) und der schrittbasierte Warmup-Cosine-Plan (D7) gleichzeitig fahren | 03 | 05 | ○ | **Neu angelegt 2026-08-06** · D7 und D9 stehen in der 05-Skizze nebeneinander, ohne dass gesagt wäre, warum sie sich nicht ins Gehege kommen. Ohne diesen Halbsatz wirkt die Kombination wie ein möglicher Konflikt. Umfang: Halbsatz |
| D42 | **Jeder Checkpoint enthält den Lernratenplan als gepickeltes Hydra-Partial.** Am Modulende registriert `add_safe_globals([linear_warmup_cosine])` die Funktion selbst als ladbares Objekt — ohne diese Zeile ließe sich **kein** so trainierter Checkpoint mehr laden | 03 | 05, 09 | ○ | **Neu angelegt 2026-08-06** · Q17 erfasst die `add_safe_globals`-Aufrufe an den sechs checkpointladenden Einstiegspunkten; dies ist eine **andere** Registrierung an anderer Stelle und eine harte Reproduktionsabhängigkeit: Wer den Scheduler umbenennt oder verschiebt, macht alle bestehenden Checkpoints unladbar. Gehört zu D39 (Checkpointnamen) und Q17. Umfang: Halbsatz |
| D43 | **Der LoRA-Export lädt den besten Checkpoint frisch** — das Live-Modul hält am Laufende die Gewichte der *letzten*, nicht der besten Epoche. Getauscht werden nur `state_dict` und der Hyperparameter `peft_mode='none'`; die Loop-States bleiben erhalten | 03 | 05 | ○ | **Neu angelegt 2026-08-06** · Präzisierung zu D12 (LoRA-Merge beim Export). Ohne das Nachladen exportierte der Merge die Gewichte der letzten Epoche und der stabile Pfad trüge ein anderes Modell als die Metriktabelle. Umfang: Halbsatz am D11--D13-Absatz |
| D44 | Fünf Kleinmechanismen des Trainingsgerüsts: `fit` und `test` sind über `cfg.train`/`cfg.test` einzeln abschaltbar (beide `True`); `RankedLogger` präfixt Meldungen mit dem Rang (Mehr-GPU-Vorsorge, hier ohne Wirkung); `src/utils/__init__.py` re-exportiert `audio_xai` und `adversarial` bewusst **nicht** (letzteres wird erst innerhalb der Methoden importiert); der Konfigurationsbaum wird in fester Reihenfolge gedruckt (`data`, `model`, `callbacks`, `logger`, `trainer`, `paths`, `extras`, dann der Rest); `callbacks=swa` empfiehlt `trainer.max_epochs=15` — also genau den Abfallhorizont aus D7 | 03 | — | – | **Neu angelegt 2026-08-06**, zugleich **bewusst weggelassen**: Gerüst- und Hygienedetails ohne Bezug zu einer Forschungsfrage. **Ausnahme mit Belegwert:** die SWA-Empfehlung `max_epochs=15` deckt sich exakt mit `horizon_epochs: 15` und wäre ein Halbsatz wert, falls der D8+D26-Absatz ohnehin geschrieben wird |

| D45 | **LLRD wird stillschweigend übersprungen**, wenn `llrd_decay` fehlt, der Backbone **eingefroren** ist (Phase 1 — dort gäbe es nichts zu staffeln) oder das Modell keine Stacks liefert. Ein `llrd_decay` in einer Phase-1-Konfiguration bliebe also wirkungslos, ohne Meldung. Parameter werden über eine `id()`-Menge dedupliziert, damit ein in zwei Stacks vorkommendes Modul nicht doppelt in den Optimierer gerät — nötig im multimodalen Zwei-Stack-Fall | 02 | 05 | ○ | **Neu angelegt 2026-08-06** · D9 führt LLRD als Phase-2-Mechanismus. Die stille Abschaltung ist dieselbe Fehlerklasse wie Q20 (Callback ohne `_target_`): eine gesetzte Konfiguration ohne Wirkung und ohne Warnung. Die Deduplizierung ist die Bedingung dafür, dass der multimodale LLRD-Lauf überhaupt korrekt ist. Umfang: Halbsatz |
| D46 | **Gradient Checkpointing wirkt nur in Phase 2 und nur im Training.** HuggingFace wendet es bei `self.training == True` an; in Phase 1 laufen die Backbones wegen der `train()`-Überschreibung (D3) im `eval`-Modus, die Einstellung ist dort also **inert**. Der `explain()`-/AttnLRP-Pfad ist ebenfalls unberührt. Gemessene Kosten: **~10 % Schrittzeit** | 02 | 05 | ○ | **Neu angelegt 2026-08-06** · D6 nennt Gradient Checkpointing als Speichermaßnahme, ohne Wirkbereich und ohne Preis. Beides gehört in die Hardwaresektion: die 10 % sind der Gegenwert für die 8-GB-Grenze, und dass die Erklärpfade unberührt bleiben, sichert die Vergleichbarkeit der Heatmaps zwischen den Phasen. Umfang: Halbsatz |
| D47 | **Der PGD-Angriff des adversarialen Trainings läuft im `eval`-Modus** — festes Dropout, danach Wiederherstellung des vorherigen Modus. Umgeschaltet wird über `self`, **nicht** `self.net`, damit die `train()`-Überschreibung der Basisklasse die Eval-Invariante des eingefrorenen Backbones erneut anwendet | 02 | 04 | ○ | **Neu angelegt 2026-08-06** · Methodisch nötig: gegen ein stochastisches Modell angegriffen wäre das Dropoutrauschen Teil des Gradienten und die Störung nicht reproduzierbar. Ergänzt H13/H15. Umfang: Halbsatz |
| D48 | **Unter Mixup werden die Metriken gegen die *unpermutierten* Labels berichtet** — nur der Verlust nutzt die gemischten Ziele (`lam·CE(y) + (1−lam)·CE(y[perm])`) | 02 | 05 | ○ | **Neu angelegt 2026-08-06** · Ohne diesen Halbsatz ist die Trainingsgenauigkeit eines Mixup-Laufs nicht interpretierbar und mit den übrigen Armen nicht vergleichbar. Gehört zu D14/D30/D36. Umfang: Halbsatz |
| D49 | **Klassengewichte werden zu einer reinen float-Liste normalisiert**, weil ein OmegaConf-Objekt in den Lightning-Hyperparametern das Laden eines Checkpoints unter `weights_only=True` bricht | 02 | 05 | ○ | **Neu angelegt 2026-08-06** · Dieselbe Klasse wie D42 (`add_safe_globals` für den Scheduler): eine Bedingung, unter der die trainierten Checkpoints überhaupt ladbar bleiben. Beide gehören in denselben Absatz. Umfang: Halbsatz |

## E — Evaluation und Metriken

| # | Mechanismus | Reg. | Kap. | Status | Notiz |
|---|---|---|---|---|---|
| E1 | **Videoweise Aggregation**: Score = max Chunk-Wahrscheinlichkeit, Label = „irgendein Chunk fake" | 02 | 04, 06 | ✓ | |
| E2 | Begründung: segmentgenaue Chunk-Labels ⇒ Fake-Video besteht überwiegend aus echten Chunks | 02 | 04 | ✓ | 04 §Min-Overlap leitet die Konsequenz für die Aggregation ab |
| E3 | `val/auc_video` steuert Checkpointing **und** Early Stopping | 02 / 10 | 05 | ✓ | |
| E4 | Metriksatz: AUROC, Accuracy, F1, Average Precision, **Recall@FPR=1 %** | 02 | 06 | ○ | 05-Skizze §Metriken listet den vollständigen Satz |
| E5 | `RecallAtFixedFPR` **adaptiert** torchmetrics' `BinarySensitivityAtSpecificity` an drei Stellen (Umparametrisierung auf `max_fpr`, Skalar statt `(sensitivity, threshold)`-Tupel, `0.0` statt `1.0` bei einklassiger Eingabe) — **keine Neuentwicklung von Grund auf**; gegen Brute-Force verifiziert | 02 / 09 | 06 | ○ | Im Beleg nicht als Eigenimplementierung darstellen · nachgetragen 2026-08-04 in die 05-Skizze (§Evaluationsmetriken) und die 09-Skizze §G (Testzeile); **nicht** in Kap. 6 — die Herkunft einer Metrik ist eine Setup-Aussage, Kap. 6 bleibt deskriptiv |
| E6 | Nur 1-%-Budget: einige hundert Videos können 0,1 % nicht auflösen | 02 | 06 | ○ | 05-Skizze listet nur `recall_at_fpr_0p01_video` auf Videoebene |
| E7 | **Kategorienweise Test-AUC**: `visual` / `audio` / `both` gegen echte Videos | 02 | 06 | ○ | 05- und 06-Skizze inkl. des Degenerationsvorbehalts |
| E8 | Sanity-Check-Durchlauf wird von `val_acc_best` und den Videometriken ausgeschlossen | 02 | 05 | ○ | nachgetragen 2026-08-04 in die 05-Skizze (§Modellauswahl und Monitoring, gemeinsam mit D23) |
| E9 | Begründung des Metriksatzes im Code: **PR-AUC** ist unter Klassenungleichgewicht die belastbare Trennschärfe (Accuracy und F1 zeichnen dort die Klassenprior nach), **Recall bei festem Fehlalarmbudget** die einsatzrelevante Zahl (eine hohe AUROC kann niedrigen Recall bei 1 % FPR verdecken) | 02 | 06 | ○ | 05-Skizze begründet die Primärmetrik exakt so |
| E10 | **Beide** Recall-Budgets (1 % und 0,1 %) werden auf **Chunk**-Ebene geloggt; nur die videoweise Aggregation beschränkt sich auf 1 % | 02 | 06 | ○ | 05-Skizze §Metriken |
| E11 | Eine Fälschungskategorie wird nur geloggt, wenn **beide** Klassen in der Maske vertreten sind; `modify_idx` wird per `amax` aggregiert — exakt, weil alle Chunks eines Videos denselben `modify_type` tragen | 02 | 06 | ○ | 05/06 nennen das Weglassen degenerierter Zellen |
| E12 | Fehlt `video_idx`, wird die **Chunk**-AUC als Ersatz unter demselben Metriknamen geloggt (mit einmaliger Warnung), damit die Callback-Monitore gültig bleiben | 02 | 05, 06 | ○ | Bei Altdaten Metrikherkunft prüfen · nachgetragen 2026-08-04 in die 05-Skizze (§Evaluationsmetriken, an B15); für den Kap.-6-Anteil Vorschlag `–`: die berichteten Läufe weisen kategorienweise Test-AUC aus, die `modify_idx` aus derselben CSV verlangt — der Rückfallpfad war dort nicht aktiv |

## F — Explainable AI

| # | Mechanismus | Reg. | Kap. | Status | Notiz |
|---|---|---|---|---|---|
| F1 | **AttnLRP** (Achtibat et al. 2024) über Input×Gradient auf gepatchten Transformerschichten | 04 | 02, 04 | ✓ | |
| F2 | Versionsgebundene lxt-Patches für VideoMAE und Wav2Vec2 (`transformers==4.57.6`) | 04 | 04 | ○ | 02-Skizze Punkt 4 beschreibt das Monkey-Patching; die Versionsbindung fehlt |
| F3 | **Bivariate Relevanz (Dual-Seed)**: Magnitude = `\|R_fake\|+\|R_real\|`, Direction = `R_fake−R_real` | 04 | 04 | ✓ | In 03 und 04 mit Formeln beschrieben |
| F4 | Kostenoptimierung: 1 Forward + 2 Backwards via `retain_graph` | 04 | 04 | ✓ | 04 nennt „zwei Rückwärtsdurchläufe für einen Vorwärtsdurchlauf“ |
| F5 | **Mathematischer Nachweis**: `R_fake−R_real` = Input×Grad der Logit-Marge (getestet) | 04 / 09 | 04 | ○ | Die Linearität wird behauptet; der Testnachweis fehlt · nachgetragen 2026-08-04 in die 04-Skizze (Halbsatz an der Linearitätsaussage) und die 09-Skizze §G (Testzeile) |
| F6 | Symmetrische Abs-Max-Normalisierung; Null bleibt exakt null (Voraussetzung der Seismic-Colormap) | 04 | 04 | ○ | 09-Skizze D nennt Abs-Max vs. Perzentil |
| F7 | Normalisierungsgranularität wählbar: je Frame / clipglobal / roh | 02 / 04 | 04 | ○ | 09-Skizze D |
| F8 | **Clipglobale Normalisierung** macht Fenster untereinander vergleichbar | 04 / 07 | 04 | ○ | 04- und 09-Skizze begründen clipglobal ausdrücklich |
| F9 | Nachverarbeitung Video: Kanalsumme → 16×16-Patch-Pool → bilineares Upsampling | 02 | 04 | ○ | 09-Skizze D (Datenflussdiagramm) · **Herkunft der 16 nachgeliefert 2026-08-06: siehe C16.** Die Poolweite ist keine frei gewählte Glättung, sondern bildet das Tokengitter ab — VideoMAE nutzt Patches von 16×16 Pixeln, 224/16 = 14 Gitterpunkte je Seite. Beim Ausschreiben mit C16 und C1 als eine Kette führen |
| F10 | **Gemeinsamer multimodaler Rückwärtspass** — Video- und Audiorelevanz auf gleicher Skala | 04 | 04 | ○ | 02- und 05-Skizze nennen den gemeinsamen Backward |
| F11 | Audio-LRP ab dem CNN-Ausgang, Kanalmittel + Interpolation auf Sample-Ebene | 02 | 04 | ○ | 02-Skizze Punkt 4 und 07 §Reflexion nennen die CNN-Grenze samt Begründung |
| F12 | **SDPA-Training / Eager-Erklärung**, mit Guard + Test abgesichert | 02 / 04 | 04, 05 | ✓ | 04 §Eager-Attention; 05-Skizze ergänzt den Paritätstest |
| F13 | Audio-Schicht **L1**: Abs-Max-Pooling (nicht Mittelwert — sonst Auslöschung) über 10-ms-Fenster | 04 | 04 | ○ | L1 ist beschrieben, aber laut eigenem §§-Kommentar falsch (Relevanz liegt **neben**, nicht unter der Wellenform); das Pooling fehlt · nachgetragen 2026-08-04 in die 04-Skizze. **Bezeichnung korrigiert:** Docstring und Matrixzeile sagen „Abs-Max-Pooling", der Code rechnet ein **Mittel der Beträge** mal dem Mehrheitsvorzeichen (`avg_pool1d` über `.abs()`, `audio_xai.py:204-207`) — kein Maximum |
| F14 | Audio-Schicht **L2**: WhisperX-Forced-Alignment, vorzeichenbehaftete Mittelung je Wort, Plattencache | 04 / 07 | 04 | ✓ | ~~04 sagt „aufsummiert“~~ — **korrigiert 2026-08-06** in `04Methodology.tex:263`. **Beide Pfade verifiziert und identisch:** `audio_xai.aggregate_word_relevance` (offline) und `_compute_word_segments` (`src/api/inference.py:2129`, `chunk.mean()`) mitteln vorzeichenbehaftet über die Samples des Wortes. Damit ist die Prüfaufgabe aus F49 erledigt und der Widerspruch **nicht** zweiteilig. **Herkunft des Fehlers:** [`docs/archive/xai.md` §3](../archive/xai.md) formuliert „Relevanz wird pro Wort-Token aufsummiert“ |
| F15 | Audio-Schicht **L3**: drei perzeptuelle Bänder (0–500 / 500–4k / 4k–8k Hz) | 04 / 07 | 04 | ✓ | Bänder und phonetische Deutung stehen in 02 und 04 |
| F16 | **Band-Ablation (`_band_confidence`)**: kausale statt attributiver Aussage, nullphasiger Butterworth | 07 | 04 | ○ | 04-Skizze und 07 §Reflexion trennen Ablations-Konfidenz und Relevanz |
| F17 | **Confidence vs. Relevance** als durchgängige Unterscheidung, bis in die Typen des Frontends | 07 / 08 | 04 | ○ | 04-Skizze Punkt 1 und 07 §Reflexion |
| F18 | **Gesichtsregionen-Partition** aus FaceMesh-Landmarks (personenspezifisch statt fester Rechtecke) | 07 | 04 | ! | **04 nennt „Mund, Augen, Kiefer, Schultern und Hintergrund“** — es sind sieben landmarkbasierte Regionen ohne Schultern und ohne Hintergrund. Vom Autor selbst als „faktisch falsch“ markiert, Korrektur steht in der 04-Skizze. **Ursache gefunden und beseitigt (2026-08-06).** Die Fünferliste „Mouth, Eye, Jaw, Shoulder, Background“ stand an **zwei** Stellen, beide nicht maßgeblich: in [`docs/archive/adversarial.md` §2.1](../archive/adversarial.md) (Planungsdokument, wörtlich dieselbe Aufzählung — die **wahrscheinlichere** Quelle des Belegsatzes) und in den Attrappenzeilen von `lib/mockData.ts`. **Beide sind bereinigt:** das Archivdokument trägt einen Korrekturhinweis, die Mockzeilen nutzen jetzt reale Regionsnamen, und `bshift` führt die kanonische Liste im Docstring. **Die Reichweite im Beleg ist größer als dieser eine Absatz:** weil außerhalb des Gesichtsovals keine Region definiert ist, ist eine \enquote{Verschiebung auf den Hintergrund} mit der implementierten Partition **nicht messbar** — messbar ist nur eine Umverteilung *zwischen* den sieben Gesichtsregionen. Genau diese Erzählung steht in `00Abstract.tex:14`, `01Einleitung.tex:161-162` und der 06-Skizze (`:222`, geplante Shift-Tabelle \enquote{semantische Regionen → Hintergrund}). Die 07-Skizze (`:123-128`) benennt das Problem bereits und schlägt als Ersatzgröße den Mean Attention Shift über alle Regionen vor. Modus C hat damit **vier** Stellen zu korrigieren, nicht eine |
| F19 | **Attention Shift** — Verschiebung der Begründung zwischen sauber und gestört | 07 / 08 | 04, 06 | ○ | Mechanismus und Forschungslücke G4 ✓; die Regionsbasis ist falsch (s. F18) und die Messgröße laut 07 nachzuziehen · nachgetragen 2026-08-04 in die 04-Skizze (§Attention-Shift, gemeinsam mit F42); für den Kap.-6-Anteil Vorschlag `–`: die 06-Skizze führt die Shift-Analyse bereits als ausstehende Lücke |
| F20 | 2-D-Yaw-Proxy + Rotationswarnung bei nahezu profiler Kopfhaltung | 01 / 07 | 04, 07 | ○ | 04-Skizze V10 nennt die Kopfrotations-Warnung |
| F21 | Heatmap-**Rückprojektion** in die Originalauflösung | 07 | 04 | ○ | 04- und 09-Skizze |
| F22 | Seismic-Colormap, literaturgestützt (Schloss 2019, Schoenlein 2026) | 08 / 12 | 04 | ✓ | Beide Quellen in 03 und 04 zitiert |
| F23 | **Darstellungsverstärkungen** (Gamma, Gain, Cap) — Farben zeigen relative, nicht absolute Werte | 07 / 08 | 04 | ○ | Abbildungslegenden prüfen · 05-Skizze verweist auf die Parametertabelle im Anhang, 09-Skizze D übernimmt sie |
| F24 | Drei Hydra-Erklärskripte erzeugen die Abbildungen reproduzierbar | 04 | 05, 09 | ○ | Die Reproduzierbarkeit der Abbildungen ist nirgends benannt · nachgetragen 2026-08-04 in die 05-Skizze (§Laufzeitkonfiguration der xAI-Analyse, gemeinsam mit F31); **nicht** in Kapitel 9 — Laufbedingung der Abbildungserzeugung, gehört ins Setup |
| F25a | **Diagnose** der flächigen Heatmap: Betreuer-Kritik aufgenommen, an echten Fake-Frames gemessen, Normierung und Thresholding als Ursache **ausgeschlossen** — ein verwertbares Ergebnis, unabhängig von der Lösung | 12 | **07** | ○ | 06-Skizze §xAI-Ergebnisse und 07 §FF1 — mit allen Zahlen |
| F25b | **Explanation-Guided-Training mit Frame-Difference-Masken** — 🔨 geplant und bestätigt, **zum Registerstand nicht implementiert** | 12 | 04+06 *oder* 08 | ○ | Kapitelzuordnung hängt davon ab, ob die Umsetzung vor Abgabe landet · 06/07/08 führen es konsistent als geplant/Ausblick |
| F25c | Zwei Punkte zu F25b gehören **unabhängig vom Ausgang** in den Beleg: die methodische Spannung eines Explanation-Guided-Loss (Prior *auf die Erklärung* — von „entdecken, warum" zu „vorschreiben, wohin"; Ross et al. 2017 dafür, konstruierte statt entdeckte Erklärung dagegen) und der Trade-off **Lokalisierung ↑ vs. Accuracy ↓** als eigenes Ergebnis | 12 | 04, 07 | ○ | 07 §FF1 fordert die Spannung explizit ein |
| F26 | **Die drei Hydra-Erklärskripte nutzen den Dual-Seed nicht** — `per_class` ist überall auf `False` vorbelegt; ihre Abbildungen sind klassische **Single-Seed**-Karten. Der bivariate Pfad läuft ausschließlich über `src/api/inference.py` | 04 | 04, 06 | ○ | Skript- und Frontend-Abbildungen desselben Clips sind **nicht dieselbe Größe** · **Fehlerquelle für Abbildungen:** Skript- und Frontend-Karten sind verschiedene Größen · nachgetragen 2026-08-04 in die 04-Skizze (Reichweite der bivariaten Kodierung) und die 06-Skizze §sec:results-xai (Herkunftshinweis, gemeinsam mit F27/F28); verifiziert: `per_class=True` kommt in `src/` nur in `api/inference.py` vor |
| F27 | `audio_xai.compute_band_relevance` verwendet die **überholte L3-Formel** (Skalarprodukt); die Laufzeitpipeline wurde auf energiegewichtete Mittelung umgestellt, `audio_xai` nicht nachgezogen | 04 / 07 | 04, 06 | ○ | L3-Abbildungen der Skripte **nicht** als Aussage über frequenzabhängige Modellaufmerksamkeit verwenden · Betrifft jede L3-Abbildung, die aus den Skripten stammt · nachgetragen 2026-08-04 in die 06-Skizze §sec:results-xai (gemeinsam mit F26/F28); die Begründung der Laufzeitformel steht als F36 in der 04-Skizze |
| F28 | Die Videofiguren der beiden Skripte sind **nicht gleich skaliert**: `explain.py` zeichnet fest `±1`, `explain_multimodal.py` mit dem Betragsmaximum *des gewählten Frames* | 04 | 04 | ○ | Nicht ohne Hinweis nebeneinanderstellen · nachgetragen 2026-08-04 in die 06-Skizze §sec:results-xai (gemeinsam mit F26/F27); verifiziert: `explain.py:78/84` fest ±1 gegen `explain_multimodal.py:113` `hm_vmax` |
| F29 | Die lxt-Patches sind **idempotent** (`_lxt_patched`) — ohne diesen Wächter würde die Attention im multimodalen Aufbau mehrfach umwickelt und der Gradient mehrfach durch den Softmax geteilt; der Fehler bliebe still (Heatmap entstünde weiter, nur falsch verteilt) | 04 | 04 | ○ | nachgetragen 2026-08-04 in die 04-Skizze (§AttnLRP, an den Eager-Absatz); löst den §§-Marker „vllt noch auf das monkey patching eingehen"; verifiziert: `attnlrp.py:89-91` und `:125-127` |
| F30 | **Auflösungsgrenze der Audiorelevanz: ~20 ms.** Der Conv-Extraktor reduziert 10.240 Samples auf **31 Frames**; die Rückgabe der Form `(B, T_samples)` enthält also 31 unterschiedliche Werte. Der L1-Kernel (160 Samples) liegt *unter* einem Wav2Vec2-Frame — 64 Bins tragen 31 Werte, je zwei benachbarte sind identisch | 02 / 04 | 04, 07 | ○ | Zeitliche Lokalisierung nie feiner als ~20 ms angeben · 07 §Reflexion nennt ~320 Samples als Auflösungsgrenze |
| F31 | Alle drei Erklärskripte erklären **ein einzelnes Sample** — den ersten Eintrag des ersten Test-Batches (`[0:1]`), keinen Datensatz | 04 | 04, 05 | ○ | Limitation der Abbildungen · nachgetragen 2026-08-04 in die 05-Skizze (§Laufzeitkonfiguration der xAI-Analyse, gemeinsam mit F24); **nicht** in Kapitel 4 — Laufbedingung, keine Designbegründung; verifiziert: `[0:1]` in `explain.py:53`, `explain_audio.py:63`, `explain_multimodal.py:88` |
| F32 | `_percentile_normalize` (99. Perzentil statt Abs-Max) — robuster gegen Einzelausreißer; Anlass laut Docstring: Abs-Max drückte Wortbalken und L1-Band gegen Weiß | 07 | 04 | ○ | 09-Skizze D |
| F33 | **L3 Band × Zeit-Gitter** in zwei Ausführungen — Confidence (Ablationsanteil `(base−ablated)/base` je 0,64-s-Fenster, clipübergreifend vergleichbar) und Relevance (bivariate Gradientenrelevanz) — plus ein multimodales Gitter, das bei fixem Video nur das Audio bandweise entfernt | 07 / 08 | 04, 06 | ○ | 04-Skizze V8 nennt Balken **und** Band-×-Zeit-Gitter |
| F34 | Beide Ablationsgitter sind **fakeness-gated** (`base[w] > 0`): ein REAL-Clip rendert konstruktionsbedingt als leeres Gitter — beabsichtigte Aussage, kein fehlgeschlagener Lauf | 07 / 08 | 04, 06 | ○ | nachgetragen 2026-08-04 in die 04-Skizze (§Akustische xAI, Ebene 3) und die 06-Skizze §sec:results-xai (Bildunterschriftshinweis, gemeinsam mit F37/F40) |
| F35 | **Empirischer Befund (im Code „verified"):** Ganzclip-Audio liegt außerhalb der Trainingsverteilung — ein Forward über die ganze Wellenform sagt selbst bei FAKE-Clips REAL, weil das eine manipulierte Fenster weggemittelt wird. Deshalb werden genau die urteilsbildenden Fenster erklärt | 07 | 04, 07 | ○ | Als Entwicklungsbefund kennzeichnen oder neu messen · Begründet, warum fensterweise erklärt wird — fehlt als Begründung · nachgetragen 2026-08-04 in die 04-Skizze (§Akustische xAI, Vorspann) und die 07-Skizze §Methodische Reflexion (Sammelabsatz mit F36/F37/F38) |
| F36 | **Empirischer Befund:** das frühere Bandmaß (Skalarprodukt) lag inhaltsunabhängig bei ~0,43 / 0,56, weil Sprachenergie fast vollständig in Low + Mid liegt; die Division durch die Bandenergie entfernt diesen Bias | 07 | 04 | ○ | **Die Werte ~0,43 / 0,56 stammen aus einem behobenen Fehler und gehören nicht in den Beleg** (Autorenentscheidung 2026-08-04) — die Registerzeile dokumentiert weiterhin, was im Docstring steht. Übernommen wird nur die heutige Formel (energiegewichtetes Mittel) samt qualitativer Begründung · nachgetragen 2026-08-04 in die 04-Skizze (§Akustische xAI, Ebene 3); für den Kap.-7-Anteil `–`: ohne die Bugwerte bleibt kein Diskussionsgegenstand, die Formel selbst ist Methodik |
| F37 | **Empirischer Befund:** das Relevanz-Gitter ist „ehrlich blass" — Gradientenrelevanz lokalisiert nicht nach Frequenz wie die Ablation; es existiert für die Umschaltkonsistenz | 07 / 08 | 06, 07 | ○ | Blasses Gitter ist Befund, nicht Fehler · Ohne diesen Hinweis wirkt das blasse Gitter wie ein Fehler · nachgetragen 2026-08-04 in die 06-Skizze §sec:results-xai (Beobachtung, mit F34/F40) und die 07-Skizze §Methodische Reflexion (Sammelabsatz mit F35/F36/F38) |
| F38 | **Empirischer Befund:** die Wortrelevanz nutzte früher `argmax(\|·\|)` und zeichnete auf echten Clips die größte Rauschspitze in voller Höhe; das Mittel über die Wortsamples ersetzt das | 07 | — | – | **Bewusst weggelassen** (Autorenentscheidung 2026-08-04): veraltetes Artefakt, das auf den heutigen Code nicht mehr zutrifft — derselbe Maßstab wie bei den aus F36 gestrichenen Bugwerten. Der Beleg beschreibt den Endstand, nicht die Fehlerhistorie. **Davon unberührt:** der `!`-Widerspruch F14 (Kapitel 4 sagt „aufsummiert", der Code mittelt vorzeichenbehaftet) bleibt in Modus C zu korrigieren |
| F39 | **Harte L2-Grenzen:** WhisperX ist fest auf Modell `medium` und `language="en"` verdrahtet (Laufzeit) bzw. `base`/`en` (Skripte); ist WhisperX nicht installiert, entfällt die Schicht **ohne Fehlermeldung** | 07 / 11 | 04, 07 | ○ | Limitation · 05-Skizze nennt WhisperX `medium`, 07-Limitation 17 die Abhängigkeit; ~~`language="en"` fehlt~~ · **nachgetragen 2026-08-06** (Verifikationslauf) in die 04-Skizze §Akustische xAI, Ebene 2 — als **Reichweitengrenze der Methode**: die Wortebene existiert nur für englischsprachige Clips |
| F40 | **Asymmetrie der Bandkonfidenz:** unimodal läuft `_band_confidence` gegen die **Max**-Marge über die Fenster, multimodal gegen die **Mittel**-Marge — die beiden 3-Balken-Ansichten sind nicht gegeneinander lesbar; für den Mittelwert steht keine Begründung im Code | 07 | 04, 06 | ○ | nachgetragen 2026-08-04 in die 04-Skizze (§Akustische xAI, Ebene 3, gemeinsam mit F41) und die 06-Skizze §sec:results-xai (Leseregel, mit F34/F37) |
| F41 | Die drei Confidence-Balken sind **auf das stärkste Band normiert** — ablesbar sind Verhältnis und Vorzeichen, **nicht** der absolute Ablationseffekt; das Band × Zeit-Gitter verhält sich umgekehrt (Anteil, clipübergreifend vergleichbar) | 07 | 04 | ○ | nachgetragen 2026-08-04 in die 04-Skizze (§Akustische xAI, Ebene 3, gemeinsam mit F40 als ein Absatz) |
| F42 | `anomalyRegions` (aus der Einzelziel-FAKE-Karte) und `regionRelevance` (bivariat) stammen aus **verschiedenen Karten** und dürfen unterschiedliche Regionen vorn zeigen, ohne dass eine falsch ist | 07 | 04 | ○ | nachgetragen 2026-08-04 in die 04-Skizze (§Attention-Shift, gemeinsam mit F19 als ein Absatz) |
| F43 | **Rückfallketten sind größtenteils unmarkiert:** nur `degradedFaceLost` erscheint im Ergebnis; der Vollbildpfad ist nur indirekt erkennbar (fehlendes `cropBox`), geometrische Regionsaufteilung und Ganzwellenform-Rückfall erzeugen **gar kein** Kennzeichen | 07 | 04, 07 | ○ | Nur am Log erkennbar — Ehrlichkeitspunkt · nachgetragen 2026-08-04 in die 04-Skizze (§Attention-Shift, gemeinsam mit F44) und die 07-Skizze §Limitationen (Ergänzungsliste Punkt 20) |
| F44 | Der geometrische Regionsrückfall benutzt **überlappende** Rechtecke (Kinn im Kiefer, Mund ragt hinein) — die Nichtüberlappungsgarantie gilt nur für die Landmark-Partition | 07 | 04 | ○ | nachgetragen 2026-08-04 in die 04-Skizze (§Attention-Shift, gemeinsam mit F43 als ein Absatz) |
| F45 | Frames außerhalb des HDF5-Pfads laufen über `_frame_transform` (PIL/torchvision), HDF5-Chunks über `cv2` — dieselbe Normalisierung, aber **nicht bitgleiche Interpolation** zur Trainingsvorverarbeitung | 07 | 04 | ○ | nachgetragen 2026-08-04 in die 04-Skizze (neuer §Demonstrator, als Präzisierung der Train/Serve-Parität B1) |
| F46 | **Der Novelty-Anspruch ist entschieden und wörtlich vorformuliert:** „eine bewusste Engineering-Komposition etablierter Methoden" mit dem Zusatz *„nach unserem Kenntnisstand … nicht beschrieben"*. Zitierpflichtige Bausteine: CLRP (Gu 2018), SGLRP (Iwana 2019), Tsunakawa 2019, LXT/Walter 2025; Abgrenzung gegen Oh & Noh 2025 (methodisch) und Payne 2024 (visualisierungsseitig) | 12 | 02, 04, 07 | ✓ | **Kein** Anspruch auf fundamentale Novelty — kein systematischer Review erfolgt · 03 §Positionierung, 04 §bivariat und 07 §Reflexion halten den Anspruch konsistent bescheiden |
| F47 | **Faithfulness-Caveat:** die Zahlen des AttnLRP-Papers wurden auf **Single-Target** gemessen, nicht auf der hier verwendeten contrastiven Variante — laut Quelldokument im Beleg explizit so zu benennen | 12 | 04, 07 | ○ | 07 nennt einen **anderen** Treue-Vorbehalt (kein Perturbationstest); dass die Paper-Zahlen auf Single-Target gemessen wurden, fehlt · nachgetragen 2026-08-04 in die 07-Skizze §Methodische Reflexion; für den Kap.-4-Anteil Vorschlag `–`: Kapitel 4 beschreibt das Verfahren, die Treuebewertung ist Deutung |
| F48 | Migrationsstand bivariat: vollständig für Echtclip-Heatmaps; **Differenzkarten, Confidence-Ansichten und Audio-L2 sind bewusst nicht bivariat** | 04 | 04 | ○ | nachgetragen 2026-08-04 in die 04-Skizze (§Bivariate Relevanz-Heatmap, Abgrenzungssatz) |
| F49 | **Die Audio-xAI existiert zweimal.** `src/utils/audio_xai.py` hat genau zwei Importeure (`explain_audio.py`, `explain_multimodal.py`); `src/api/inference.py` nutzt es **nicht**, sondern implementiert Wortaggregation und Frequenzbänder eigenständig neu — teils mit anderen Formeln. Geteilt ist zwischen Offline-Figuren und Laufzeit-xAI **allein `attnlrp.py`** | 04 | 04 | ○ | **Neu angelegt 2026-08-06** · Das ist die gemeinsame Ursache von F26, F27 und F28 — bislang stehen die drei als Einzelbefunde in den Skizzen, ohne dass der strukturelle Grund benannt wäre. **Prüfaufgabe erledigt 2026-08-06:** die Laufzeit mittelt die Wortrelevanz mit **derselben** Formel wie der Offline-Pfad (`inference.py:2129`, `chunk.mean()`) — der Widerspruch F14 war also nicht zweiteilig und ist korrigiert. Die Verdopplung wirkt sich auf L3 (F27) und die Farbwege (F28/F53) aus, nicht auf L2. Umfang: 1 Satz |
| F50 | **Die Skriptfiguren erklären die *vorhergesagte* Klasse, nicht ein festes FAKE-Ziel.** `configs/explain.yaml` setzt `target_class: null`, und `compute_attnlrp` erklärt dann `argmax(logits)`. Bei einem als REAL klassifizierten Clip zeigt die Karte also Evidenz **für REAL** | 04 | 05, 06 | ○ | **Neu angelegt 2026-08-06** · Ohne diesen Satz liest eine Prüferin jede Skript-Heatmap als Fake-Evidenz. Schließt an F26 (Single-Seed) und F31 (ein Sample) an und gehört in denselben Absatz der 05-Skizze. Umfang: Halbsatz |
| F51 | **L3 der Skripte ist auf `Summe der Beträge = 1` normiert** (vorzeichenerhaltend). Die drei Bandwerte sind damit **Anteile**, keine absoluten Relevanzen — genau die Normierung, die der Frontend-Kommentar in S27 als \enquote{the same lie the backend sum=1 normalisation made} bezeichnet | 04 | 04 | ○ | **Neu angelegt 2026-08-06** · S27 zitiert diese Normierung bereits, ohne dass sie irgendwo definiert wäre; F41 beschreibt die *Frontend*-Normierung auf das stärkste Band, was etwas anderes ist. Umfang: Halbsatz |
| F52 | **Die Wellenform in den L1-Skriptfiguren ist nicht das Audio**, sondern das z-normalisierte Modelleingangssignal (Spitzen von ±3--4 σ). Das obere Panel hat deshalb bewusst kein festes `ylim` | 04 | 06 | ○ | **Neu angelegt 2026-08-06** · Gehört in die Bildunterschrift jeder L1-Abbildung: die gezeigte Amplitude ist keine Lautstärke. Verwandt mit B1 (Train/Serve-Parität) und F45. Umfang: Halbsatz |
| F53 | **Es gibt drei Farbkonventionen, nicht zwei.** Neben der aufgehellten F2-Rampe des Frontends und matplotlibs seismic im Backend-PNG (S28) nutzen die Skriptfiguren eine eigene Kombination: seismic mit fester ±1-Skala für den L1-Streifen, `firebrick`/`steelblue` für die L2- und L3-Balken | 04 / 08 | 04, 06 | ○ | **Neu angelegt 2026-08-06** · S28 und S35 beschreiben nur Frontend gegen Backend. Wer die Regel \enquote{je Abbildung den Erzeugungsweg nennen} anwendet, braucht alle drei. Umfang: Halbsatz, an den S28+S35-Absatz |
| F54 | **Welche Audiofiguren überhaupt entstehen, hängt an zwei Schaltern und an WhisperX.** `explain_multimodal.py` erzeugt bis zu fünf Dateien; L2 und L3 entfallen bei abgeschaltetem `enable_layer2`/`enable_layer3` oder wenn WhisperX keine Segmente liefert. **L3 läuft dabei unabhängig von L2** — die beiden Zweige sind bewusst als `else`-Blöcke statt als frühe `return`s ausgeführt, weil vorher ein stummer Clip auch die Frequenzbandabbildung verschluckte | 04 | 05 | ○ | **Neu angelegt 2026-08-06** · Erzeugungsbedingung der Abbildungen, dieselbe Klasse wie F24/F31/S11/Q13; erklärt außerdem, warum für manche Clips nur ein Teil der Schichten vorliegt. Umfang: 1 Satz |
| F55 | **Korrektheitsbedingung des Dual-Seeds:** `compute_attnlrp_per_class` setzt `x.grad = None` **vor jedem Seed**. Ohne diesen Reset akkumulierten die beiden Rückwärtspässe auf demselben Gradienten und die Direction-Karte wäre still falsch. Beide Kernfunktionen werfen zudem explizit, wenn `.grad` `None` bleibt — multimodal unter Nennung des Tensorindex, sodass eine nur teilweise gepatchte Backbone-Kombination sofort auffällt | 04 | 04 | ○ | **Neu angelegt 2026-08-06** · F4 behauptet die Kostenoptimierung (1 Forward, 2 Backwards über `retain_graph`); dies ist die Bedingung, unter der sie *korrekt* ist. Ohne den Halbsatz ist die Optimierung eine unbelegte Abkürzung. Umfang: Halbsatz an F4 |
| F57 | **Attention Rollout ist nirgends implementiert.** Eine repositoriumsweite Suche nach `rollout` über alle `.py`, `.ts`, `.tsx` und `.yaml` liefert **keinen einzigen Treffer** in `src/`, `scripts/` oder `frontend/`. Es existiert kein Modul, keine Funktion und keine Konfiguration dafür; die einzige xAI-Implementierung ist `attnlrp.py` | 04 / 00 | 02, 04, 06, 08, 09 | ! | **Neu angelegt 2026-08-06** (Archivprüfung) · **Der Beleg behauptet an fünf Stellen das Gegenteil:** `04Methodology.tex:226-228` führt einen eigenen Unterabschnitt \enquote{Attention Rollout (Baseline)} mit \enquote{Als Vergleichsbasis dient…} und \enquote{wird in dieser Arbeit primär als leichtgewichtige Referenz herangezogen}; `06Results.tex:155-157` plant eine Vergleichstafel \enquote{Rollout vs. AttnLRP an denselben Frames}; `09Appendix.tex:46` eine weitere; `08Conclusion.tex:38` behauptet im Fazit \enquote{AttnLRP als Primärmethode mit Attention Rollout als Referenz}. **Herkunft:** [`docs/archive/xai.md` §1](../archive/xai.md) führt Rollout als \enquote{Lösung 1} der Planungsphase. **Der Autor hat den Fehler selbst vermutet** (`%§§` in `04Methodology.tex:227`: \enquote{aber das ist doch gar nicht unsere baseline?}) und in `:229` bereits die Auflösung notiert. **Kapitel 2 ist nicht betroffen** — dort ist Rollout Methodengrundlage und darf als solche stehen bleiben. **ERLEDIGT 2026-08-06 (Autorenentscheidung: Rollout wird nicht implementiert, Aussagen über seine Verwendung sind strikt falsch, es dient allein dem Verständnis von AttnLRP).** Korrigiert: `04Methodology.tex` — der Unterabschnitt heißt jetzt \enquote{Von Attention Maps zu AttnLRP} und sagt ausdrücklich, dass Rollout weder Baseline noch Referenz ist; `06Results.tex:155` und `09Appendix.tex:46` — die beiden Vergleichstafeln sind gestrichen (im Anhang bleibt der herstellbare Vergleich Single-Seed gegen bivariat, F26); `08Conclusion.tex:38` — der Halbsatz \enquote{mit Attention Rollout als Referenz} ist entfernt. Alle vier Stellen tragen einen datierten Kommentar mit der Zeilen-ID |
| F56 | Vier Kleinmechanismen der Erklärskripte: `compute_attnlrp` kapselt alles in `torch.enable_grad()` (aus `no_grad` heraus sicher aufrufbar); `LABEL_NAMES` ist die einzige Quelle der Figurenbeschriftung; `ckpt_path` wird doppelt erzwungen (Hydra `???` **und** Laufzeit-`ValueError`); der WhisperX-Cacheschlüssel enthält neben der Wellenform auch den Sprachcode | 04 | — | – | **Neu angelegt 2026-08-06**, zugleich **bewusst weggelassen**: Implementierungshygiene ohne Bezug zu einer Forschungsfrage. Sammelzeile, damit die Prüfung dokumentiert ist und nicht erneut aufläuft |

## G — Phase 3: Robustheit

| # | Mechanismus | Reg. | Kap. | Status | Notiz |
|---|---|---|---|---|---|
| G1 | **Drei** gesweepte Degradationsachsen: CRF, Bildratenreduktion, Downscale→Upscale | 05 | 04, 06 | ○ | 04 nennt die Achsen, zählt aber Rauschen mit (s. G1b) · nachgetragen 2026-08-05 in die 04-Skizze (gemeinsam mit G4, ein Absatz): drei Videoachsen als Gitter CRF×FPS plus je ein eigener Durchgang für Skalierung und Audiobitrate. Die Rauschfrage bleibt beim `!`-Widerspruch G1b |
| G1b | **Rauschen ist nicht Teil des Offline-Sweeps** — der `noise`-Filter existiert nur im interaktiven Pfad (`_ffmpeg_degrade`) und ist zeitlich variierendes **Gleichverteilungs**rauschen (`allf=t+u`), nicht gaußsch | 05 / 07 | 04, 06 | ! | Nicht als gesweepte Achse führen; Frontend beschriftet den Regler dennoch „Gauß" · **04 führt Gauß-Rauschen als Sweep-Achse.** Verifiziert: `eval_robustness_sweep.py` kennt keinen Rauschparameter. Auch die 04-Skizze irrt hier („im Sweep-Code als Achse vorhanden“) |
| G2 | Audiodegradation getrennt über AAC-Bitrate | 05 / 07 | 06 | ✓ | |
| G3 | **CRF × FPS-Gitter** über den Testsatz | 05 | 06 | ○ | 05- und 06-Skizze nennen das vollständige Gitter |
| G4 | **Upscale-Sweep** `scale=640:360,scale=1280:720`. Relativ zur 224×224-Quelle ist das eine **Hochskalierung mit Seitenverhältniswechsel** (1:1 → 16:9), **kein** Reupload in Originalauflösung; der Detailverlust stammt aus Resampling-Kette und Neukodierung | 05 | 06 | ○ | Motivation „TikTok/WhatsApp" belastbar, Vorgangsbeschreibung nicht · 04 schreibt „Down-/Upsampling“ (eigener Zweifel im §§-Kommentar berechtigt); 05/06 nennen korrekt nur die Upscale-Stufe · nachgetragen 2026-08-05 in die 04-Skizze (mit G1) und als Halbsatz an der Upscale-Zeile der 06-Skizze |
| G5 | Multimodaler Sweep mit **gemeinsamer** Video- und Audiodegradation | 05 | 06 | ○ | 05-Skizze inkl. der Interpretationsgrenze „Audio fest bei 64 kbps“ |
| G6 | `face_lost`-Flag: Ausfall der *Gesichtserkennung* getrennt vom Detektorversagen | 07 | 06, 07 | ○ | Ohne das Flag ist nicht unterscheidbar, ob Detektor oder Klassifikator versagt · nachgetragen 2026-08-05 in die 04-Skizze (mit G7, ein Absatz) und die 07-Skizze §FF3 (Deutung); für den Kap.-6-Anteil Vorschlag `–`: das Flag entsteht nur im interaktiven Pfad, die Phase-3-Zahlen stammen aus dem Offline-Sweep |
| G7 | Rückfallbox aus dem sauberen Lauf — aber nicht bei Auflösungsänderung | 07 / 09 | 04 | ○ | nachgetragen 2026-08-05 in die 04-Skizze (§Phase 3, gemeinsam mit G6) |
| G8 | **Breaking Point ist keine Kipppunktsuche.** `BreakingPoint` (`RobustnessPanel.tsx:188`) führt keinen Sweep durch, sondern stuft den relativen Konfidenzverlust *eines* gefahrenen Parametersatzes ein: `critical` > 50 %, `moderate` > 25 %, sonst `low`; eigene Pfade für „Konfidenz steigt" und „< 0,05 pp Änderung" | 08 | 06 | ○ | „erster Parameterwert, an dem das Urteil kippt" wäre eine Auswertung, die es nicht gibt · Als sweepbasierter Einbruchpunkt in 02/05/06 legitim verwendet; die **Frontend-Komponente** darf nicht als Kipppunktsuche zitiert werden · nachgetragen 2026-08-05 in die 06-Skizze §Phase 3 als Bildunterschriftshinweis. **Achtung:** die Kennzahlentabelle in [05](05_robustheit_adversarial.md) definiert Breaking Point selbst als „erster Gitterpunkt, an dem das Urteil kippt" und verweist dafür auf `RobustnessPanel.tsx:188` — das widerspricht dem eigenen Kasten in [08](08_frontend.md); maßgeblich ist der Code |
| G9 | Robuste Augmentierung als Gegenmaßnahme (`train_*_robust`) | 01 / 10 | 04, 06 | ○ | 05-Skizze führt `augment_strength=robust` als Arm |
| G10 | Ergebnisse dokumentiert in `vault/Results/phase3-robustness-social-media-sweep.md` + 3 Abbildungen | 12 | 06 | ○ | 06-Skizze bindet Notiz und drei Abbildungen ein |
| G11 | **Die voreingestellten Gitter konkret:** CRF `18 23 28 35 40 45 51` × FPS `25 15 10 5` = **28 Videogitterpunkte**; AAC `128 64 32 16` kbps bei fest CRF 23 / FPS 25; Upscale-Durchgang ebenda; der multimodale Sweep fährt dasselbe CRF×FPS-Gitter bei fest 64 kbps | 05 | 05, 06 | ○ | 05-Skizze §Sweeps |
| G12 | Fehlt ein Checkpoint oder lässt sich das Modell nicht laden, wird der betreffende Teil-Sweep **mit Warnung übersprungen statt abgebrochen** — ein Lauf kann unvollständig durchlaufen, ohne zu scheitern | 05 | 06 | ○ | Vollständigkeit der Ergebnistabellen prüfen · Erklärt potenziell unvollständige Ergebnistabellen · nachgetragen 2026-08-05 in die 06-Skizze §Phase 3 (mit G13, ein Absatz vor der Haupttabelle) |
| G13 | `--multimodal` bedeutet in den beiden Sweeps **Verschiedenes**: im Robustheitssweep ein *zusätzlicher* Arm, im Adversarialsweep ein *ersetztes* Ziel (Video- **oder** Fusionsmodell, nie beide) | 05 | 06 | ○ | nachgetragen 2026-08-05 in die 06-Skizze §Phase 3 (gemeinsam mit G12) |
| G14 | `mean_fake_prob_delta = baseline − gestört`; **positives Vorzeichen bedeutet Verschiebung Richtung REAL** | 05 | 06 | ○ | Vorzeichenkonvention in jede Ergebnistabelle · 06-Skizze berichtet Δfake mit Vorzeichen; die Konvention selbst fehlt |
| G15 | Verschiedene Ground-Truth-Aufgaben je Sweeparm: Audiosweep gegen `label_audio`, multimodaler Sweep gegen das kombinierte `label`, Videosweep gegen `label` — die Zahlen sind nicht dieselbe Größe | 05 | 06 | ○ | 06-Skizze nennt die abweichende Label-Basis des Video-Zweigs |
| G16 | Der Sweep poolt Videolabels per „ein Video ist fake, wenn irgendein Chunk fake ist" — **dieselbe Regel** wie die Trainingsevaluation (E1) | 05 | 06 | ○ | ~~(Notizspalte war leer -- der Punkt hatte kein Ziel)~~ · **nachgetragen 2026-08-06** (Verifikationslauf) in die 06-Skizze §Phase 3, im Vollständigkeitsabsatz mit G12/G13/G18. **Warum es zählt:** die Bestätigung ist die Bedingung dafür, dass die Phase-3-Zahlen mit den Phase-1/2-Zahlen auf derselben Aggregationsebene liegen |
| G17 | `_run_audio_for_robustness` reduziert die Audioinferenz auf Konfidenz und Frequenzbänder (kein WhisperX, kein Relevanz-Rückwärtspass) — der Phase-3-Audiotest liefert keine L2-Schicht | 07 | 06 | ○ | nachgetragen 2026-08-05 in die 06-Skizze §Phase 3 · verifiziert und **verschärft:** auch der Offline-Sweep verzichtet auf die Wortebene — `eval_robustness_sweep.py` nutzt `run_audio_inference_score` (`src/api/inference.py:2605`), das zusätzlich den Input×Gradient-Rückwärtspass auslässt |

| G18 | **Die effektive Stichprobengröße schwankt je Gitterpunkt.** Clips, deren Inferenz an einem Gitterpunkt scheitert, fallen dort **samt ihrem Baselineeintrag** heraus; Gitterpunkte ohne einen einzigen gültigen Clip werden übersprungen | 05 | 06 | ○ | **Neu angelegt 2026-08-06** · G12 erklärt fehlende *Teil-Sweeps* (Checkpoint nicht ladbar), G18 die schwankende Zeilenbasis **innerhalb** eines Sweeps. Zusammen mit dem NaN-Sentinel `-1.0` (H11) ist das die zweite Ursache dafür, dass eine Ergebnistabelle Lücken hat. Für den Beleg: die Clipzahl je Gitterpunkt ist keine Konstante und gehört als Spalte oder Fußnote in die Phase-3-Tabelle. Umfang: Halbsatz |

## H — Phase 4: Adversarial

| # | Mechanismus | Reg. | Kap. | Status | Notiz |
|---|---|---|---|---|---|
| H1 | **FGSM = PGD mit `steps=1`** — eine Implementierung, dadurch vergleichbar | 05 / 07 | 04, 06 | ○ | 02-Skizze fordert die Einführung als Spezialfall; 04 listet beide noch getrennt |
| H2 | **Drei** PGD-Ziele, nicht zwei: ungezielt gegen das **wahre Label** (`adversarial.py`, Training 4.2), ungezielt gegen die **eigene saubere Vorhersage** (`inference.py`, interaktiver Angriff 4.1, braucht kein Ground Truth), **gezielt** auf eine gewählte Klasse (`uap.py`, 4.1) | 05 | 04 | ! | Verwechslung kehrt Interpretation um · **04 §4.1 schreibt dem Angriff die Maximierung gegen das *wahre Label* zu.** Der Sweep-/Frontend-Angriff arbeitet gegen die **eigene saubere Vorhersage**; gegen das wahre Label läuft nur das adversariale Training |
| H3 | ε im **normalisierten** Pixelraum (nicht `[0,255]`) | 05 | 04, 06 | ○ | Ohne diesen Hinweis sind die ε-Werte mit der Literatur nicht vergleichbar · nachgetragen 2026-08-05 in die 04-Skizze §Phase 4.1; für den Kap.-6-Anteil Vorschlag `–` (das ε-Gitter steht in der 05-Skizze, die Raumangabe genügt einmal) |
| H4 | Angriff über den **ganzen** Clip, nicht nur ein Fenster | 07 | 04 | ○ | nachgetragen 2026-08-05 in die 04-Skizze §Phase 4.1 (mit H5, ein Absatz zu den zwei Angriffsgranularitäten) |
| H5 | `_remax_pool` — verhindert Überschätzung des Angriffserfolgs bei Einzelchunk-Angriff | 07 | 04 | ○ | Erklärt, warum Sweep-Fooling-Rates niedriger ausfallen als die Demonstration · nachgetragen 2026-08-05 in die 04-Skizze (gemeinsam mit H4) |
| H6 | **Gemeinsamer** multimodaler PGD (ein Backward hält Cross-Modal-Gradienten konsistent) | 02 / 05 | 04 | ○ | `attack_modalities` genannt; der gemeinsame Rückwärtspass nicht · nachgetragen 2026-08-05 in die 04-Skizze §Phase 4.1 (mit H7, ein Absatz) |
| H7 | Getrennte ε-Budgets für Video und Audio; `attack_modalities`-Schalter | 05 | 04, 06 | ○ | Schalter ✓; getrennte ε-Budgets nur in der 05-Skizze · nachgetragen 2026-08-05 in die 04-Skizze (gemeinsam mit H6) |
| H8 | **UAP** (Moosavi-Dezfooli 2017): eine clipunabhängige Störung, Transfer auf ungesehene Clips | 05 | 04, 06 | ✓ | 04 §4.1 beschreibt UAP samt Zweck |
| H9 | Universeller **Audioschnipsel** wird gekachelt; Gradient über `_fold_audio_grad` zurückgefaltet. **In der Voreinstellung (10.240 Samples = Fensterlänge) ergibt das genau eine Kachel und ist wirkungslos** — erst ein kleineres `--audio-uap-samples` macht δ* periodisch | 05 | 04 | ○ | Als *Möglichkeit* beschreiben, nicht als Eigenschaft der gefahrenen Läufe · nachgetragen 2026-08-05 in die 04-Skizze §Phase 4.1 (UAP-Punkt) |
| H10 | **Fooling Rate** schließt bereits zielkonforme Clips aus | 05 | 06 | ○ | 02 definiert die Sweep-Variante korrekt; die abweichende UAP-Definition fehlt · nachgetragen 2026-08-05 in die 05-Skizze §Evaluationsmetriken (mit H20/H21, ein Absatz). Abweichung von der Kap.-Spalte: Metrikdefinition und -bedingung sind Setup-Aussagen (Begründung wie bei E5) |
| H11 | AUC-Sentinel `-1.0` in W&B-Tabellen bedeutet „nicht bestimmbar", **kein Messwert** | 05 | 06 | ○ | Rohtabellen prüfen · **Fehlerquelle für Ergebnisdiagramme** · nachgetragen 2026-08-05 in die 06-Skizze §Phase 4 (mit H22/H23, ein Absatz zur Tabellenherkunft) |
| H12 | Methode × ε-Gitter mit Wiederaufnahme-Checkpoint | 05 | 06 | ○ | 05-Skizze nennt das Gitter; ~~die Wiederaufnahme fehlt~~ · **nachgetragen 2026-08-06** (Verifikationslauf) in die 05-Skizze §Sweeps, im Absatz mit H24--H26. Abweichung von der Kap.-Spalte (dort 06): die Wiederaufnahme ist eine Laufbedingung. **Anschluss:** sie ist die verlustfreie Alternative zu den Log-Scrapern H22/H23 |
| H13 | **Adversariales Training (4.2)**: 1:1-Mischung, halber Batch durch PGD ersetzt | 02 / 05 | 04, 06 | ✓ | 04 §4.2 beschreibt 1:1-Mischung und Batch-Splitting |
| H14 | Adversariales Finetuning verlangt entfrorenen Backbone (sonst härtet nur der Kopf) | 10 | 04 | ○ | nachgetragen 2026-08-05 in die 04-Skizze §Phase 4.2 · Detail aus [10](10_konfiguration.md): `train_*_adversarial.yaml` erzwingt `freeze_backbone: false` und warmstartet vom sauberen Phase-1-Modell |
| H15 | Angriffsschleife verschmutzt keine Gewichtsgradienten (getestet) | 05 / 09 | 04 | ○ | nachgetragen 2026-08-05 in die 04-Skizze §Phase 4.2 (Halbsatz) und die 09-Skizze §G (Testzeile) |
| H16 | **Keine Ergebnisnotiz zu Phase 4 im Vault** | 12 | 06, 07 | ✓ | ~~Läufe durchgeführt?~~ · **Vom Autor beantwortet (2026-08-06): nein, und es ist auch nicht mehr geplant** — Phase 3 und 4 werden wegen des verkleinerten Projektumfangs nicht weiter bearbeitet (s. Kasten *Umfangsentscheidung*). 06/07/08 führen Phase 4 durchgängig als „implementiert, Ergebnisse ausstehend“ — diese Formulierung bleibt korrekt und ist die einzige zulässige. **Nicht** zu „noch zu untersuchen“ abschwächen und **nicht** zu einem Ergebnis aufwerten |
| H17 | `untargeted_pgd` (Training) klemmt **nicht** auf den Wertebereich der Eingabe, `_pgd_attack` (Angriff) klemmt zusätzlich auf `[x.min(), x.max()]` — beide Implementierungen sind trotz gleicher Schrittweitenheuristik **nicht bitgleich** | 05 / 07 | 04 | ○ | nachgetragen 2026-08-05 in die 04-Skizze §Phase 4.1 (mit H28, ein Absatz zu den beiden PGD-Implementierungen) |
| H18 | **Die UAP-Anpassungsmenge ist eine methodische Entscheidung:** `fit_label` ist stets die Gegenklasse — eine δ*→REAL-Umgehung wird ausschließlich auf **echt gefälschten** Chunks angepasst (auf der Gegenklasse gäbe es keinen Gradienten). Transferauswertung auf einer klassenbalancierten, fake-angereicherten Teilmenge (200 Chunks je Klasse) | 05 | 04, 06 | ○ | Die Anpassungsmenge ist eine methodische Entscheidung, keine Formalität · nachgetragen 2026-08-05 **geteilt**: die Anpassungsmenge (`fit_label` = Gegenklasse) in die 04-Skizze §Phase 4.1 (mit H19), die balancierte Transferstichprobe (200 je Klasse) in die 05-Skizze §Sweeps. Die Begründung „~6 % Fake-Chunks" aus dem Codekommentar ist beim Ausschreiben wegzulassen (Widerspruch A11) |
| H19 | UAP passt δ* auf **HDF5-Trainingschunks** an, nicht auf neu dekodierte MP4-Frames — also exakt auf den Bytes, auf denen trainiert wurde | 05 | 04 | ○ | nachgetragen 2026-08-05 in die 04-Skizze §Phase 4.1 (gemeinsam mit H18) |
| H20 | UAP berichtet `fooling_rate_fake` und `_real` getrennt; die belegrelevante Zahl ist **`fooling_primary`** (Rate auf der *Gegenklasse*), dazu `mean_target_prob_delta` — **nicht** dieselbe Größe wie die Fake-Prob-Differenz der Sweeps | 05 | 06 | ○ | nachgetragen 2026-08-05 in die 05-Skizze §Evaluationsmetriken (mit H10/H21) |
| H21 | **Zwei Fooling Rates, ein Name:** die Sweeps bedingen auf *baselinekorrekt*, die UAP auf *nicht schon in der Zielklasse*. Nicht ineinander überführbar | 05 | 06 | ○ | Dürfen nicht ohne Bedingungsangabe in einer Tabelle stehen · **Zwei verschiedene Größen unter einem Namen** · nachgetragen 2026-08-05 in die 05-Skizze §Evaluationsmetriken (mit H10/H20) |
| H22 | **Drei Log-Scraper** rekonstruieren die W&B-Tabellen aus den Konsolenlogs, weil alle Sweeps ihre Tabelle **genau einmal am Ende** schreiben und ein Abbruch jeden gerechneten Gitterpunkt verlöre. Die Rekonstruktion ist **nicht verlustfrei**: `n_clips` (Phase 4) bzw. `adv_acc_fake`/`_real` (UAP) bleiben leer | 05 | 06, 09 | ○ | Rekonstruierte Tabelle ist nicht gleichwertig zur geloggten · nachgetragen 2026-08-05 in die 06-Skizze §Phase 4 (mit H11/H23, ein Absatz) und als Halbsatz in die 09-Skizze §E |
| H23 | Verlustfreie Alternative für Phase 4: `eval_adversarial_sweep.py --resume-csv` (Wiederaufnahme je Gitterpunkt). Für die UAP-Läufe sind `adv_acc_fake`/`_real` **prinzipiell** nicht rekonstruierbar (berechnet, aber nie ausgegeben — im Code als „KNOWN GAP") | 05 | 06 | ○ | nachgetragen 2026-08-05 in die 06-Skizze §Phase 4 (gemeinsam mit H22/H11) |
| H24 | **Zwei PowerShell-Runbooks:** Volllauf aus **neun unabhängigen Schritten** (1 Robustheits-, 4 Adversarial-, 4 UAP-Läufe), Fehlschlag bricht die Kette nicht ab, PASS/FAIL-Tabelle mit Laufzeiten, Transkript nach `logs/phase34/`; Schätzung **~60 h**. Drei Vorbedingungen werden vorab geprüft | 05 | 05, 09 | ○ | Der Volllauf (~60 h) ist selbst eine Aufwandsangabe für die Arbeit · nachgetragen 2026-08-05 in die 05-Skizze §Sweeps (mit H25/H26, ein Absatz) und als Tabellenzeile in die 09-Skizze §A; verifiziert gegen `scripts/smoke_phase34.ps1:9` und `scripts/run_phase34.ps1:58-60,196-211` |
| H25 | Der Smoke ist **kein verkleinerter Volllauf**, sondern eine Teilmenge: 6 Videos, ein Gitterpunkt, nur FGSM, nur zwei der vier adversarialen Konfigurationen, W&B offline | 05 | 05, 09 | ○ | nachgetragen 2026-08-05 in die 05-Skizze §Sweeps (gemeinsam mit H24) und die 09-Skizze §A |
| H26 | `-ResumeDir` legt **je Konfiguration eine eigene** Resume-CSV an — unimodaler Videolauf und multimodaler `video`-Lauf erzeugen sonst denselben Schlüssel `(method, "video", ε)` und der zweite überspränge die Punkte des ersten | 05 | 05 | ○ | nachgetragen 2026-08-05 in die 05-Skizze §Sweeps (gemeinsam mit H24/H25); Kandidat für den Anhang, falls gekürzt wird |
| H27 | Der **Attention Shift der Sweeps** entsteht anders als der der Oberfläche: Batchfassung = Single-Seed-FAKE-Heatmap über **geometrische Rechtecke**, interaktiv = bivariat über die **Landmark-Partition**; multimodal mittelt die Batchfassung Regionen **und** Bänder gemeinsam | 05 / 07 | 06 | ○ | Keine Werte derselben Größe — nicht gegenüberstellen · Sweep- und Oberflächenwerte dürfen nicht gegeneinandergestellt werden · nachgetragen 2026-08-05 in die 04-Skizze §Attention-Shift (Anschluss an F19/F42). Abweichung von der Kap.-Spalte: die Messgröße wird in Kapitel 4 definiert, Kapitel 6 hat für Phase 4 noch keine Zahlen |
| H28 | ε-Klemmung erfolgt auf `[x.min(), x.max()]` des jeweiligen **sauberen** Tensors, also auf einen **clip-abhängigen** Bereich statt auf einen festen gültigen Bildbereich; die Schrittweitenkonstante 2,5 steht ohne Begründung im Code | 07 | 04 | ○ | nachgetragen 2026-08-05 in die 04-Skizze §Phase 4.1 (gemeinsam mit H17) |

| H29 | **Bei Batchgröße 1 findet kein adversariales Training statt.** `num_adversarial_samples` rechnet `batch_size // 2` und rundet ab; die Mischung ersetzt die erste Hälfte des **Mikro**batches in `training_step`, `accumulate_grad_batches` gleicht das nicht aus. `train_multimodal_adversarial.yaml:22` setzt `batch_size: 1` ⇒ **null** adversariale Samples je Schritt, der Lauf trainiert vollständig auf sauberen Daten | 05 / 10 | 05, 07 | ○ | überführt 2026-08-06 in die 07-Skizze als **bedingter** Punkt (nur aufnehmen, falls Phase 4.2 doch berichtet wird) · **Neu angelegt 2026-08-06 — der schwerwiegendste Fund dieses Durchlaufs.** Zwei belegte Fakten ergeben ihn: `batch_size // 2` mit ausdrücklicher Abrundungsfolge ([05 §adversarial.py](05_robustheit_adversarial.md)) und `batch_size: 1` (verifiziert gegen `configs/experiment/train_multimodal_adversarial.yaml:22`). Die anderen beiden Arme sind unauffällig: Video 2 → 1 von 2, Audio 16 → 8 von 16, je 50 %. **Berührt H13** (`✓`, \enquote{04 §4.2 beschreibt 1:1-Mischung}): die Beschreibung stimmt für Video und Audio, für den multimodalen Arm nicht. **Vom Autor beantwortet (2026-08-06): der multimodale Adversariallauf wurde nicht ausgeführt und wird es voraussichtlich auch nicht** (verkleinerter Projektumfang, s. Kasten *Umfangsentscheidung*). Der Defekt ist damit **latent, nicht wirksam** — er hat keine berichtete Zahl verfälscht. Er bleibt dokumentiert, weil er sonst beim Wiederaufgreifen still zuschlägt: Wird der Lauf je gefahren, ist `batch_size` zu erhöhen, sonst ist das Ergebnis gegenstandslos. Umfang: nur dann ein Satz in den Limitationen; andernfalls entfällt der Punkt mit dem Abschnitt |
| H30 | **Das Trainingsbudget der Phase 4.2 ist mit dem Angriffsbudget der Phase 4.1 vergleichbar:** alle drei Module setzen `adv_epsilon = 0,03` und `adv_steps = 7`, die drei `train_*_adversarial`-Konfigurationen übernehmen dieselben Werte, und die Schrittweite ist überall `ε / steps · 2,5` — dieselbe Heuristik wie in `_pgd_attack` | 05 / 10 | 05 | ○ | **Neu angelegt 2026-08-06** · Die 05-Skizze führt das ε-Gitter der **Angriffe** (0,01--0,1), nennt aber kein Trainingsbudget. Ohne H30 ist unklar, ob die Härtung im selben Budgetbereich stattfindet wie der Angriff, gegen den sie schützen soll — sie tut es (0,03 liegt mitten im Angriffsgitter). Verifiziert gegen `train_*_adversarial.yaml:32-34`. Umfang: Tabellenzeile |
| H31 | **Das UAP-Optimierungsverfahren und seine Datentrennung:** δ\* wird **stochastisch je Chunk** angepasst — jeder einzelne Chunk liefert einen gezielten Abstiegsschritt auf das gemeinsame δ, das nach *jeder* Aktualisierung neu auf die ε-Kugel projiziert wird; `epochs` Durchläufe mit je neu gemischter Chunkreihenfolge (`seed`). Voreinstellungen: ε 0,03, Schrittweite ε/10, 5 Epochen. **Angepasst wird auf `train_metadata.csv`, ausgewertet auf `test_metadata.csv`** | 05 | 04, 05 | ○ | **Neu angelegt 2026-08-06** · H18/H19 beschreiben *worauf* angepasst wird, nicht *wie*. Die Split-Trennung Fit/Transfer ist die Bedingung, unter der \enquote{Transfer auf ungesehene Clips} (H8) überhaupt eine Aussage ist — ohne sie wäre die Transferzahl zirkulär. Umfang: 1 Satz Methodik + Tabellenzeilen im Setup |
| H32 | **δ\* wird als zeigbares Artefakt gespeichert:** `.pt` mit Tensoren **und** Metadaten (Modalität, Zielklasse, ε, Epochenzahl, Fit-Label, Chunkzahl, **gemessenes L∞ gegen das Budget**) plus eine PNG-Visualisierung des Videoanteils — Mittel über Frames und Kanäle, `seismic`, symmetrisch um 0 skaliert; Dateiname `uap_<modality>_<target>_eps<ε>` | 05 | 06, 09 | ○ | **Neu angelegt 2026-08-06** · Für eine Phase, deren Ergebnisse ausstehen, ist das eine **herstellbare Abbildung**: eine universelle Störung sichtbar zu machen ist anschaulicher als jede Fooling-Rate-Tabelle. Das gemessene L∞ ist zugleich die Selbstkontrolle, dass die Projektion gehalten hat. Umfang: Abbildung + Halbsatz |
| H33 | **Paarweise Auswertung als Vergleichbarkeitsgarantie:** `evaluate_*_uap` benutzt für sauber und gestört **einen** Codepfad (`delta=None` liefert die Baseline), sodass beide Seiten garantiert dieselbe Vorverarbeitung sehen; `_evaluate_transfer` wertet beide in einem Durchgang aus und verwirft einen Chunk aus **allen** Ergebnislisten, sobald eine der beiden Vorhersagen scheitert — die Indizes bleiben ausgerichtet | 05 | 04 | ○ | **Neu angelegt 2026-08-06** · Dieselbe methodische Klasse wie S17 (zusätzlicher sauberer Durchlauf vor dem Angriff): der Vorher-Nachher-Vergleich ist nur gleichartig, wenn beide Seiten denselben Pfad nehmen. Gehört als Halbsatz an die UAP-Beschreibung. Umfang: Halbsatz |
| H34 | **Im voreingestellten Adversarialsweep sind die ε-Budgets *nicht* getrennt:** ohne `--audio-epsilon` spiegelt das Audiobudget je Gitterpunkt den Video-ε-Wert | 05 | 05, 06 | ○ | **Neu angelegt 2026-08-06** · H7 führt die getrennten Budgets als Fähigkeit. Im gefahrenen Sweep sind sie in der Voreinstellung gekoppelt — eine Aussage \enquote{Audio ist bei gleichem ε anfälliger} ist damit möglich, eine über *unabhängig gewählte* Budgets nicht. Umfang: Halbsatz an die Sweep-Tabelle |
| H35 | Sieben Kleinmechanismen der Phase-3/4-Werkzeuge: `_project_linf` klemmt δ selbst statt der Differenz zu einem Original (δ *ist* die Störung); `compute_multimodal_uap` wirft `ValueError`, wenn der Audioschnipsel länger als das Modellfenster ist; `VIDEOMAE_CKPT_PATH` ist auch im multimodalen UAP-Lauf Pflicht; der saubere Baselinedurchgang des Robustheitssweeps entfällt vollständig, wenn nur der multimodale Arm läuft (er baut seine eigene Baseline); fehlende MP4s werden gezählt und mit Verweis auf `backfill_normalized.py` gemeldet; die Scraper-Regexe verankern auf ASCII statt auf `Δ`/`ε`/`δ`/`L∞` (cp1252-Konsole) und trennen an `\r` **und** `\n` (tqdm-Fortschrittsbalken), die Musterreihenfolge ist bindend; `upload_wandb` darf nur auf einen **gestoppten** Lauf nachladen | 05 | — | – | **Neu angelegt 2026-08-06**, zugleich **bewusst weggelassen**: Werkzeug- und Bedienhygiene ohne Bezug zu einer Forschungsfrage. Sammelzeile, damit die Prüfung dokumentiert ist |

## S — Systemdemonstrator

| # | Mechanismus | Reg. | Kap. | Status | Notiz |
|---|---|---|---|---|---|
| S1 | FastAPI-Backend, fünf Router mit sieben Routen, Modelle als Lazy Singletons | 06 | 04 | ○ | 08-Skizze nennt den Demonstrator als Nebenprodukt, 09-Skizze F das Systemdiagramm |
| S2 | Nicht-blockierendes Modell-Preload; Server sofort ansprechbar | 06 | 04 | ○ | nachgetragen 2026-08-05 in die 09-Skizze §F (Beschriftung des Systemdiagramms, mit S1). Abweichung von der Kap.-Spalte: Eigenschaft des Systemaufbaus, keine Designbegründung. Verifiziert: `src/api/app.py:53-73` lädt **nur** VideoMAE und Wav2Vec2, das multimodale Modell erst bei der ersten Anfrage |
| S3 | Checkpoint-Auswahl über Umgebungsvariablen; fehlender Checkpoint → HTTP 503 | 06 | 04, 09 | ○ | nachgetragen 2026-08-05 in die 05-Skizze §Laufzeitkonfiguration (mit S22, ein Absatz: 503, stilles `audio: null`, Zuordnung Fusionsmodus) und die 09-Skizze §G (Variablentabelle). Abweichung von der Kap.-Spalte: Laufbedingung, keine Designbegründung — wie D20/D29 |
| S4 | Cache-Schlüssel kodiert **jeden** einstellbaren Parameter | 06 | 04 | ○ | nachgetragen 2026-08-05 in die 05-Skizze §Laufzeitkonfiguration (mit S13/S14/S20, ein Absatz) |
| S5 | 20 Pydantic-Schemas als API-Vertrag; TS-Gegenstück **manuell** synchron gehalten | 06 / 08 | 04 | ○ | nachgetragen 2026-08-05 in die 09-Skizze §G; Kandidat für `–`, falls der Anhang gekürzt wird (trifft keine Forschungsfrage) |
| S6 | Kein Upload-Pfad — Clips kommen aus der Registry | 06 | 04 | ○ | Namen sind irreführend · nachgetragen 2026-08-05 in die 04-Skizze (neuer §Demonstrator, gemeinsam mit S7) |
| S7 | Vorschaubild aus dem HDF5 (zeigt, was das Modell sieht) | 06 | 04 | ○ | nachgetragen 2026-08-05 in die 04-Skizze (neuer §Demonstrator, gemeinsam mit S6, ein Absatz) |
| S8 | React-Oberfläche mit vierstufiger Clipauswahl, Heatmap-Overlay, Chunk-Zeitleisten, Gesichtsschema, drei Audioschichten | 08 | 04 | ○ | 04-Skizze Punkt 3 listet V1–V10 mit Beschreibungsauftrag |
| S9 | **Erklärsystem** mit 15 Inhaltsmodulen und wiederverwendbaren Widgets | 08 | 04 | ○ | Eigenständiges Ergebnis · nachgetragen 2026-08-05 in die 04-Skizze (§Demonstrator, gemeinsam mit S33 als ein Absatz) |
| S10 | Synchronisierter Doppelspieler für Vorher/Nachher | 08 | 04 | ○ | 04-Skizze nennt den Crop-Vergleichsplayer |
| S11 | Mock-Modus (`VITE_USE_MOCK`) — **Screenshots müssen aus dem echten Backend stammen**. Der Mock deckt **nur** Clipliste und Phase 3/4 ab; `analyzeClip()` hat **keinen** Mock-Pfad, die Hauptanalyse braucht immer ein laufendes Backend | 08 | 05 | ○ | „Ohne Backend vorführbar" trifft nicht zu · **Screenshot-Fehlerquelle** · nachgetragen 2026-08-05 in die 05-Skizze (§Laufzeitkonfiguration, an die Erzeugungsvorschrift S4/S13/S14/S20). Die Spalte *Kap.* war leer; Zuordnung zu Kapitel 5 mit derselben Begründung wie F24/F31 (Laufbedingung der Abbildungserzeugung) |
| S12 | Kein Frontend-Test vorhanden | 09 | 07, 09 | ○ | Limitation · überführt 2026-08-06 in die 07-Skizze (gemeinsam mit Q21, ein Satz) |
| S13 | **Der Analysecache wird nie invalidiert** — der Schlüssel kodiert Clip und Parameter, **nicht das Modell**. Nach einem Checkpointwechsel liefert derselbe Clip weiter das alte Ergebnis; `data/analysis_cache/` muss von Hand geleert werden | 06 | 04, 07 | ○ | Fehlerquelle für Abbildungen · nachgetragen 2026-08-05 in die 05-Skizze §Laufzeitkonfiguration (mit S4/S14/S20) und die 07-Skizze §Limitationen (Punkt 21). Abweichung von der Kap.-Spalte: die Vorschrift dagegen ist eine Laufbedingung, Kapitel 4 bekommt keinen eigenen Eintrag |
| S14 | Auch die Registry- und CSV-Modulcaches werden nie invalidiert: Änderungen an `clips.json` oder den Metadaten-CSVs wirken erst nach einem **Serverneustart** | 06 | 04 | ○ | nachgetragen 2026-08-05 in die 05-Skizze §Laufzeitkonfiguration (mit S4/S13/S20) |
| S15 | Fehlende `crop_*`/`orig_*`-Spalten in Alt-CSVs ergeben **lautlos** eine Vollbild-Box `(0,0,224,224)` — die Rückprojektion wird zur Identität, die Heatmap *sieht* korrekt aus, sitzt aber an der falschen Stelle | 06 | 04, 07 | ○ | Passiert ohne jede Meldung · nachgetragen 2026-08-05 in die 04-Skizze (Bedingung der Reprojektion, Gliederung 4.5.3) und die 07-Skizze §Limitationen (Punkt 22, verwandt mit F43) |
| S16 | Die „Ganzclip"-Analyse deckt nur die Fenster ab, in denen ein Gesicht gefunden wurde — **die Chunkfolge darf Lücken haben**, nicht zwingend die volle Cliplaufzeit | 06 | 04 | ○ | Im Beleg vorsichtig formulieren · nachgetragen 2026-08-05 in die 04-Skizze (neuer §Demonstrator) |
| S17 | Phase 4 fährt **vor** dem Angriff einen zusätzlichen sauberen Durchlauf **desselben** Modells als Baseline — nur so ist der Vorher-Nachher-Vergleich gleichartig; die „CLEAN"-Seite kann deshalb von der Hauptanzeige abweichen | 06 / 08 | 04, 06 | ○ | Preis: doppelte Inferenz je Angriff · nachgetragen 2026-08-05 in die 04-Skizze §Phase 4.1 (Verfahren) und die 06-Skizze §Phase 4 (Bildunterschriftshinweis; entfällt mit dem Abschnitt, falls Phase 4 gestrichen wird) |
| S18 | **Urteil und Konfidenz sind getrennte Felder — mit Absicht.** Alle Konfidenzen sind *richtungslos* (immer ≥ 0,5: Konfidenz **in** dem jeweiligen Urteil). Aus gestiegener Konfidenz folgt **nicht** „stärker FAKE", und ein Urteilsumschlag ist aus ihr unsichtbar; die Codekommentare weisen an, das Urteil nie zurückzurechnen | 06 / 08 | 04, 06 | ○ | Gehört in jede Abbildungslegende · Betrifft die Lesart **jeder** Phase-3/4-Abbildung · nachgetragen 2026-08-05 in die 04-Skizze (neuer §Demonstrator, Definition, Anschluss an S19) und die 06-Skizze (Lesehinweis, gemeinsam mit G14) |
| S19 | `perChunkConfidence` ist die **rohe** Fake-Wahrscheinlichkeit je Fenster, das Urteil dagegen max-gepoolt — eine hohe Anzeige bei überwiegend realer Kurve ist kein Widerspruch, sondern die Aggregationsregel | 06 / 08 | 04, 06 | ○ | 04-Skizze verlangt genau diese Begründung (V2 vs. V5) |
| S20 | Abwärtskompatible Vorgabewerte (`[]`, `0.0`, `False`, `None`) halten alte Cachedateien gültig — **leere Listen sind als „altes Ergebnis" zu lesen, nicht als „gemessene Null"** | 06 | 04 | ○ | nachgetragen 2026-08-05 in die 05-Skizze §Laufzeitkonfiguration (mit S4/S13/S14) |
| S21 | Die Parametergrenzen der Labore sind schema-erzwungen (CRF 18–51, fps 5–30, σ 0–50, AAC 8–320 kbps, 0 < ε ≤ 0,5, Schritte 1–100, `fusion_mode` ∈ {cross_attention, concat}); Verletzung ⇒ HTTP 422, die Anfrage erreicht die Inferenz nie | 06 / 08 | 04 | ○ | nachgetragen 2026-08-05 in die 05-Skizze §Laufzeitkonfiguration; Abweichung von der Kap.-Spalte wie bei H10/H20/H21. **Nicht identisch mit dem Offline-Gitter** — die Oberfläche lässt Zwischenwerte zu |
| S22 | **Der Fusionsmodus wird geprüft, aber nicht erzwungen**: ein Checkpoint mit abweichendem `fusion_mode` wird nach einer bloßen Logwarnung trotzdem benutzt | 06 / 07 | 04 | ○ | Zuordnung Modus ↔ Umgebungsvariable prüfen · nachgetragen 2026-08-05 in die 05-Skizze §Laufzeitkonfiguration (mit S3). Verifiziert: `src/api/inference.py:184-194` loggt nur eine Warnung und cacht das Modell trotzdem. **Anschluss: S23 (Reg. 07) gehört in denselben Absatz — am 2026-08-05 als Punkt (d) desselben Absatzes nachgetragen** |
| S23 | **Nur `run_multimodal_inference` reicht `fusion_mode` durch.** Alle Phase-4-Ergebnisse und alle multimodalen Sweep-Werte gelten für **`cross_attention`** — unabhängig davon, was im Frontend umgeschaltet ist; einen Concat-Vergleich gibt es dort nicht | 07 | 04, 06 | ○ | **Alle Phase-4-Zahlen gelten für `cross_attention`** — ohne diesen Satz droht eine Fehlzuordnung · nachgetragen 2026-08-05 **dreigeteilt**: Reichweite der Phasen 3/4 in die 04-Skizze (§Methodik zur Evaluierung der Robustheit, Vorspann), Laufbedingung in die 05-Skizze (§Laufzeitkonfiguration, Punkt (d) des Absatzes S3+S22), Gültigkeitsangabe zu den Zahlen in die 06-Skizze (§Phase 3 und §Phase 4). Verifiziert: `src/api/inference.py:151` (Vorgabewert `cross_attention`), `:2420` einzige Weitergabe, `:3343/:3615/:3673` Aufruf ohne Argument; `eval_robustness_sweep.py:569,628`, `eval_adversarial_sweep.py:389,453`, `compute_uap.py:388`. **Abgrenzung:** die interaktive Robustheitsroute reicht den Modus durch (`src/api/routers/robustness.py:57`) — die feste Verdrahtung dort ist das Frontend-Gegenstück **S24** (Reg. 08) und weiterhin offen |
| S24 | Im Robustheitslabor ist der Fusionsmodus **fest auf `cross_attention`** verdrahtet; eigenständiger Wav2Vec-Audiotest und Multimodalmodus schließen sich gegenseitig aus | 08 | 05 | ○ | nachgetragen 2026-08-05 in die 05-Skizze als Punkt (e) des Absatzes S3+S22+S23 — damit ist die Abgrenzung aus S23 eingelöst: auch die interaktive Route läuft praktisch auf `cross_attention`. Abweichung von der Kap.-Spalte wie bei S3/S22/S23. Verifiziert: `RobustnessPanel.tsx:362`, `:490-498` |
| S25 | **Die Achse der Shift-Tabelle ist nicht REAL ↔ FAKE**, sondern „weniger ↔ mehr Aufmerksamkeit"; die Urteilsrichtung steckt allein in der Farbe. `MAG_FULL_SCALE`/`DIR_FULL_SCALE` sind **feste** Skalen („do not derive them from the data") ⇒ Balkenlängen sind über Läufe hinweg vergleichbar | 08 | 04, 06 | ○ | Die Tabelle ist als Visual in der 04-Skizze genannt, ihre Achsensemantik nicht · nachgetragen 2026-08-05 in die 04-Skizze (§Attention-Shift, Anschluss an F19/F42/H27) und als Bildunterschriftshinweis in die 06-Skizze §Phase 3 (mit G8/S18) · **Ergänzung 2026-08-06:** die Zeilen sind nach `\|ΔMagnitude\|` **absteigend sortiert** (die Reihenfolge im Bild ist also keine anatomische), und jede Zeile trägt **zwei** Punkte — Mitte = vorher, Spitze = nachher. Beides beim Ausschreiben der Bildunterschrift mitnehmen |
| S26 | `emphasizeRelevance` (`\|v\|^2,5 × 1,8`) ist **keine bloße Darstellungsverstärkung**, sondern unterdrückt schwache Evidenz bis zur Unsichtbarkeit (Rauschband 0,20–0,25 → ~0,03) — **die Abwesenheit eines L2-Balkens ist kein Freispruch**. Wirkt nur in der Relevance-Ansicht; der Auslesetext zeigt den transformierten Wert | 08 | 04, 07 | ○ | nachgetragen 2026-08-05 in die 04-Skizze (§Akustische xAI, gemeinsam mit S27 als ein Absatz) und in die 07-Skizze §Limitationen (Punkt 23) |
| S27 | `boostMagnitude` wirkt **nur** in der Confidence-Ansicht (Farbboden 0,55); in der Relevance-Ansicht bewusst nicht — dort wäre das Anheben „the same lie the backend sum=1 normalisation made" | 08 | 04 | ○ | nachgetragen 2026-08-05 in die 04-Skizze (§Akustische xAI, gemeinsam mit S26 als ein Absatz — die beiden Verstärkungen wirken gegenläufig und in verschiedenen Ansichten) |
| S28 | Frontend und Backend nutzen **verschiedene Farbrampen**: durchgängig die aufgehellte F2-Variante (`relevanceToRgb`) gegen matplotlibs seismic im Backend-PNG. Geteilt ist nur die **Kodierungslogik** (Alpha aus Magnitude, Farbton aus Richtung), nicht die Farbwerte | 08 | 04 | ○ | Canvas- und PNG-Abbildungen zeigen dieselben Daten in anderen Tönen · nachgetragen 2026-08-05 in die 04-Skizze (§Bivariate Relevanz-Heatmap, gemeinsam mit S35 als ein Absatz); die Regel „je Abbildung den Erzeugungsweg nennen" schließt an F26/F27/F28 in der 06-Skizze an |
| S29 | Das Gesichtsschema zeichnet **sechs von sieben** Regionen — `Chin` wird vom Backend geliefert, hat aber keine Fläche im Schema, zählt jedoch in `totalMag`; die angezeigten Prozente summieren sich sichtbar nicht auf 100 % | 08 | 04 | ○ | Fällt „MOST ATTENDED" auf `Chin`, ist die genannte Region nicht im Bild · nachgetragen 2026-08-05 in die 04-Skizze (§Demonstrator, Punkt V4); die Bildunterschriftsfolge alternativ in Anhang F. Verifiziert: `FaceSchematic.tsx:36` und `:112` |
| S30 | Die Urteilstafel hat **drei** Anzeigeformen (multimodal / unimodal mit Tonspur / ohne Tonspur) — welche entsteht, hängt vom Modus ab | 08 | 04 | ○ | Gehört in jede Bildunterschrift · 04-Skizze V5 nennt die Verdict-Gauges |
| S31 | **Die zeitliche Auflösung der Oberfläche ist überall gröber als die Datenbasis**: L1 bündelt auf 0,64-s-Fenster, das Heatmap-Overlay folgt `timeupdate` (≈ 4 Hz), die Crop-Doppelspieler drosseln den Bildwechsel auf 250 ms | 08 | 04, 07 | ○ | Lokalisierungsaussagen nicht feiner formulieren · nachgetragen 2026-08-05 in die 04-Skizze (§Demonstrator) und in die 07-Skizze als **Erweiterung von Limitation 13** statt als neuer Punkt; nicht mit der datenseitigen Grenze F30 (~20 ms) vermengen |
| S32 | Die drei Standardbausteine des Erklärsystems tragen die methodischen Kernaussagen und sind **zitierfähig**: `BivariateLrpNote` (Begründung des Dual-Seeds), `DeadzoneNote` (der real-Pol bleibt schwach — Relevanz nahe 0 ist kein Real-Beweis; L1/L2 setzen eine Dead-Zone, L3-Magnitude bewusst nicht), `RelevanceScaleNote` (Relevanz ist relativ, **kein Prozentwert**, nur innerhalb desselben Visuals vergleichbar) | 08 | 04 | ○ | Letzteres in jede Abbildungslegende · nachgetragen 2026-08-05 in die 04-Skizze (§Demonstrator, eigener Absatz) und als Halbsatz in die 06-Skizze §sec:results-xai. Die Dead-Zone-Aussage ist eine **Entwurfsbegründung**, kein gemessener Befund — beim Ausschreiben so kennzeichnen |
| S33 | Struktur des Erklärsystems: **15** erklärbare Visualisierungen (alle belegt, keine Lücke), 14 Abschnittsarten mit kanonischer Reihenfolge, 13 Widgets — und der Typ `ConfidenceRelevance` an **jeder** Visualisierung (6× relevance, 5× both, 3× confidence, 1× neither) | 08 | 04 | ○ | Konsequente Umsetzung von F17 · nachgetragen 2026-08-05 in die 04-Skizze (§Demonstrator, gemeinsam mit S9 als ein Absatz) |
| S34 | **Nicht jede Zusicherung des Backends erreicht das Bild:** `anomalyRegions` und `differenceFrames` existieren im Schema, werden aber von keiner Komponente gezeichnet; `FrameTimeline.tsx` ist toter Code (nirgends importiert) | 08 | 04, **06** | ○ | Wer vom Schema auf die Oberfläche schließt, beschreibt Ansichten, die es nicht gibt · nachgetragen 2026-08-05 in die 04-Skizze (§Demonstrator, abschließende Reichweitenangabe) und in die 06-Skizze §Phase 4 (Korrektur der Abbildungsplanung: das geplante Frame-Triptychon verlangt eine Differenzkarte, die die Oberfläche nicht zeichnet — offline erzeugen oder auf Clean/Adversarial verkürzen). **Der Kap.-7-Anteil ist auf `–` gesetzt** (Autorenentscheidung 2026-08-05): als Limitation wäre er eine Doppelung des Kap.-4-Satzes und betrifft keine berichtete Zahl |
| S35 | Das Gesichtsschema weicht mit `FILL_OPTS` in **allen fünf** Gamma-/Gain-Parametern vom Backend-Rendering ab (großflächige Regionen statt Pixel) — Schema und Pixel-Heatmap sind nicht farbgleich | 08 | 04 | ○ | nachgetragen 2026-08-05 in die 04-Skizze (§Bivariate Relevanz-Heatmap, gemeinsam mit S28 als ein Absatz). Verifiziert: `FaceSchematic.tsx:71` |
| S36 | **Die Vergleichsspieler werden vom Backend beliefert:** `_encode_crop_video` kehrt die ImageNet-Normalisierung um (`x·std + mean` → uint8) und kodiert die Crop-Frames über eine FFmpeg-Pipe als H.264/yuv420p. Das **saubere** Video hängt nicht von den Degradations-/Angriffsparametern ab und wird über `reuse_existing` je Clip-Stamm genau einmal kodiert, danach für alle Parametersätze und beide Phasen wiederverwendet | 07 | 04 | ○ | überführt 2026-08-06 in die 04-Skizze (gemeinsam mit S38, ein Absatz) · S10 beschreibt nur die Frontend-Komponente; woher deren Bildmaterial stammt, fehlt · **Zeile neu angelegt 2026-08-05** beim vollständigen Lesen von [07 §3](07_inference_pipeline.md) — der Mechanismus war im Register beschrieben, hatte aber keine Matrixzeile und damit keinen Status. Umfangvorschlag: Halbsatz, alternativ Anhang F (Screenshots je Panel) |
| S37 | **Die Laufzeitpipeline schreibt auf die Platte:** die MP4-Crop-Videos der Phase-3/4-Spieler liegen unter `data/phase_media/`, umlenkbar über `PHASE_MEDIA_DIR`. Der Dateiname der Degradations- bzw. Angriffsvideos ist an den Cacheschlüssel gebunden (`media_prefix`), der des sauberen Videos an den Clip-Stamm; beide Ausgaben überdauern den Prozess | 07 / 06 | 05 | ○ | überführt 2026-08-06 in die 05-Skizze (§Laufzeitkonfiguration, an den Cacheabsatz) · Die zweite Plattenausgabe (WhisperX-Transkriptcache unter `.whisperx_cache/`) ist bereits über F14 („Plattencache") abgedeckt · gehört in denselben Absatz wie S4/S13/S14/S20 (Cacheverhalten, 05-Skizze §Laufzeitkonfiguration) · **Zeile neu angelegt 2026-08-05** aus [07 §Was `inference.py` nicht tut](07_inference_pipeline.md) und [06 §phase\_media.py](06_backend_api.md) |
| S38 | **Dichte Box-Zuordnung je Fenster:** die Ganzclip-Analyse gibt jedem zusammenhängenden 16-Frame-Fenster seine eigene Bounding Box (`_resolve_per_window_boxes`, `_load_all_frames_cropped_per_window`); gesichtslose Fenster erben die vorherige Box, Lücken am Anfang nutzen die Rückfallbox. Der Crop folgt damit der Kopfbewegung, statt eine Anfangsbox über Minuten festzuhalten | 07 | 04 | ○ | überführt 2026-08-06 in die 04-Skizze (gemeinsam mit S36, ein Absatz) · Voraussetzung jeder zeitlichen Aussage des Demonstrators; schließt an S16 (die Chunkfolge darf Lücken haben) und F21 (Rückprojektion) an · **Zeile neu angelegt 2026-08-05** aus [07 §6](07_inference_pipeline.md). Umfangvorschlag: Halbsatz im Demonstrator-Abschnitt |

| S39 | **Die Statusanzeige der Kopfzeile ist der Provenienznachweis eines Screenshots.** `useBackendHealth` pollt `/api/health` alle 15\,s (Zeitlimit 5\,s je Anfrage) und meldet `online`/`offline`/`pending`; bei `VITE_USE_MOCK=true` wird **gar nicht gepollt**, der Status bleibt statisch `mock` | 08 | 05 | ○ | **Neu angelegt 2026-08-06** · Die 05-Skizze fordert unter S11 bereits \enquote{Mock-Modus aus} als Teil der Erzeugungsvorschrift. Dies ist die Stelle im Bild, an der sich die Einhaltung **nachprüfen** lässt — jeder Screenshot trägt seinen eigenen Herkunftsnachweis in der Kopfzeile. Umfang: Halbsatz an den S11-Punkt |
| S40 | **Die Deckkraft des Heatmap-Overlays ist ein Bedienparameter, kein fester Wert:** Regler 0--1 in Schritten von 0,05, Startwert 0,85. Das Overlay ist trotz des Komponentennamens kein `<canvas>`, sondern ein `<img>` mit `objectFit: contain`; `mixBlendMode` steht bewusst auf `'normal'` statt `'screen'`, weil letzteres auf dunklem Videomaterial auswäscht | 08 | 05, 06 | ○ | **Neu angelegt 2026-08-06** · Jede Overlay-Abbildung im Beleg zeigt eine **gewählte** Deckkraft; ohne die Angabe ist der visuelle Eindruck zweier Abbildungen nicht vergleichbar. Gehört zu F23 (Darstellungsverstärkungen) und S28/S35/F53 (Farbwege). Umfang: Halbsatz |
| S41 | **Die Relevanzzeitreihe sättigt ab 0,25.** Die Balkenhöhe ist `min(1, m × 4)` (`RELEVANCE_DISPLAY_GAIN = 4`); alle Magnituden darüber sind visuell identisch, der Tooltip gibt das als \enquote{% of scale} aus. Zusätzlich dürfen die **beiden Zeitreihen unterschiedlich lang** sein — `ConfidenceChart` zählt Forward-Pass-Chunks, `RelevanceChart` Heatmap-Fenster — und werden trotzdem unabhängig auf dieselbe Breite gezeichnet | 08 | 04, 06 | ○ | **Neu angelegt 2026-08-06** · Zwei getrennte Fehlerquellen in einer Abbildung: oberhalb 0,25 zeigt die Kurve keine Unterschiede mehr, und eine senkrechte Position der oberen Kurve entspricht **nicht** derselben der unteren. Der zweite Punkt steht in der Fehlerquellentabelle unter S31, aber in keiner Zeile und in keiner Skizze (die 04-Skizze zu S31 nennt nur die drei Zeitraster). Umfang: 1 Satz |
| S42 | **Der Vergleichsspieler zeigt eine andere Größe als die Hauptansicht:** das 224er-Gesichts-Crop mit der Heatmap **im Crop-Raum**, nicht die auf die Originalauflösung rückprojizierte Karte (F21). Ein gemeinsamer Regler blendet beide Videospuren aus (0\,\% = nur Heatmaps); bei eingeschalteter Regionsüberlagerung liefert nur die **saubere linke** Seite ein Bild, die rechte bleibt leer | 08 | 04, 06 | ○ | **Neu angelegt 2026-08-06** · S10 und S36 beschreiben den Doppelspieler und seine Datenquelle, nicht den Bildraum. Für den Beleg heißt das: Heatmaps aus Hauptansicht und Phase-3/4-Panel sind **nicht** dieselbe Darstellung desselben Clips. Die leere rechte Seite ist Absicht, kein fehlgeschlagener Lauf. Umfang: 1 Satz |
| S43 | **Die beiden L3-Ansichten sind verschieden stark verstärkt:** `confCell` zeichnet mit `dirGain 1,4`, `relCell` mit `dirGain 4` — ausdrücklich, weil das Gradientensignal schwächer ist. Farbintensität ist zwischen den Ansichten damit nicht vergleichbar. Der `ViewToggle` schaltet außerdem **alle drei Audioschichten gleichzeitig** um; eine Abbildung mit L1 in Relevance und L3 in Confidence ist nicht herstellbar | 08 | 04, 06 | ○ | **Neu angelegt 2026-08-06** · Macht F37 (\enquote{das Relevanz-Gitter ist ehrlich blass}) erst lesbar: es ist blass **trotz** dreifach höherer Verstärkung. Ergänzt S26/S27 um die dritte ansichtsabhängige Verstärkung. Umfang: Halbsatz |
| S44 | **Das Regionspanel verschwindet lautlos:** Lasche und `FaceSchematic` werden vollständig ausgeblendet, wenn `regionRelevance` leer ist — also beim gesichtslosen Rückfall oder bei einem älteren Cache | 08 | 07 | ○ | **Neu angelegt 2026-08-06** · Ein Screenshot **ohne** Gesichtsschema belegt nicht, dass keine Regionen berechnet wurden. Dieselbe Klasse wie F43 (unmarkierte Rückfälle) und S20 (leere Listen sind kein gemessener Nullwert); gehört in die Ergänzungsliste der Limitationen. Umfang: Halbsatz |
| S45 | **Die Reglerbereiche der Labore sind enger als die Schemagrenzen und zwischen den Modalitäten asymmetrisch:** Video-ε 0,001--0,1 gegen Audio-ε 0,01--0,5 — das **Audiobudget ist fünfmal so weit**. Startwerte: CRF 28, 25\,fps, σ 0, AAC 64\,kbps, ε 0,03, PGD-Schrittzahl 20. Upscale ist **kein Regler**, sondern ein Kontrollkästchen (640×360 → 1280×720) | 08 | 05 | ○ | **Neu angelegt 2026-08-06** · S21 nennt die **schema**-erzwungenen Grenzen (u. a. `0 < ε ≤ 0,5`); die Oberfläche bietet für Video nur ein Fünftel davon an. Die Asymmetrie der ε-Budgets berührt H7 (getrennte Budgets) und ist bei jedem Vergleich \enquote{Video gegen Audio} zu nennen. Umfang: Tabellenzeilen an den S21-Absatz |
| S46 | **Der Demonstrator läuft nicht ohne den Vite-Dev-Server:** ein `VITE_API_URL` existiert weder in den `.env`-Dateien noch im Quelltext, `api/client.ts` ruft ausschließlich relative Pfade auf, und `/api`, `/clips`, `/media` werden in `vite.config.ts` fest auf `http://localhost:8000` geproxyt. Stack der Oberfläche: React 19, Vite 8, TypeScript 6, Recharts 3, Tailwind 4, framer-motion 12 | 08 | 05, 09 | ○ | **Neu angelegt 2026-08-06** · Reproduktionsbedingung, die neben Q9 (Pfadwiderspruch im Container) gehört: der dokumentierte Weg zur Inbetriebnahme ist der **lokale** Start mit laufendem Dev-Server. Der Stack ergänzt die Software-Tabelle der 05-Skizze, die bislang nur \enquote{React+TS+Vite} führt. Umfang: Halbsatz + Tabellenzeile |
| S47 | Zehn Kleinmechanismen der Oberfläche: `minWidth: 1280` (kein responsives Layout); Rasterkommentar \enquote{60/40} gegen tatsächliches `1fr 0.67fr`; die Phasentabs sind nicht gesperrt, die Sperre sitzt in den Panels; `leanColor` dämpft die Sättigung, damit ein ausgemittelter Lean nicht voll rot erscheint; `useActiveWordIndex` läuft über `requestAnimationFrame` statt `timeupdate`, damit auch kurze Wörter erfasst werden; `VARIANT_ORDER` erzwingt eine feste 2×2-Reihenfolge und `pickRepr` bevorzugt die `real`-Variante für Vorschaubilder; `allRealConf` schreibt einen Hinweistext ins leere Gitter; `wavesurfer.js`, `clsx` und `tailwind-merge` sind eingetragen, aber nirgends importiert; `dist/` ist als Build eingecheckt | 08 | — | – | **Neu angelegt 2026-08-06**, zugleich **bewusst weggelassen**: Bedienungs- und Hygienedetails ohne Bezug zu einer Forschungsfrage. Sammelzeile, damit die Prüfung dokumentiert ist und nicht erneut aufläuft. **Ausnahme mit Belegwert:** der rAF-getriebene Wortindex ist die einzige Ansicht, die *feiner* als das 4-Hz-Raster arbeitet — falls S31 ausgeschrieben wird, gehört der Halbsatz dorthin |

| S48 | **Unimodal und multimodal analysieren dieselbe Clip-ID aus verschiedenen Quellen.** Der unimodale Videopfad liest die vorverarbeiteten Chunks **aus dem HDF5** (exaktes Trainingsformat), der multimodale Pfad dekodiert das **Rohvideo** neu, weil er die Tonspur braucht | 06 | 04, 06 | ○ | **Neu angelegt 2026-08-06** · Zusammen mit F45 (HDF5-Frames über `cv2`, Ganzclip-Frames über PIL/torchvision, **nicht bitgleiche Interpolation**) heißt das: die beiden Modellmodi starten bei demselben Clip nicht von denselben Pixeln. Jeder Vergleich unimodal ↔ multimodal im Demonstrator trägt diesen Unterschied mit. Umfang: 1 Satz |
| S49 | **Gesichtsverlust wird in den beiden Modi gegensätzlich behandelt:** unimodal fällt die Analyse auf eine Vollbildauswertung zurück (`_run_video_inference_fullframe`, laut Docstring außerhalb der Trainingsverteilung), multimodal wirft `run_multimodal_inference` einen `RuntimeError` → **HTTP 500**, ohne jeden Rückfall | 06 | 04, 07 | ○ | **Neu angelegt 2026-08-06** · Ergänzt F43 (Rückfallketten) um die Gegenrichtung: derselbe Clip kann unimodal ein Ergebnis liefern und multimodal komplett scheitern. Für den Demonstrator ist das eine Reichweitengrenze, für den Beleg ein Ehrlichkeitspunkt. Umfang: Halbsatz |
| S50 | **Ein Phase-3-Ergebnis spiegelt seine Parameter unvollständig zurück.** `Phase3ParamsSchema` führt nur `crf`, `fps` und `noiseSigma`; `upscale` wird von der Inferenz mitgeliefert, aber vom Schema **verworfen**, und `audio_bitrate` erscheint nur mittelbar in `audioRobustness.bitrate` | 06 | 05, 06 | ○ | **Neu angelegt 2026-08-06** · Aus einem gespeicherten Ergebnis oder einem Screenshot ist **nicht ablesbar, ob die Skalierungsachse aktiv war**. Dieselbe Klasse wie S39 (Provenienz einer Abbildung); betrifft jede Phase-3-Bildunterschrift. Umfang: Halbsatz |
| S51 | **Die Regionswerte des Demonstrators sind ein Clipmittel.** `RegionRelevanceSchema` liefert je Gesichtsregion einen bivariaten Wert, der **über alle Frames des Clips gemittelt** ist — kein Framewert und kein Vorher-Nachher-Verschub (letzterer ist `AttentionShiftSchema` in Phase 3/4) | 06 | 04, 06 | ○ | überführt 2026-08-06 in die 04-Skizze (§Attention-Shift, an F19/F42) · **Neu angelegt 2026-08-06, Begründung korrigiert am selben Tag.** **Kein Widerspruch zu F25a:** der Clipmittelwert *ist* das gewichtete Mittel aus Fake-Fenstern und Restclip (16,7 % liegt zwischen 17,4 % und 16,5 %) — die beiden Zahlen sind ineinander verschachtelt, nicht konkurrierend. Der Punkt ist ein anderer: **das Regionspanel mittelt genau den zeitlichen Kontrast weg, auf dem der Kernbefund beruht.** Ein Screenshot des Panels kann die Aussage \enquote{an den manipulierten Frames nicht auf die manipulierte Region} weder zeigen noch stützen; sie stammt aus einer frameweisen Offline-Auswertung. In der Bildunterschrift ist deshalb zu sagen, dass die Anteile für den **ganzen Clip** gelten. Umfang: 1 Satz |
| S52 | **Drei stille Degradationen des Demonstrators:** fehlt `conf/clips.json`, gibt es nur eine Warnung und eine **leere Registry** (die Oberfläche zeigt keine Clips, ohne Fehler); fehlt `data/normalized/`, wird der `/clips`-Mount mit Logeintrag übersprungen (die Analyse läuft weiter, die Videos sind nur nicht abspielbar); und der Videopfad wird **fest konstruiert** als `data/normalized/{video_id}.mp4`, weder aus `clips.json` noch über `CLIPS_DIR` konfigurierbar | 06 | 05, 09 | ○ | **Neu angelegt 2026-08-06** · Inbetriebnahmebedingungen, die zu S46 (kein `VITE_API_URL`, fester Proxy) und Q9 (Pfadwiderspruch im Container) gehören: der Demonstrator läuft nur aus dem Projektwurzelverzeichnis mit genau dieser Verzeichnisstruktur. Umfang: Halbsatz + Anhangzeile |
| S53 | **Serialisierung je Router, nicht global:** jeder der drei rechnenden Router hält einen **eigenen** `ThreadPoolExecutor(max_workers=1)`. Analysen serialisieren also untereinander, ebenso Robustheits- und Adversarialläufe — aber ein Analyse-, ein Phase-3- und ein Phase-4-Lauf können sich **gleichzeitig** auf der GPU überlappen | 06 | 05 | ○ | **Neu angelegt 2026-08-06** · Laufbedingung für jede im Demonstrator gemessene Zeitangabe: eine Laufzeitmessung ist nur belastbar, wenn kein zweites Labor parallel rechnet. Betrifft die 8-GB-Karte doppelt. Umfang: Halbsatz |
| S54 | **Die Cachegültigkeit hat zwei Seiten.** S20 beschreibt die eine: abwärtskompatible Vorgabewerte halten alte Dateien gültig. Die andere ist `load_cached` — schlägt die Pydantic-Validierung fehl, wird die Datei **still verworfen** und die Analyse neu gerechnet, ohne Meldung | 06 | 05 | ○ | **Neu angelegt 2026-08-06** · Beide Hälften gehören in denselben Satz, sonst wirkt der Cache entweder zu starr oder zu nachgiebig. Praktische Folge: nach einer Schemaänderung verschwinden Ergebnisse kommentarlos und werden mit dem *aktuellen* Checkpoint neu berechnet — anders als beim nie invalidierten Schlüssel aus S13. Umfang: Halbsatz an den S20-Punkt |
| S55 | Acht Kleinmechanismen des Backends: kein zentraler Exception-Handler (die `503`-Abbildung steht in jedem Router einzeln); Path-Traversal-Guards an drei Stellen (`clips.py`, `analysis_cache.py`, `phase_media.py`, getestet); `_parse_chunk_index` liefert bei nicht parsebarer ID still `0`; `load_clips` filtert die serverseitigen Schlüssel `videoPath` und `h5ChunkId` aus der Antwort; `save_cache` protokolliert Schreibfehler und schluckt sie; `/api/health` antwortet auch während des Ladens mit `status: "ok"`; vier `*_configured`-Flags steuern, welche Modelltoggles die Oberfläche anbietet; Audio wird nur versucht, wenn `hasAudio` in `clips.json` gesetzt ist | 06 | — | – | **Neu angelegt 2026-08-06**, zugleich **bewusst weggelassen**: Implementierungs- und Bedienhygiene ohne Bezug zu einer Forschungsfrage. Sammelzeile, damit die Prüfung dokumentiert ist. **Zwei Ausnahmen mit Belegwert**, falls die Abschnitte ohnehin geschrieben werden: die Path-Traversal-Guards passen als Halbsatz zu A44 (Anhang G), und die `status: "ok"`-Eigenschaft präzisiert S39 — die Kopfzeile belegt, dass *ein Backend* antwortet, nicht dass die Modelle geladen sind |

## I — Reproduzierbarkeit und Qualitätssicherung

| # | Mechanismus | Reg. | Kap. | Status | Notiz |
|---|---|---|---|---|---|
| Q1 | Hydra-Konfiguration, keine Hyperparameter im Code | 10 | 05 | ○ | 05-Skizze §Software-Stack |
| Q2 | Aufgelöste Konfiguration wird je Lauf mitgeschrieben | 03 | 05 | ○ | 05-Skizze §Reproduzierbarkeit |
| Q3 | Deterministische Seeds an **fünf** Stellen: Training (42), Identity-Split (11), Ablationsauswahl (42), Sweep-Stichprobe, **Frame-Perturbation je Chunk** (`seed + idx`). Voller Determinismus zusätzlich optional über `trainer.deterministic=true` (langsamer, Standard `False`) | 03 / 10 | 05 | ○ | 05-Skizze nennt Seed 42 und `split_seed` 11; ~~die übrigen drei Stellen fehlen~~ · **nachgetragen 2026-08-06** (Verifikationslauf) in die 05-Skizze §Reproduzierbarkeit, als Fünferliste mit Fundstellen. **Die drei nachgetragenen tragen die berichteten Zahlen:** Ablationsauswahl, Sweep-Stichprobe (A23) und die Frame-Perturbation je Chunk, an der nach D24 die Reproduzierbarkeit des Spatial-Dominance-Tests hängt |
| Q4 | 336 Tests; 13 davon weisen **methodische Eigenschaften** nach | 09 | 09 | ○ | 09-Skizze G fordert die Zuordnung Test → Silent-Failure-Klasse |
| Q5 | CI: ruff + `pytest -m "not slow"` bei jedem Push | 11 | 09 | ○ | nachgetragen 2026-08-06 in die 09-Skizze §G (ein Absatz mit Q6 und Q11). Verifiziert: `.github/workflows/ci-pipeline.yml:4` (Push und Pull-Request), `:49-50`, `:55` |
| Q6 | Pre-Commit-Hooks | 11 | 09 | ○ | nachgetragen 2026-08-06 in die 09-Skizze §G (mit Q5/Q11). Verifiziert: `.pre-commit-config.yaml:3` (ruff-pre-commit v0.11.0), `:10` (pre-commit-hooks v5.0.0) |
| Q7 | DVC für den Datenbestand | 11 | 05 | ○ | 05-Skizze §Software-Stack |
| Q8 | W&B für Experimentverfolgung; Launch-Queue mit Windows-Anpassung | 11 | 05 | ○ | 05-Skizze nennt W&B und Launch |
| Q9 | Docker + Devcontainer | 11 | 09 | ○ | nachgetragen 2026-08-06 in die 09-Skizze §G als eigener Punkt; der Pfadwiderspruch (`models/` in Compose vs. `checkpoints/` im Projekt) ist als Bedienbedingung mit aufgenommen |
| Q10 | Silent-Failure-Audit dokumentiert (`docs/audit_2026-06.md`) | 12 | 07 | ○ | 09-Skizze G nennt das Audit inkl. der geprüften False Alarms |
| Q11 | **Grenzen der CI-Prüfung:** der Lint-Schritt deckt nur `src/` und `tests/` ab (`scripts/` mit 5.819 Zeilen bleibt ungeprüft); `ruff check` **repariert** wegen `fix = true` und endet mit Rückgabewert **0**, schlägt also nur bei nicht automatisch behebbaren Verstößen fehl; drei verschiedene Ruff-Versionen sind im Umlauf | 11 | 09 | ○ | „Pre-Commit ist grün" ≠ „CI ist grün" · nachgetragen 2026-08-06 in die 09-Skizze §G (trägt den Absatz mit Q5/Q6). Verifiziert: `pyproject.toml:4` (`fix = true`), `ci-pipeline.yml:49` (nur `src/ tests/`), `requirements-dev.txt:1` (ruff ungepinnt) |
| Q12 | **Gemischte Versionsbindung:** Kernbibliotheken exakt gepinnt (`torch==2.11.0`, `transformers==4.57.6`, `lxt==2.1`, `numpy==2.4.4`), Peripherie nur als Mindestversion; `ruff`/`pytest` ungepinnt, `whisperx` zeigt ohne Commit-Angabe auf den Git-Hauptzweig | 11 | 05, 09 | ○ | Reproduzierbarkeitsgrenze · nachgetragen 2026-08-06 in die 05-Skizze §Software-Stack (an die Stack-Tabelle); der Kapitel-9-Anteil ist auf `–` gesetzt (Doppelung derselben Liste). Verifiziert gegen `requirements.txt` und `requirements-dev.txt` |
| Q13 | **WhisperX steht in `requirements-dev.txt`**, nicht in `requirements.txt` — eine Installation nur aus der Laufzeitdatei erzeugt eine Oberfläche **ohne Wort-Zeitleiste und ohne sichtbare Fehlermeldung** (der `ImportError` wird auf `debug` geloggt). Das Docker-Abbild installiert beide Dateien | 11 | 05, 09 | ○ | nachgetragen 2026-08-06 in die 05-Skizze §Laufzeitkonfiguration als **vierte Erzeugungsbedingung** der Abbildungen (neben S3+S22+S23+S24, S4/S13/S14/S20 und S11); der Kapitel-9-Anteil ist auf `–` gesetzt. Verifiziert: `src/api/inference.py:2090-2092` (`log.debug`, Rückgabe `[]`) |
| Q14 | **Der Datenbestand ist versionskennzeichenbar:** `data.dvc` trägt `md5 1a1063a7…dir` über 59.777 Dateien / 10,7 GB — die zitierfähige Kennung des Datenstands | 11 | 04, 05 | ○ | 05-Skizze verknüpft Code-Commit und Datensatz-Hash |
| Q15 | Der W&B-Launch-Shim behebt einen **konkreten** Defekt: wandb baut POSIX-`VAR=value`-Präfixe, die `cmd.exe` nicht auflösen kann (Job scheitert sofort, `WANDB_*` erreichen den Unterprozess nie). Gültig nur bei `max_jobs: 1`; flickt ein wandb-Internum und muss bei Updates nachgezogen werden | 11 | 05, 09 | ○ | nachgetragen 2026-08-06 in die 05-Skizze §Software-Stack (am W&B-Launch-Punkt); `max_jobs: 1` ist zugleich die Erklärung für die serielle Laufreihenfolge. Der Kapitel-9-Anteil ist auf `–` gesetzt |
| Q16 | Versionierte Reproduktionskette vorhanden: `configs/paths` über `DEEPFAKE_DATA_DIR`/`_LOG_DIR`/`_CKPT_DIR` umlenkbar, `PROJECT_ROOT` per `rootutils.setup_root` — alle Pfade unabhängig vom Arbeitsverzeichnis | 10 / 11 | 05, 09 | ○ | nachgetragen 2026-08-05 in die 05-Skizze (§Reproduzierbarkeit, an den D20+D34-Absatz) und die 09-Skizze §A (Variablentabelle); verifiziert gegen `configs/paths/default.yaml:4-19`: drei `oc.env`-Überschreibungen mit Vorgabewert, `root_dir` als einzige ohne |
| Q17 | `torch.set_float32_matmul_precision("medium")` und `add_safe_globals` an **allen sechs** checkpointladenden Einstiegspunkten (sonst scheitert das Entpicklen unter `weights_only=True`) | 03 | 05, 09 | ○ | überführt 2026-08-06 in die 09-Skizze §G (Reproduzierbarkeitsabsatz, mit D42/D49) |
| Q18 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` wird **nur auf Nicht-Windows** gesetzt — in der Windows-Entwicklungsumgebung dieses Projekts wirkungslos | 03 | 05 | ○ | 05-Skizze §Hardware nennt `expandable_segments` als Linux-only |
| Q19 | Ruff-Ausnahme für `src/models/*` (F821, UP037 aus), weil jaxtyping-Achsennamen als nackte Bezeichner in Zeichenketten stehen und das automatische Entfernen der Anführungszeichen `@beartype`-Methoden beim Import zum Absturz brächte — derselbe Grund im Agenten-Hook | 11 | 09 | ○ | nachgetragen 2026-08-06 in die 09-Skizze §G als Halbsatz an den bereits eingetragenen Punkt C12 (Laufzeit-Formprüfung); allein ohne C12 nicht sinnvoll. Verifiziert: `pyproject.toml:19-23` |
| Q20 | **Silent-Failure-Klasse in der Konfiguration:** Callback-Einträge **ohne `_target_` werden kommentarlos übersprungen** — ein Lauf kann ohne Early Stopping oder Checkpointing durchlaufen, ohne dass das Log es anzeigt. Gegenmittel ist der mitgeschriebene `config_tree.log` | 03 | 05, 09 | ○ | überführt 2026-08-06 in die 09-Skizze §G (Silent-Failure-Absatz, an Q10); das Gegenmittel `config_tree.log` entsteht nur, weil `task_wrapper` auch im Fehlerfall durchläuft — s. Q33 |
| Q21 | **Weitere Abdeckungslücken** neben dem fehlenden Frontend-Test (S12): keine End-to-End-Tests über die HTTP-Schicht, keine Trainingskonvergenztests, **keine numerische Prüfung der Renderfunktionen** (`_array_to_data_uri`, `_upproject_heatmap`, `seismicColormap.ts` nur indirekt abgedeckt) | 09 | 07, 09 | ○ | Limitation · überführt 2026-08-06 in die 07-Skizze (gemeinsam mit **S12** als ein Satz) |
| Q22 | **Implementiert ≠ durchgeführt:** 29 Dateien unter `configs/experiment/` = 27 Trainingsexperimente + 1 Diagnoselauf + 1 Template-Rest; Ergebnisnotizen liegen im Vault für **acht** Läufe vor | 10 / 12 | 05 | ○ | **Neu angelegt 2026-08-05** (in 10 §4 und Beobachtung 2 beschrieben, hatte keine Matrixzeile) · nachgetragen in die 05-Skizze (§Ablations- und Diagnostikläufe, letzter Punkt) · verifiziert: `vault/Research/deepfake-detection/Results/` enthält acht Notizen |
| Q24 | **Die Trainingszeiten der Phase-1/2-Läufe stehen in der Baseline-Notiz und füllen einen offenen Platzhalter:** VideoMAE eingefroren 20 Epochen / ~41\,h → `auc_video` 0,730; entfroren 86.228.738 Parameter mit `llrd_decay` 0,75, 12 Epochen / ~30\,h → 0,999 | 12 | 05 | ○ | **Neu angelegt 2026-08-06** · Die 05-Skizze hält dafür ausdrücklich eine Tabelle frei: \enquote{Tabelle beobachteter Trainingszeiten je Modell und Phase (Platzhalter aus dem Archiv-Entwurf noch mit echten Werten füllen)} (`05Experimental_Setup.tex:46-47`). Die Werte liegen vor und sind hier belegt. **Nicht doppelt eintragen:** die 3.074 trainierbaren Parameter des eingefrorenen Laufs stehen bereits in `06Results.tex:39`. Umfang: Tabellenzeilen |
| Q25 | **`0,976` bezeichnet im Projekt zwei verschiedene Größen.** In `wav2vec2-phase1-audio-baseline.md` ist es die **Test-`auc_video` des eingefrorenen Audiomodells** (so im Beleg verwendet: `06Results.tex:43`, `08Conclusion.tex:23`, `00Abstract.tex:24`). In [12 §3.2](12_dokumentation_vault.md) bezeichnet dieselbe Zahl die **Val-AUC im Dateinamen des multimodalen Fusions-Checkpoints** (dort neben Concat 0,963). Ebenso getrennt zu halten: Test-`auc_video` der Fusion **0,960** gegen die Val-AUC **0,976** desselben Modells | 12 | 05, 06 | ○ | **Neu angelegt 2026-08-06** · **Die risikoreichste Zahlenverwechslung des Registers:** zwei verschiedene Modelle, zwei verschiedene Metrikebenen, eine Zahl. Wer die Checkpointdateinamen als Ergebniszahlen zitiert, berichtet Validierungswerte als Testwerte. Die 05-Skizze zitiert die Checkpoints ausdrücklich \enquote{mit den val-AUC-Werten der Dateinamen} (vgl. D39) — dort ist die Ebene zu benennen. **Kein Methodenfehler, sondern ein Berichtsfehler:** dass `val/auc_video` die Modellauswahl steuert (E3), ist korrekt und **darf nicht geändert werden** — eine Auswahl auf dem Testsplit wäre Leakage und machte jede berichtete Testzahl optimistisch verzerrt. Val ist die Auswahlgröße, Test die Generalisierungsschätzung; getrennt zu halten ist allein die Berichterstattung. Als Übertragungsauflage zusätzlich als **V8** in der Vault-Checkliste geführt. **⏳ Verfallszahl N5** — die konkreten Zahlenpaare ändern sich, sobald neue Läufe stattfinden; die *Regel* bleibt gültig. Siehe §*Zahlen mit Verfallsdatum*. Umfang: Halbsatz je Fundstelle |
| Q26 | **`vault/Writing/` enthält 12 Textentwürfe, die die Kapitelskizzen als `QUELLE` zitieren** (u. a. `experimental-setup-de.md`, `results-de.md`, `conclusion-de.md`). Ob der Entwurf oder die `.tex`-Datei aktueller ist, ist **ungeprüft** | 12 | — | ○ | **Neu angelegt 2026-08-06** · [12 §3.5](12_dokumentation_vault.md) verlangt genau diese Prüfung. Es ist dieselbe Fehlerklasse wie beim Archiv, nur unmarkiert: die Entwürfe tragen keine Warnung, werden aber in mindestens fünf Skizzenblöcken als Quelle geführt. **Vor dem Ausschreiben** je Entwurf einmal gegen Code und Register gegenlesen. Umfang: Arbeitsanweisung, kein Belegtext |
| Q27 | **`docs/kapitel/archiv/` enthält 10 ältere Kapitelfassungen mit denselben Fehlern, die am 2026-08-06 im Fließtext korrigiert wurden** — `00abstract.txt:24` und `06Results.txt:32` tragen weiterhin die Erzählung \enquote{semantische Regionen → Hintergrund} | 12 | — | ○ | **Neu angelegt 2026-08-06** · Die Korrekturen an `00Abstract.tex`, `01Einleitung.tex`, `06Results.tex` und `07Discussion_Limitations.tex` gelten **nicht** für diese Kopien. Wer beim Umschreiben eines Kapitels auf die `.txt`-Vorstufe zurückgreift, holt die korrigierte Aussage zurück. **ERLEDIGT 2026-08-06 — als Agentenwarnung umgesetzt, nicht als Belegtext.** Der Autor hält das Risiko für einen Menschen für vernachlässigbar, für LLM-Agenten aber nicht. Angelegt wurde deshalb `docs/kapitel/archiv/README.md` mit der harten Regel und einer Tabelle der konkret enthaltenen Falschaussagen; zusätzlich führt `CLAUDE.md` jetzt einen eigenen Abschnitt *\enquote{Veraltete Quellen — harte Regel für Agenten}* mit allen **drei** betroffenen Pfaden (`docs/archive/`, `docs/kapitel/archiv/`, `vault/…/Writing/`) und den vier belegten Durchschlägen. Kein Zielkapitel — reine Repositoriumshygiene |
| Q31 | **Die Übertragungsauflagen für Kapitel 6 sind als Checkliste geführt.** Die Tabelle *\enquote{Zahlen aus dem Vault, die nicht ungeprüft in `06Results.tex` dürfen}* trägt die stabilen IDs **V1--V8** und eine Statusspalte (`offen` / `erledigt` / `–`) | 12 | 06 | ○ | **Neu angelegt 2026-08-06** als Antwort auf die Beobachtung der 12er-Prüfung, dass diese Gruppe weder ID noch Status hatte. **V8 ist dabei neu hinzugekommen** (Val- gegen Test-AUC, vgl. Q25). Die acht Einträge zählen **nicht** in die Gesamtbilanz der Matrix — sie beschreiben keine implementierten Mechanismen, sondern Auflagen beim Übertragen von Zahlen. Beim Ausschreiben von Kapitel 6 Zeile für Zeile abhaken |
| Q28 | **`docs/superpowers/plans/2026-06-15-gpu-side-normalization.md` (18\,KB) ist ein Planungsdokument mit ungeprüftem Umsetzungsstand** | 12 | — | ○ | **Neu angelegt 2026-08-06** · Potenziell dieselbe Klasse wie `relevance_regularization.md`: ausführlich geplant, im Code nicht vorhanden. [12 §1.3](12_dokumentation_vault.md) führt es ohne Statusangabe. **Vor jeder Erwähnung im Beleg gegen `src/` prüfen** — sonst droht ein zweiter Fall F57. Umfang: Prüfauftrag |
| Q29 | **Bezeichnerkollisionen zwischen Register und Beleg.** `G1`–`G17` sind in dieser Matrix Phase-3-Zeilen, im Beleg bezeichnen `G1`–`G5` die Forschungslücken aus `Knowledge/Research Gaps.md`. `H2` ist hier ein PGD-Angriffsziel, in `frontend_roadmap.md` ein Oberflächenkürzel (neben `I1`–`I4`, `A1`, `A2-Box`, `E1`, `E2`, die in Codekommentaren auftauchen) | 12 | — | ○ | **Neu angelegt 2026-08-06** · **GELÖST 2026-08-06 durch eine Zitierkonvention** statt durch Umnummerierung (die Register-IDs sind als stabil zugesichert und extern referenziert): `Reg. G4` bezeichnet die Matrixzeile, `Gap G4` bzw. `Forschungslücke G4` die Forschungslücke. Dieselbe Regel für `Reg. H2` gegen das Oberflächenkürzel `H2`. Die Konvention steht in [README §Zitierkonvention](README.md); die beiden **live mehrdeutigen** Stellen sind entschärft (`04Methodology.tex:1176` und `06Results.tex:440` tragen jetzt `Reg. G…` samt Warnkommentar). **Im Fließtext der Belegarbeit tauchen Register-IDs nie auf** — dort ist `G4` immer die Forschungslücke, die Kollision existiert nur in den Skizzenkommentaren. **Offen bleibt der Anschlusspunkt:** die Forschungslücken G1--G5 werden in Kapitel 4 und 7 referenziert, aber nirgends eingeführt; die Einführungstabelle am Ende von Kapitel 3 ist in der 03-Skizze bereits gefordert (`03Related Work.tex:87-92`) und noch nicht geschrieben |
| Q30 | Fünf Dokumentbestände ohne eigenen Mechanismus: `Knowledge/Claim Map.md` (Zuordnung Aussage → Quelle), `Method Taxonomy.md`, `Literature Overview.md`, `docs/explanations/` (6 Glossardateien, 43\,KB), `docs/process.md` (21\,KB) und die sechs Tagesnotizen unter `Daily/` | 12 | — | – | **Neu angelegt 2026-08-06**, zugleich **bewusst weggelassen**: Arbeits- und Nachschlagematerial, das keinen implementierten Mechanismus beschreibt. Sammelzeile, damit die Prüfung dokumentiert ist. **Ausnahme:** `Claim Map.md` ist beim Ausschreiben von Kapitel 3 ein nützliches Werkzeug, aber kein Belegsinhalt |
| Q35 | **Der Umfang der Implementierung ist nirgends beziffert.** Gemessen: **485 Projektdateien** (aus 138.591 im Arbeitsordner — 99,6 % sind Datenbestand oder Fremdcode), **110 Python-Module mit 25.245 Zeilen** (`src/` 13.165, `scripts/` 5.819, `tests/` 6.085, `launch/` 176), **61 TS/TSX-Module mit 11.019 Zeilen**, 75 YAML-Dateien, 10 `.tex`-Kapitel | 00 | 09 | ○ | **Neu angelegt 2026-08-06** · Für eine Arbeit, deren Beitrag zu einem großen Teil in gebauter Software besteht, ist die Umfangsangabe eine legitime und derzeit fehlende Zahl. **Immer mit Abgrenzung nennen** (mit/ohne Tests, mit/ohne Build-Konfiguration), sonst ist sie nicht nachprüfbar — die Frontend-Zahl existiert genau deshalb in zwei Fassungen (60 Module/10.995 Zeilen unter `frontend/src/`, mit `vite.config.ts` 61/11.019). **⏳ Verfallszahl N1** — vor Abgabe erneut messen, siehe §*Zahlen mit Verfallsdatum*. Umfang: Tabelle im Anhang |
| Q36 | **Zwei Register-Angaben zum Datenbestand widersprechen sich.** [00 §5](00_inventar.md) nennt `data/` mit **59.894 Dateien / 11,3\,GB** (29.318 MP4, 30.537 JSON-Sidecars, 4 HDF5); Zeile **Q14** führt den DVC-Hash `1a1063a7…dir` über **59.777 Dateien / 10,7\,GB** | 00 / 11 | 05 | ○ | **Neu angelegt 2026-08-06** · Differenz: 117 Dateien und 0,6\,GB. Plausibel ist, dass der DVC-Hash einen Teilbaum oder einen früheren Stand erfasst — geprüft ist es **nicht**. **Vor jeder Angabe im Setup-Kapitel klären, welche Größe gemeint ist:** Q14 führt den Hash als \enquote{zitierfähige Kennung des Datenstands}, und eine Kennung, die nicht den ganzen Bestand deckt, trägt diese Rolle nur eingeschränkt. Umfang: Prüfauftrag, dann eine Zahl |
| Q37 | **Wo das Codegewicht liegt — für die Umfangs-Triage relevant.** Die zehn größten Python-Module tragen etwa die Hälfte des Produktivcodes; `src/api/inference.py` ist mit **3.744 Zeilen** mit Abstand das größte (rund 28 % von `src/`). Im Frontend sind die beiden größten Module `RobustnessPanel.tsx` (780) und `AdversarialPanel.tsx` (726) | 00 | 09 | ○ | **Neu angelegt 2026-08-06** · Zwei Lesarten, beide brauchbar. (1) Der Aufwandsschwerpunkt liegt in der **Laufzeit-Analysepipeline**, nicht im Training — das deckt sich damit, dass [07](07_inference_pipeline.md) das dichteste Registerdokument ist. (2) **Für die Umfangsentscheidung:** die beiden größten Frontend-Module liegen genau in den Phasen, deren Gewicht im Beleg reduziert werden soll. Der gebaute Apparat ist dort also groß, während die Ergebnisse fehlen — genau die Konstellation, für die Befund 2 den Apparat als Berichtsgegenstand vorsieht. **⏳ Verfallszahl N2** — vor Abgabe erneut messen; die Rangfolge kann kippen, siehe §*Zahlen mit Verfallsdatum*. Umfang: Halbsatz, alternativ eine Anhangtabelle mit Q35 |
| Q32 | **Drei Abbruchguards, die Zufallszahlen als Messwerte verhindern:** `eval.py` bricht ohne `ckpt_path` mit `ValueError` ab (sonst würde ein **frisch initialisiertes** Modell evaluiert und sein Zufallsergebnis als Messwert ausgewiesen); `get_metric_value` wirft bei gesetztem, aber unbekanntem Metriknamen (sonst optimierte ein Optuna-Sweep still gegen `None`); `instantiate_callbacks`/`_loggers` werfen `TypeError`, wenn die Konfiguration kein `DictConfig` ist | 03 | 09 | ○ | **Neu angelegt 2026-08-06** · Dieselbe Klasse wie A31, A44 und B20 — Guards, die stille Fehlmessungen verhindern. Der erste ist der schwerwiegendste: ohne ihn wäre eine berichtete Zahl schlicht Rauschen. Gehört in den Anhangsabsatz zur Qualitätssicherung. Umfang: Halbsatz |
| Q33 | **Jeder Lauf hinterlässt zwei Nachweise, auch wenn er abstürzt:** `task_wrapper` schreibt den Traceback nach `<output_dir>/<task_name>.log`, schließt im `finally`-Zweig den W&B-Lauf sauber ab und wirft weiter — ein abgestürzter Lauf einer Multirun-Reihe bleibt so nicht als \enquote{läuft noch} hängen. Der Konfigurationsbaum wird mit `resolve=True` geschrieben, die Interpolationen stehen also **aufgelöst** in `config_tree.log` | 03 | 05 | ○ | **Neu angelegt 2026-08-06** · Präzisierung zu Q2 (\enquote{aufgelöste Konfiguration wird je Lauf mitgeschrieben}): `resolve=True` ist der Grund, warum die Datei selbsttragend ist und nicht auf die Hydra-Bäume verweist. Der `task_wrapper`-Teil ergänzt Q20 — dessen Gegenmittel `config_tree.log` entsteht nur, weil dieser Pfad auch im Fehlerfall durchläuft. Umfang: Halbsatz |
| Q34 | **`enforce_tags` feuert praktisch nie und erklärt trotzdem einen Schalter der Befehlsreferenz:** die Prüfung greift nur bei leerem `cfg.tags`, und `configs/train.yaml`/`eval.yaml` belegen beide `tags: ["dev"]` vor. Würde sie greifen, wirft sie im **Multirun** einen `ValueError` und fragt im Einzellauf **interaktiv** nach — deshalb setzen Skriptaufrufe `extras.enforce_tags=false`, sonst blockierte ein Prompt einen nicht-interaktiven Lauf | 03 | 09 | ○ | **Neu angelegt 2026-08-06** · Erklärt den Schalter, der in `CLAUDE.md` und `docs/commands.md` an den `explain.py`-Aufrufen steht und sonst unmotiviert wirkt. Umfang: Halbsatz in der Befehlsreferenz des Anhangs; Kandidat für `–`, falls der Anhang gekürzt wird |
| Q23 | **Konfigurationskommentare veralten, `class_weights: auto` nicht:** die Gewichtswerte in `configs/model/*.yaml` sind Momentaufnahmen und werden zur Fit-Zeit neu berechnet; `train_video_balanced.yaml` nennt für dasselbe Gewicht „~8,7“ statt 7,361 — im Beleg aus dem Lauf-Log zitieren | 10 | 05 | ○ | **Neu angelegt 2026-08-05** · **Betrifft eine bereits notierte Zahl:** die 05-Skizze führt [0,54; 7,36] / [0,54; 7,37] / [0,56; 4,90] als „zuletzt gemessene Gewichte“ — das sind die YAML-Momentaufnahmen · nachgetragen in die 05-Skizze (§Klassenungleichgewicht) · gilt sinngemäß auch für die ~0,56 aus D37 |

---

## Strukturbefunde der Bestandsaufnahme

Punkte, die beim Erstellen des Registers auffielen und **nicht** aus einer einzelnen
Codezeile folgen. Sie stehen ausführlich in den verlinkten Dokumenten; hier gebündelt,
damit sie beim Abgleich nicht untergehen.

### Dokumentationslücken (Ergebnisse fehlen, Code ist da)

| Befund | Fundstelle | Konsequenz |
|---|---|---|
| **Keine Ergebnisnotiz zu Phase 4.** Adversarial und UAP sind vollständig implementiert — Sweeps mit Wiederaufnahme, UAP-Kern, drei Log-Scraper, fünf Testmodule — aber `vault/Results/` enthält **acht** Notizen, alle zu Phase 1, 2 und 3. | [12 §3.2](12_dokumentation_vault.md), Zeile H16 | Klären: Läufe nicht durchgeführt oder nur nicht dokumentiert? Ohne Ergebnisse ist Phase 4 im Beleg nur Methode ohne Befund. |
| **Keine Ergebnisnotiz zu VideoMAE Phase 2**, zu den **LoRA-Läufen** und zum **adversarialen Training** (4.2). | [12 §3.2](12_dokumentation_vault.md) | Betrifft D2, D10–D13, H13. Neun der 29 Experimentkonfigurationen haben keinen dokumentierten Lauf. |
| **`06Results.tex` ist mit 14 KB knapp** — das drittkleinste Kapitel, bei acht dokumentierten Experimenten plus zwei vollständigen Sweeps. `04Methodology.tex` ist mit 68 KB fast fünfmal so groß. | [12 §2](12_dokumentation_vault.md) | Missverhältnis Methode ↔ Ergebnis. Prüfen, ob Vorhandenes fehlt oder ob schlicht Läufe ausstehen. |
| **29 Experimentkonfigurationen existieren, ~8 Läufe sind dokumentiert.** | [10 §Beobachtungen](10_konfiguration.md) | Im Beleg zwischen *implementiert* und *durchgeführt* trennen. |

### Zahlen mit Verfallsdatum — vor Abgabe erneut messen

Diese Angaben sind **heute korrekt und werden es nicht bleiben.** Sie hängen am
Codebestand, nicht an einer Messung, und ändern sich mit jeder weiteren Implementierung.
Der Autor hat für den verbleibenden Projektverlauf **mindestens eine weitere Umsetzung**
angekündigt (Stand 2026-08-06); erwartet wird eine moderate, keine radikale Änderung.

**Regel: erst ausschreiben, dann kurz vor Abgabe einmal nachmessen und die Zahlen
ersetzen.** Keine dieser Zeilen ist ein Grund, mit dem Schreiben zu warten — aber jede ist
ein Grund, die Zahl nicht früh in mehreren Kapiteln zu verteilen.

| # | Betroffene Zeile | Was verfällt | Auslöser |
|---|---|---|---|
| N1 | **Q35** | 485 Projektdateien; 110 Python-Module / 25.245 Zeilen; 61 TS/TSX / 11.019 Zeilen; 75 YAML; 10 `.tex` | **jede** neue Datei oder Codezeile |
| N2 | **Q37** | Codegewicht-Rangfolge, `inference.py` 3.744 Zeilen, \enquote{die zehn größten tragen etwa die Hälfte}, Frontend-Spitzenreiter | dito; die Rangfolge kann kippen |
| N3 | **Q4** | \enquote{336 Tests, 13 davon weisen methodische Eigenschaften nach} | neue Testmodule |
| N4 | **Q22**, **D20** | \enquote{29 Dateien unter `configs/experiment/`, 27 Trainingsexperimente, acht dokumentierte Läufe} sowie \enquote{`ckpt_export_name` ist in 27 der 29 Konfigurationen gesetzt} | eine neue Experimentkonfiguration — **beide Zahlen sind dieselbe Grundgesamtheit und müssen gemeinsam nachgezogen werden** |
| N5 | **Q25**, **V8** | die konkreten Zahlenpaare Val gegen Test (Fusion 0,976 / 0,960; Audio 0,976) | **nur bei neuen Läufen** — nicht durch Codeänderung allein. Die *Regel* selbst (Val ist Auswahlgröße, Test Generalisierungsschätzung) bleibt gültig |
| N6 | **Q36**, **Q14**, **A14** | die Dateizahlen des Datenbestands (59.894 / 59.777), der DVC-Hash `1a1063a7…dir` **und die Splitgrößen** — A14 ist zugleich ein offener `!`-Widerspruch (9.482/1.382/1.471 bei 165 Identitäten gegen 9.959/861/1.180 bei ~30) | Neugenerierung oder Erweiterung von `data/`. **Reihenfolge beachten:** erst A14 als Widerspruch entscheiden, dann die Zahl einmal einsetzen — nicht umgekehrt |
| N7 | **F25b**, **F25c** | Status \enquote{geplant und bestätigt, Umsetzung steht aus} | **die angekündigte Implementierung selbst.** Landet sie, wechselt F25b von Ausblick zu Methodik + Ergebnis und das Register ist in [02](02_modelle.md) (Verlustterm), [04](04_xai.md) (Maskenerzeugung), [10](10_konfiguration.md) (neue Experimentkonfiguration) und [09](09_tests.md) nachzutragen |
| N8 | **F2**, **Q12** | die Versionsbindungen: `transformers==4.57.6`, `torch==2.11.0`, `lxt==2.1`, `numpy==2.4.4` | **jede Abhängigkeitsaktualisierung.** Der kritische Eintrag ist `transformers`: die lxt-Patches für VideoMAE und Wav2Vec2 sind auf **genau diese Version** geschrieben, ein Upgrade bricht AttnLRP. Wird aktualisiert, ist F2 neu zu verifizieren, bevor irgendeine Heatmap neu erzeugt wird |
| N9 | **C10**, **D25** | sämtliche Parameterzahlen: 2.101.248 (beide Attention-Blöcke), `concat`-Kopf ~1,32\,M statt 3,42\,M, 86.228.738 (entfroren), 3.074 (eingefroren) | **jede Architekturänderung.** Ein neuer Verlustterm oder Kopf verschiebt sie; D25 nennt `log_hyperparameters` als Quelle — dort ist nachzumessen, nicht zu schätzen |
| N10 | **S1**, **S5**, **S9**, **S33**, **S29** | die Bestandszahlen des Demonstrators: fünf Router / sieben Routen; 20 Pydantic-Schemas; 15 erklärbare Visualisierungen, 14 Abschnittsarten, 13 Widgets; \enquote{sechs von sieben Regionen} | jede neue Route, jedes neue Schema, jede neue Ansicht. **Konkret erwartbar:** bekommt die angekündigte Implementierung eine Ergebnisansicht, ändern sich S9/S33 — und die Aussage \enquote{alle 15 `VisualId` sind belegt, es gibt keine Lücke} ist dann neu zu prüfen |
| N11 | **F46**, **F25c**, Zitierbarkeitstabelle | `references.bib` mit **46 Einträgen**, die Aussage \enquote{alle werden in `03Related Work.tex` zitiert}, die drei Notizen ohne Bib-Eintrag und die vier fehlenden xAI-Quellen | **jede Aufnahme in `references.bib`.** Anders als die übrigen Zeilen ist das keine Beobachtung, sondern eine **Bringschuld**: Tsunakawa 2019, Kohlbrenner 2020 und Ross et al. 2017 *müssen* aufgenommen werden, sonst sind F46 und F25c nicht belegbar formulierbar |
| N12 | **Q22**, [12 §3.2](12_dokumentation_vault.md) | der Vault-Bestand: 104 Dateien, 48 Paper-Notizen, **8 Ergebnisnotizen** | jede neue Notiz. **Der wichtigste Einzelfall:** Läuft die angekündigte Implementierung, sollte eine **neunte Ergebnisnotiz** entstehen. Fehlt sie, ist das Ergebnis nicht dokumentiert und darf nicht in `06Results.tex` — derselbe Maßstab, der Phase 4 auf \enquote{implementiert, Ergebnisse ausstehend} festlegt |

> **Auch außerhalb dieser Matrix betroffen:** der Kopf von [README](README.md)
> (\enquote{Erfasst: 485 Dateien; 110 Python-Module (25.245 Zeilen), 61 TS/TSX-Module
> (11.019 Zeilen), 75 YAML}) und die Verteilungstabellen in [00 §2–§4](00_inventar.md).
> Wer N1 nachmisst, zieht diese drei Stellen im selben Zug nach.
>
> **Drei Klassen, drei Handhabungen.** Die zwölf Einträge sind nicht gleichartig:
>
> - **Reine Nachmessung** (N1, N2, N3, N4, N9, N10, N12) — die Zahl wird neu erhoben,
>   die Aussage bleibt. Aufwand: Minuten, wenn die Zahl an *einer* Stelle steht.
> - **Bedingte Nachmessung** (N5, N6) — ändert sich nur bei neuen Läufen bzw. neuen
>   Daten. Findet beides nicht statt, ist nichts zu tun.
> - **Echte Bringschuld** (N7, N8, N11) — hier reicht Nachmessen nicht. N11 verlangt drei
>   Bibliografieeinträge, ohne die zwei Registerzeilen nicht formulierbar sind; N8 verlangt
>   eine Verifikation, bevor eine Abbildung neu erzeugt wird; N7 verlangt eine
>   Umverteilung über vier Registerdokumente und drei Kapitel.
>
> **Praktische Konsequenz für das Schreiben:** Jede Zahl aus dieser Tabelle gehört an
> **genau eine** Stelle im Beleg, auf die die übrigen Kapitel verweisen. Steht dieselbe
> Zahl in Einleitung, Methodik und Fazit, verdreifacht sich der Nachtragsaufwand — und
> genau so entstehen die Inkonsistenzen, die diese Kontrollreihe an anderer Stelle
> aufgeräumt hat.

---

### Zahlen aus dem Vault, die **nicht** ungeprüft in `06Results.tex` dürfen

Diese Befunde stehen in den Ergebnisnotizen selbst und sind beim Übertragen zwingend
mitzuführen. Sie sind die risikoreichste Gruppe des ganzen Registers: ein Fehler hier wird
zu einer falschen Zahl im Beleg.

> **Als Checkliste zu benutzen** (eingerichtet 2026-08-06, Registerzeile Q31). Die acht
> Einträge tragen die stabilen IDs **V1–V8** und eine eigene Statusspalte. Sie zählen
> **nicht** in die Gesamtbilanz der Matrix, weil sie keine implementierten Mechanismen
> beschreiben, sondern Übertragungsauflagen. Statuskürzel: `offen` = beim Ausschreiben von
> Kapitel 6 noch zu berücksichtigen · `erledigt` = im Fließtext umgesetzt · `–` =
> gegenstandslos geworden. **Jeder Eintrag ist erst abzuhaken, wenn die betroffene Zahl im
> Beleg steht oder nachweislich nicht vorkommt.**

| # | Befund | Fundstelle | Konsequenz | Status |
|---|---|---|---|---|
| **V1** | **Eine Zahl ist zurückgezogen.** Die früher berichtete **visual-only-AUC 0,832 des Audiomodells** ist in drei Notizen gleichlautend widerrufen (Korrektur vom 2026-06-16): Die Kategorie hat unter `label_audio` nur **4** positive Videos, die Metrik ist Rauschen. | [12 §3.2](12_dokumentation_vault.md) | **Darf in `06Results.tex` nicht auftauchen.** Damit entfällt auch die Erzählung „Audio ist schwach auf visuellen Fakes → motiviert Fusion". | offen |
| **V2** | **Unimodale und multimodale Läufe sind verschiedene Label-Aufgaben** (`label_audio`/`label_video` vs. kombiniertes `label`). | [12 §3.2](12_dokumentation_vault.md) | Ihre `auc_video` dürfen **nicht direkt verglichen** werden — auch nicht in einer gemeinsamen Tabellenspalte. | offen |
| **V3** | **Referenzwert-Konflikt beim Perturbationstest.** `videomae-frame-perturbation-temporal.md` führt den Clean-Wert als `auc_video` **0,745**, `videomae-unimodal-video-baseline.md` berichtet für denselben eingefrorenen Lauf **0,730** aggregiert — 0,745 ist dort die *visual-only*-Teilkategorie (273 positive Videos). | [12 §3.2](12_dokumentation_vault.md), Zeile B10 | **Vor der Übernahme festlegen, welche Größe gemeint ist.** Drei andere Notizen zitieren „frozen 0,745" durchgängig als visual-only-Vergleichspunkt. | offen |
| **V4** | **Die Datensatz-Ablation ist ausdrücklich kein Ergebnis.** `dataset-ablation-pairing-diversity.md` trägt `status: in-progress`: nur der `keep_pairs`-Arm ist trainiert, der Kontrollarm ist vorverarbeitet aber untrainiert, die SWAN-DF-Evaluation fehlt. Wörtlich: „Do **not** cite a pairing/diversity effect from this yet." | [12 §3.2](12_dokumentation_vault.md), Zeilen A20, A21 | Die **Methodik** (A20/A35/A36) gehört in den Beleg, ein **Effekt** nicht. | offen |
| **V5** | **Der Phase-3-Sweep mischt zwei Datenstufen:** die evaluierten Checkpoints wurden auf der 32-Identitäten-Stufe trainiert, die Evaluation läuft auf 1471 Testvideos der 165-Identitäten-Stufe. Deshalb liegt die dort berichtete Clean-AUC (0,857) unter der Baseline-AUC (0,999). | [12 §3.2](12_dokumentation_vault.md), Zeile G10 | Die Asymmetrie gehört **benannt** in `05Experimental_Setup.tex`; die Leckagefreiheit ist über den deterministischen Hash begründet. | offen |
| **V6** | **Der Perturbationsbefund lief nur auf dem eingefrorenen Phase-1-Checkpoint**, nicht auf dem Phase-2-Modell (0,999). | [12 §3.2](12_dokumentation_vault.md), Zeile B10 | Vor einer allgemeinen Aussage über die Zeitnutzung des Modells dort zu wiederholen. | offen |
| **V7** | **Der Heatmap-Lokalisierungsbefund ist `n = 1`.** Gemessen an einem Clip: Mund erhält an den tatsächlich gefälschten Frames 17,4 % der Relevanz gegen 16,5 % im Rest; Untergesicht im Fake-Fenster sogar *weniger* (40,4 % vs. 49,2 %); Mund nur in 29/237 Frames stärkste Region. | [12 §1.2](12_dokumentation_vault.md), Zeile F25a | Als **Einzelfallmessung** formulieren, nicht als Modell- oder Datensatzeigenschaft. Die Schlussfolgerung „genau, aber nicht faithful lokalisiert" ist selbst ein verwertbares xAI-Ergebnis. | offen |
| **V8** | **Val-AUC ist nicht Test-AUC.** Die Checkpointdateinamen tragen die **Validierungs**-AUC (`epoch_006-val_auc_video_1.000_video_phase2.ckpt`); die Ergebnisnotizen berichten die **Test**-`auc_video`. Für die Fusion stehen deshalb zwei Werte nebeneinander: Test **0,960**, Val **0,976** — und dieselbe Zahl 0,976 ist zugleich die Test-AUC des eingefrorenen **Audio**modells. | [12 §3.2](12_dokumentation_vault.md), Zeilen Q25, D39, E3 | **Die Auswahl auf `val/auc_video` ist methodisch korrekt und darf nicht geändert werden** — eine Selektion auf dem Testsplit wäre Leakage. Zu trennen ist allein die *Berichterstattung*: Val ist die Auswahlgröße, Test die Generalisierungsschätzung. Nie einen Dateinamenwert als Ergebnis zitieren. | offen |

### Zitierbarkeit und Bibliografie

| Befund | Fundstelle | Konsequenz |
|---|---|---|
| **Drei Paper-Notizen haben keinen Bib-Eintrag** und sind damit nicht zitierbar: `audio-adversarial-carlini-2018`, `deeperforensics-jiang-2020`, `in-ictu-oculi-li-2018`. Umgekehrt existiert `korshunov2023swandf` als Eintrag ohne Notiz. | [12 §3.3](12_dokumentation_vault.md) | Vor dem Zitieren aufnehmen oder auf die Aussage verzichten. `references.bib` hat 46 Einträge, **alle** werden in `03Related Work.tex` zitiert. |
| **Vier Quellen der xAI-Argumentation fehlen in `references.bib`:** Tsunakawa 2019 und Kohlbrenner 2020 (Abgrenzung der contrastiven Kodierung) sowie Ross et al. 2017 (Begründung der Relevanz-Regularisierung). | [12 §3.3](12_dokumentation_vault.md), Zeilen F46, F25c | Ohne Aufnahme sind F46 und F25c nicht belegbar formulierbar. |
| **`Archive/istvt-2023.md` ist nicht zitierfähig** — `status: archived`, `evidence-level: metadata`, Paper paywalled, ausdrücklicher Vermerk „**do not cite**", bewusst nicht in `references.bib`. | [12 §3.6](12_dokumentation_vault.md), Zeile C8 | ISTVT darf nur als **verworfene Architekturoption** erwähnt werden — ohne Ergebnis- oder Methodenzahlen. |

### Zustandsbefunde im Repositorium

| Befund | Fundstelle | Konsequenz |
|---|---|---|
| **`configs/model/istvt.yaml` ist leer (0 Bytes).** ISTVT ist in `CLAUDE.md` als mögliche Erweiterung genannt und in `vault/Archive/istvt-2023.md` recherchiert, aber nicht implementiert. | [00 §6](00_inventar.md), [10 §2](10_konfiguration.md), Zeile C8 | Gehört in `08Conclusion.tex` (Ausblick), **nicht** in `02Tech_Explanations.tex` oder `04Methodology.tex` als Baseline. |
| **`src/data/` hat keine `__init__.py`**, alle anderen `src/`-Unterpakete schon. Importe laufen über Namespace-Packages. | [00 §6](00_inventar.md), [01](01_datenpipeline.md) | Kein Fehler, aber eine Inkonsistenz. Falls der Beleg die Paketstruktur abbildet, nicht als reguläres Paket darstellen. |
| **Widerspruch bei der Zeilenlänge:** `CLAUDE.md` fordert ≤ 88, `pyproject.toml` setzt `line-length = 120` **und** ignoriert `E501` zusätzlich. | [11 §1](11_infrastruktur.md) | Durchgesetzt wird 120. Wenn der Beleg Codekonventionen nennt, ist `pyproject.toml` die Quelle — nicht `CLAUDE.md`. |
| ~~**Zwei Laufartefakte im Wurzelverzeichnis** müssten entfernt werden~~ — **erledigt/gegenstandslos.** `server_debug.log` und `tea_debug.log` liegen nur lokal; `git ls-files` kennt sie nicht, `.gitignore` L71 (`*.log`) fasst sie. Ein geklontes Repositorium enthält sie nicht. | [11 §6](11_infrastruktur.md) korrigiert [00 §6](00_inventar.md) | **Im Beleg nichts zu erwähnen.** |
| **`docs/archive/` enthält Dateinamen, die mit aktuellen identisch sind** (`project.md`, `xai.md`, `frontend.md`, `adversarial.md`). | [12 §1.5](12_dokumentation_vault.md) | Reale Verwechslungsgefahr beim Zitieren. `CLAUDE.md` warnt ausdrücklich, das Archiv nicht als aktuelle Quelle zu verwenden. **Die Warnung kam zu spät — siehe die Zeile darunter.** |
| **Das Archiv ist nachweislich in den Beleg durchgeschlagen: vier Fehler mit belegter Herkunft** (geprüft 2026-08-06). `docs/archive/xai.md` §3 schreibt „Relevanz wird pro Wort-Token **aufsummiert**" (→ Widerspruch **F14**, korrigiert), §1 führt Attention Rollout als „Lösung 1" (→ **F57**, offen), §2 beschreibt, wie sich die Attention „von wirrem Suchen am Rand präzise auf den Mund verschiebt" und §3 stellt die Lip-Sync-Frage als zentrale Forschungsfrage (→ beide durch die Messung **widerlegt**); `docs/archive/adversarial.md` §2.1 nennt die Regionsliste „Mouth, Eye, Jaw, Shoulder, Background" (→ **F18**, Quelle bereinigt). | Gemessen; [`../archive/xai.md`](../archive/xai.md), [`../archive/adversarial.md`](../archive/adversarial.md) | **Jede weitere Aussage im Beleg, die sich nur auf ein Archivdokument stützt, ist gegen den Code zu prüfen.** Das Archiv beschreibt die Planung von April/Mai 2026, nicht das gebaute System. Beide Archivdateien tragen jetzt Korrekturkästen an den betroffenen Stellen. |
| ~~**Zwei ungetrackte Dokumente**: `xai_pipeline_reference.md` und `relevance_regularization.md`~~ — **erledigt.** Beide sind inzwischen versioniert (`e3ec619` *Add Regularization Plan*, `49e2772` *Add older xai doc*); `git ls-files` führt beide. | [12 §1.2](12_dokumentation_vault.md), per `git ls-files` geprüft | Keine Maßnahme mehr nötig. Beide bleiben Belegquellen (F25, F46). |
| **`.env` und `.env.example` sind beide unversioniert.** `.gitignore` L64 fasst `.env`, L65 zusätzlich `.env.*` — womit auch die *Vorlage* nicht im Repositorium liegt. | [11 §1](11_infrastruktur.md) | Ein geklontes Repositorium enthält keine Vorlage für die Checkpoint-Variablen. **Entschieden 2026-08-06:** Die Variablenliste wird im Anhang (S3, §G) ausgeschrieben — die *Namen* stehen im Register und sind belegbar, für die maschinenabhängigen *Werte* (Pfade, Checkpoint-Dateinamen) genügen Platzhalter. Keine eigene Matrixzeile; der Halbsatz gehört in den S3-Eintrag. |
| **Pfadwiderspruch beim containerisierten Demonstrator:** `.env.example` und `docker-compose.yml` erwarten die Checkpoints unter `models/`, tatsächlich liegen sie unter `checkpoints/`; `checkpoints/` wird von Compose nicht eingehängt. | [11 §3](11_infrastruktur.md) | Ohne Kopieren scheitert die erste Anfrage mit `ModelNotReadyError` — immerhin laut. Im Anhang den **lokalen** Start als Reproduktionsweg nennen. |
| **Zwei überholte Konfigurationskommentare.** `train_audio_smoothing.yaml` behauptet, Wav2Vec2 unterstütze kein Mixup (der Code ruft `_mixup_training_loss`, `train_audio_mixup.yaml` setzt `mixup_alpha`); die Auto-Klassengewichte stehen in `videomae.yaml` als `[0.536, 7.361]`, in `train_video_balanced.yaml` als „~8,7". | [10 §4, §Beobachtungen](10_konfiguration.md) | Für den Beleg gilt der Code. Klassengewichte **aus dem Lauf-Log** zitieren, nicht aus der YAML — `class_weights: auto` rechnet sie zur Fit-Zeit neu. |
| **Zeilenzahl des Frontends — nachgemessen.** `frontend/src/` enthält **60** TS/TSX-Module mit **10.995** Zeilen, mit `vite.config.ts` (24 Z.) **61 Module / 11.019 Zeilen**. [08](08_frontend.md) stimmt damit exakt. | Gemessen; [08](08_frontend.md) bestätigt | Für den Beleg gilt die gemessene Zahl. Umfangsangaben immer mit Abgrenzung (mit/ohne Build-Config) versehen. **Korrigiert 2026-08-06:** Dieser Befund führte bis dahin, [00 §3](00_inventar.md) nenne 10.994 und sei nicht nachgezogen. Das trifft nicht mehr zu — [00 §3](00_inventar.md) führt für TS/TSX **gar keine Zeilenzahl** mehr, nur die Modulzahlen (49 `.tsx` + 12 `.ts` = 61). Es gibt hier also keine offene Inkonsistenz mehr; die Zahlen stehen in [README](README.md) und [08](08_frontend.md). |
| **Drei Commits kamen nach der Erstaufnahme dazu** (bestätigt): Basis war `19dd0d5`, danach `db5608f` (nur Kommentare), `e3ec619` und `49e2772` (die beiden xAI-Dokumente). Einzelne Fachdokumente wurden danach überarbeitet, andere nicht. | [README §Stand](README.md), `git log` | **Inhaltlich abgedeckt** — beide Dokumente stehen in [12 §1.2](12_dokumentation_vault.md) und waren als Dateien bereits inventarisiert. Offen bleiben nur zwei Textstellen: die Zeile darunter und der erledigte Ungetrackt-Befund. |
| ~~**[08 §4](08_frontend.md) enthält eine Aussage, die `db5608f` als veraltet korrigiert hat:** als Rückfallgrund für die L3-Balkenansicht wird „multimodale Ergebnisse ohne Gitter" genannt.~~ — **behoben 2026-08-06.** [08 §4](08_frontend.md) nennt jetzt nur noch Altcaches als Rückfallgrund und führt die Korrektur in einem eigenen Kasten. | Gemessen an `src/api/inference.py`; `git show db5608f` | Für den Beleg unverändert: das Fehlen des L3-Gitters darf **nicht** mit dem multimodalen Modus begründet werden — **beide** Audiopfade berechnen die Gitter (`inference.py:2348` unimodal, `:2547` multimodal). Zeile F33 gibt den korrekten Stand wieder. |
| **`vault/_system/lint-report.md`** meldet 5 Ergebnisnotizen ohne zugehörige Experimentnotiz; das im Schema vorgesehene Verzeichnis `Experiments/` existiert nicht. | [12 §3](12_dokumentation_vault.md) | Betrifft die Vault-Konsistenz, nicht den Beleginhalt. Nur relevant, falls der Anhang die Vault-Struktur beschreibt. |

### Fehlerquellen für die Abbildungen im Beleg

| Befund | Fundstelle | Konsequenz |
|---|---|---|
| **Darstellungsverstärkungen sind allgegenwärtig** (`color_gain = 3.0`, `RELEVANCE_DISPLAY_GAIN = 4`, `REL_GAMMA = 2.5`, Gamma- und Cap-Parameter). | [07 §3](07_inference_pipeline.md), [08](08_frontend.md), Zeile F23 | Screenshots zeigen **relative Muster, keine absoluten Relevanzwerte**. Gehört in die Abbildungslegenden. |
| **Mock-Modus** (`VITE_USE_MOCK=true`) erzeugt vollständige, aber **synthetische** Ergebnisse. | [08 §8](08_frontend.md), Zeile S11 | Jeder Screenshot muss aus dem echten Backend stammen. |
| **`-1.0` in den W&B-Sweep-Tabellen ist ein NaN-Sentinel**, kein Messwert — es bedeutet „nicht bestimmbar", meist weil die Stichprobe nur eine Klasse enthielt. | [05](05_robustheit_adversarial.md), Zeile H11 | Beim Übertragen von Rohtabellen in Ergebnisdiagramme herausfiltern, sonst entstehen erfundene Tiefpunkte. |
| **Zwei getrennte Farbimplementierungen** ohne automatischen Abgleich: `seismicColormap.ts` (Frontend-Canvas) und `_array_to_data_uri` (Backend-PNG). Sie teilen die **Kodierungslogik**, aber nicht die Farbwerte — das Frontend zeichnet durchgehend mit der aufgehellten F2-Rampe, das Backend mit matplotlibs seismic. | [08 §8](08_frontend.md), [07 §3](07_inference_pipeline.md), Zeile S28 | Canvas- und PNG-Abbildungen zeigen dieselben Daten **konstruktionsbedingt** in leicht verschiedenen Tönen. Zusätzlich weicht das Gesichtsschema mit `FILL_OPTS` in allen fünf Parametern ab (S35). |
| **Skript-Abbildungen und Frontend-Ansichten sind nicht dieselbe Größe.** `explain.py`, `explain_audio.py` und `explain_multimodal.py` rufen `explain()` ohne `per_class` — ihre Karten sind **Single-Seed**, das Frontend zeigt bivariate. | [04 §1](04_xai.md), Zeile F26 | Dürfen nicht als Vorher/Nachher oder als dieselbe Größe gegenübergestellt werden. |
| **Die L3-Abbildungen der Erklärskripte zeigen weitgehend das Energiespektrum von Sprache**, nicht das Frequenzverhalten des Modells — `audio_xai.compute_band_relevance` wurde nicht auf die energiegewichtete Formel nachgezogen. | [04](04_xai.md), [07 §8](07_inference_pipeline.md), Zeile F27 | Nicht als Aussage über frequenzabhängige Modellaufmerksamkeit verwenden; dafür ist die Frontend-Ansicht zuständig. |
| **Die beiden Videofiguren sind unterschiedlich skaliert:** `explain.py` fest `±1`, `explain_multimodal.py` auf das Betragsmaximum des gewählten Frames. | [04](04_xai.md), Zeile F28 | Gleiche Farbe bedeutet dort **nicht** gleiche Relevanz. Nicht ohne Hinweis nebeneinanderstellen. |
| **Konfidenzwerte sind richtungslos** (immer ≥ 0,5) — aus einer gestiegenen Konfidenz folgt nicht „stärker FAKE", ein Urteilsumschlag ist aus ihr unsichtbar. | [06](06_backend_api.md), Zeile S18 | In jeder Abbildungslegende das **Urteilsfeld** berichten, nie aus der Konfidenz zurückrechnen. |
| **`emphasizeRelevance` macht schwache Fake-Evidenz unsichtbar** (Rauschband 0,20–0,25 → ~0,03) — eine bewusste Rauschunterdrückung mit Nebenwirkung, nur in der Relevance-Ansicht. | [08 §4](08_frontend.md), Zeile S26 | **Die Abwesenheit eines L2-Balkens ist kein Freispruch.** Gehört in die Legende der Wortabbildung. |
| **Relevanz ist relativ, kein Prozentwert und nur innerhalb desselben Visuals vergleichbar** — Leseanweisung aus dem Erklärsystem selbst (`RelevanceScaleNote`). | [08 §10](08_frontend.md), Zeile S32 | Gehört in **jede** Abbildungslegende, die Relevanzwerte zeigt — insbesondere beim Vergleich zweier Clips. |
| **Das Gesichtsschema zeigt sechs von sieben Regionen** (`Chin` fehlt), zählt die siebte aber in die Prozentsumme. | [08 §3](08_frontend.md), Zeile S29 | Prozentangaben summieren sich sichtbar nicht auf 100 %; „MOST ATTENDED" kann eine Region nennen, die im Bild nicht hervorgehoben ist. |
| **`xai_pipeline_reference.md` ist ausdrücklich ein *älteres* Dokument** (Commit-Nachricht `49e2772`: „Add older xai doc"), wird in [12 §1.2](12_dokumentation_vault.md) aber ohne Vorbehalt als „**die Quelle für die Abbildungslegenden**" geführt. Sein §6.3 beschreibt `AnomalyRegionBars.tsx` — **die Komponente existiert nicht** (`find` liefert nichts); [08 §5](08_frontend.md) hält fest, dass die Tafel „TOP ANOMALY REGIONS" entfernt wurde. | Gemessen; [08 §5](08_frontend.md), Zeile S34 | **Die Tuning-Zahlen in §9 sind belastbar** — stichprobenartig gegen [07](07_inference_pipeline.md)/[08](08_frontend.md) geprüft, alle 13 Parametersätze stimmen überein. Der **Komponenten-Bestand** darin ist es nicht: Wer Abbildungslegenden daraus schreibt, riskiert die Beschreibung entfernter Ansichten. |
| **Die Zeitachsen der Oberfläche sind gröber als die Daten** (L1 0,64 s, Overlay ≈ 4 Hz, Doppelspieler 250 ms), und die beiden Chunk-Zeitreihen dürfen **unterschiedlich lang** sein (Forward-Chunks vs. Heatmap-Fenster), werden aber auf dieselbe Breite abgebildet. | [08 §3, §4](08_frontend.md), Zeile S31 | Eine senkrechte Position in der oberen Kurve entspricht nicht zwingend derselben in der unteren. Lokalisierungsaussagen nicht feiner formulieren als das Raster. |

---

## Vorab identifizierte Lückenkandidaten

Beim Erstellen des Registers fielen diese Punkte auf. Sie sind **Hypothesen**, keine
festgestellten Lücken — der Abgleich mit den `.tex`-Dateien steht noch aus:

| Priorität | Punkt | Zeile | Warum verdächtig |
|---|---|---|---|
| **P0** | Phase-4-Ergebnisse | H16 | Vollständig implementiert (Sweeps, UAP, Scraper, Tests), aber **keine Ergebnisnotiz im Vault**. `06Results.tex` hat nur 14 KB. |
| **P0** | `_band_confidence` (Band-Ablation) | F16 | Methodisch die stärkste Audio-Aussage (kausal statt attributiv), aber ein spätes Feature — leicht zu übersehen. |
| **P0** | Bivariate Relevanz | F3–F5 | Der zentrale xAI-Beitrag, dokumentiert in `docs/attnlrp_relevance_explanations_and_decision.md` (34 KB). |
| **P0** | Heatmap-Lokalisierung | F25a/b | Die **Diagnose** gehört schon jetzt in die Diskussion. Die **Lösung** (Explanation-Guided-Training) ist bestätigt geplant, aber noch nicht implementiert — bis dahin nicht in der Methodik beschreiben. Nach der Umsetzung ist das Register nachzuziehen. |
| **P1** | Spatial-Dominance-Diagnostik | B9, B10 | Ergebnisnotiz existiert (`videomae-frame-perturbation-temporal.md`), aber die Methode ist ungewöhnlich und braucht Erklärung. |
| **P1** | Datensatz-Ablation Frame-Zwillinge | A20, A21 | Eine echte methodische Auseinandersetzung mit einer Datensatzschwäche — gehört prominent in die Methodik. |
| **P1** | Cross-Dataset SWAN-DF | A22 | Vollständige Infrastruktur vorhanden; Ergebnisnotiz fehlt. |
| **P1** | LoRA-Merge-Kreis | D10–D13 | Drei zusammenhängende Mechanismen; Beleg erwähnt vermutlich nur „LoRA". |
| **P1** | Kategorienweise Test-AUC | E7 | Diagnostisch wertvoll (welche Manipulationsart wird erkannt), leicht zu übersehen. |
| **P2** | Recall@FPR=1 % | E4–E6 | Eigene Metrikimplementierung mit Begründung. |
| **P2** | SDPA/Eager-Asymmetrie | F12 | Klingt nach Implementierungsdetail, ist aber eine geschlossene Fehlerklasse. |
| **P2** | Darstellungsverstärkungen | F23 | Betrifft die Lesbarkeit **jeder** Abbildung im Beleg. |
| **P2** | ISTVT nicht implementiert | C8 | Muss im Ausblick stehen, nicht in der Methodik. |
| **P2** | Einzelläufe ohne Seed-Varianz | D19 | Gehört in die Limitationen. |
| **P0** | Zurückgezogene und nicht vergleichbare AUC-Zahlen | — | Die visual-only-AUC 0,832 ist widerrufen; unimodale und multimodale `auc_video` sind verschiedene Label-Aufgaben. Betrifft direkt `06Results.tex`. |
| **P0** | Widersprüchliche Aussagen zur Zeitnutzung | B9, B10 | Die Konfiguration heißt `frame_shuffle`, setzt aber `tubelet_shuffle`; die Ergebnisnotiz widerlegt die Spatial-Dominance-Hypothese. Beleg muss angeben, **welche** Störung welchem AUROC zugrunde liegt. |
| **P1** | Sweep- ↔ Frontend-Zahlen trennen | H21, H27, S23 | Fooling Rate, Attention Shift und Fusionsmodus sind in Sweeps und Oberfläche **verschiedene Größen**. Drei Stellen, an denen eine gemeinsame Tabelle falsch würde. |
| **P1** | Wiederaufnahme- und Scraper-Infrastruktur | H22–H26 | Erklärt, warum manche Tabellen unvollständig sind, und ist selbst eine Engineering-Leistung. Gehört in Anhang oder Limitationen. |
| **P1** | Empirische Befunde im Code | F35–F38, C13 | Vier Audio-Befunde und der Wav2Vec2-Konvergenzbefund begründen zentrale Designentscheidungen — als Entwicklungsbefund kennzeichnen oder neu messen. |
| **P2** | Novelty-Abgrenzung und Faithfulness-Caveat | F46, F47 | Formulierung ist im Quelldokument bereits entschieden; drei Quellen fehlen noch in `references.bib`. |
| **P2** | Stille Rückfälle und unmarkierte Zustände | F43, S15, D23 | Vollbildpfad, geometrische Regionen, Vollbild-Crop-Box und Nicht-Best-Checkpoint-Testwerte sind am Ergebnis nicht erkennbar. Ehrlichkeitspunkt für die Limitationen. |
| **P3** | Kein Frontend-Test | S12 | Limitation, falls Qualitätssicherung thematisiert wird. |
| **P3** | Weitere Testlücken und CI-Grenzen | Q11, Q21 | Nur relevant, falls der Beleg Qualitätssicherung als abgesichert darstellt. |

---

## Arbeitsreihenfolge nach dem Abgleich

Der Abgleich ist durchgeführt; die Statusspalte ist gefüllt. Was jetzt zu tun ist:

1. **Die neun `!`-Zeilen zuerst.** Eine falsche Beschreibung ist schlimmer als eine
   fehlende, und sechs davon stehen in bereits geschriebenem Text (Kapitel 4), sind also
   nicht durch das Ausschreiben der Skizzen abgedeckt. Zwei — `G1b` und `D2` — stehen
   sogar **in den Skizzen selbst** und würden beim Ausschreiben ungeprüft übernommen.
2. **Die `✗`-Zeilen sichten und entscheiden.** 130 Punkte sind zu viel für den Beleg. Für
   jeden gilt: aufnehmen, oder bewusst als `–` mit Begründung markieren. Die Entscheidung
   dokumentiert zu haben, ist in der Verteidigung mehr wert als die Vollständigkeit.
   Vorschlag für die Priorisierung innerhalb der `✗`-Zeilen:
   - **Muss:** alles, was die Lesbarkeit der Abbildungen betrifft (S18, S26, S28, S31,
     F26, F27, F43, H11) — ohne diese Sätze sind die Bildunterschriften falsch.
   - **Muss:** `S23` (alle Phase-4-Zahlen gelten für `cross_attention`) und `H21`
     (zwei Fooling Rates unter einem Namen) — beides erzeugt sonst falsche Tabellen.
   - **Sollte:** der Phase-4-Apparat (H18–H27) als das, was statt der Ergebnisse
     berichtet werden kann.
   - **Kann:** Infrastruktur (Q5, Q6, Q9, Q12–Q21) — nur falls Reproduzierbarkeit
     ein eigenes Thema wird.
3. **Die `○`-Zeilen brauchen keine Entscheidung**, nur Fließtext. Sie sind beim
   Ausschreiben der Kapitel 05–09 automatisch abgedeckt — mit Ausnahme der beiden unter
   Punkt 1 genannten Fehler.
4. ~~**Die `~`-Zeilen einzeln nachschärfen.**~~ — **erledigt.** Seit dem 2026-08-05 ist
   `~` leer; die betroffenen Stellen sind in die Skizzen überführt und stehen auf `○`.
5. **Beim Ausschreiben die Statusspalte nachziehen**, damit sie den Fortschritt abbildet.

6. **Zuletzt, kurz vor Abgabe: die zwölf Verfallszahlen N1–N12 abarbeiten.** Sie hängen am
   Codebestand und nicht an einer Messung — jede weitere Implementierung verändert sie.
   Der Autor hat mindestens eine weitere Umsetzung angekündigt. Die Liste steht in
   §*Zahlen mit Verfallsdatum*; betroffen sind Q35, Q37, Q4, Q22/D20, Q25/V8, Q36/Q14/A14,
   F25b/F25c, F2/Q12, C10/D25, S1/S5/S9/S33/S29, F46/F25c und der Vault-Bestand, außerhalb
   der Matrix zusätzlich der Kopf von [README](README.md) und [00 §2–§4](00_inventar.md).

   **Drei davon sind keine Nachmessung, sondern eine Bringschuld und gehören früher
   erledigt:** **N11** (drei Bibliografieeinträge — ohne Tsunakawa 2019, Kohlbrenner 2020
   und Ross et al. 2017 sind F46 und F25c nicht belegbar formulierbar), **N8** (bei einem
   `transformers`-Upgrade ist F2 zu verifizieren, *bevor* eine Heatmap neu erzeugt wird)
   und **N7** (landet die angekündigte Implementierung, ist das Register in vier Dokumenten
   nachzutragen und F25b wandert vom Ausblick in Methodik und Ergebnis).

   **Der Rest ist ein Nachtrag, kein Vorbehalt:** kein Grund, mit dem Schreiben zu warten —
   aber ein Grund, jede dieser Zahlen an **genau einer** Stelle im Beleg zu führen und von
   dort zu verweisen.
