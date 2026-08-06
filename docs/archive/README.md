# Archiv — Planungs-Dokumente (Stand April/Mai 2026)

Diese Dateien stammen aus der **Planungsphase** des Projekts. Ihr noch gültiger
Inhalt wurde in die aktuellen Dokumente unter [`../`](../) überführt; sie bleiben
hier zu Referenz- und Nachvollziehbarkeitszwecken **unverändert** erhalten.

> ⚠️ **Nicht als aktuelle Quelle verwenden.** Wo Planung und umgesetzte Realität
> abweichen (z. B. ISTVT → VideoMAE, "16 GB VRAM Mindestanforderung",
> `src/`-Struktur, nicht existierende `plot_style.py`), gelten die aktuellen
> Dokumente.

> 🔴 **Diese Warnung ist nachweislich zu spät gekommen** (geprüft 2026-08-06). Vier
> Aussagen aus diesem Verzeichnis sind in die Belegarbeit gelangt und dort falsch:
>
> | Archivstelle | Aussage | Stand |
> |---|---|---|
> | `xai.md` §1 | Attention Rollout als "Lösung 1" | **nie implementiert** — Widerspruch F57, im Beleg an fünf Stellen behauptet |
> | `xai.md` §2 | Attention "verschiebt sich präzise auf den Mund" (über 50 Epochen) | **gemessen und widerlegt**; Training läuft über 30 Epochen |
> | `xai.md` §3 | Relevanz je Wort "aufsummiert" | Code **mittelt** vorzeichenbehaftet — Widerspruch F14, korrigiert |
> | `adversarial.md` §2.1 | Regionen "Mouth, Eye, Jaw, Shoulder, Background" | `Shoulder`/`Background` **existieren nicht** — Widerspruch F18 |
>
> Alle vier Stellen tragen inzwischen einen Korrekturkasten. **Wer hier etwas
> nachschlägt, prüft es vorher gegen den Code oder gegen
> `docs/vollstaendigkeitsliste/`.** Die Dateien bleiben zur Nachvollziehbarkeit stehen;
> ihre inhaltlichen Aussagen sind ohne Gegenprobe wertlos.

| Archiv-Datei | Ersetzt durch | Anmerkung |
|---|---|---|
| `project.md` | [`../project.md`](../project.md) | Status auf Juni 2026 aktualisiert; Phase-3/4-Roadmap integriert |
| `adversarial.md` | [`../project.md`](../project.md) §5, [`../commands.md`](../commands.md) §7 | Phase-3/4-Detail & Priorisierung übernommen, Sweeps umgesetzt |
| `xai.md` | [`../xai.md`](../xai.md) | AttnLRP als implementierte Methode ergänzt; ISTVT → VideoMAE |
| `tech.md` | [`../engineering.md`](../engineering.md) §1–3 | Struktur/Hardware an Realität angepasst |
| `mlops.md` | [`../engineering.md`](../engineering.md) §5, [`../launch.md`](../launch.md) | — |
| `code_quality.md` | [`../engineering.md`](../engineering.md) §4 | — |
| `frontend.md` | [`../engineering.md`](../engineering.md) §6, [`../commands.md`](../commands.md) §9 | — |
| `dataset_links.md` | [`../datasets.md`](../datasets.md) | Kandidaten-Liste; finale Auswahl AV-Deepfake1M + SWAN-DF |
| `todo.md` | [`../project.md`](../project.md) §4–5 | Veralteter Meilenstein-Tracker |
