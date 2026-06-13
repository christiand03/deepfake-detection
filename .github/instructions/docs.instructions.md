# docs/ – Dokumentationsstil & Konventionen

## Sprache
- **Dokumentation in docs/ wird auf Deutsch verfasst** (Belegarbeit ist auf Deutsch).
- Fachbegriffe (z.B. "Cross-Modal Attention", "Layer-wise Relevance Propagation") bleiben auf Englisch.
- Code-Beispiele in Docs bleiben auf Englisch (konsistent mit src/).

## Struktur jeder Markdown-Datei
Jede Datei in docs/ folgt einem einheitlichen Aufbau:
1. **Überschrift (H1)** – Klarer Titel des Themenbereichs
2. **Nummerierte Abschnitte (H2)** – Logische Gliederung mit `## 1.`, `## 2.` etc.
3. **Unterabschnitte (H3)** – Für Details innerhalb eines Abschnitts
4. **Weiterführende Recherche** – Am Ende jeder Datei: Stichworte, Paper-Titel, Tool-Namen für Vertiefung

## Querverweise
- Verweise auf andere docs/-Dateien im Format: `Siehe docs/model.md` oder `Details in docs/xai.md`.
- Verweise auf Code-Dateien: `Implementierung in src/models/VideoMAE_module.py`.

## Datei-Übersicht
Einstiegspunkt: `README.md` (Navigationsübersicht). Frühere Planungs-Docs unter
`archive/` werden **nicht** weiter gepflegt.

| Datei | Thema | Aktualisieren bei... |
|---|---|---|
| `README.md` | Navigationsindex aller Docs | Neue/umbenannte Doc-Datei |
| `project.md` | Überblick, Phasen, Team, Status, Roadmap | Änderung der Methodik/Phasen/Status |
| `engineering.md` | Tools, Hardware, Struktur, MLOps, Testing, Frontend | Neues Tool, Strukturänderung, Test-Pattern |
| `datasets.md` | Datensätze, Preprocessing, QA | Neuer Datensatz, Pipeline-Änderung |
| `model.md` | Architekturen, Training, Ablation | Modell-Änderung, neue Experimente |
| `xai.md` | xAI-Methoden, Plotting | Neue Visualisierung, Plot-Style |
| `commands.md` | Befehls-Referenz | Neuer/geänderter Befehl oder Flag |
| `performance_roadmap.md` | Umgesetzte Performance-Features | Neues SOTA-Feature |
| `launch.md` | W&B Launch (Queue/Agent) | Infrastruktur-Änderung |
| `audit_2026-06.md` | Silent-Failure-Audit | Neuer Audit-Befund |
| `explanations/` | Glossar | Neuer Fachbegriff |

## Formatierung
- **Tabellen** für Vergleiche und Übersichten nutzen.
- **Code-Blöcke** mit Sprachkennung (` ```python `, ` ```yaml `, ` ```text `).
- **Fett** für wichtige Begriffe und Warnungen.
- **Kursiv** für Erklärungen und Nebenbemerkungen.
- Bullet-Listen für Aufzählungen, nummerierte Listen für sequentielle Schritte.
