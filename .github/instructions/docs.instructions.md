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
- Verweise auf Code-Dateien: `Implementierung in src/models/istvt.py`.

## Datei-Übersicht
| Datei | Thema | Aktualisieren bei... |
|---|---|---|
| `project.md` | Projektüberblick, Phasen, Team | Änderung der Methodik/Phasen |
| `datasets.md` | Datensätze, Preprocessing, QA | Neuer Datensatz, Pipeline-Änderung |
| `tech.md` | Tools, Hardware, Projektstruktur | Neues Tool, Strukturänderung |
| `model.md` | Architekturen, Training, Ablation | Modell-Änderung, neue Experimente |
| `xai.md` | xAI-Methoden, Plotting | Neue Visualisierung, Plot-Style |
| `mlops.md` | W&B, DVC, Lightning, CI/CD | Infrastruktur-Änderung |
| `code_quality.md` | Testing, Linting, Workflow | Neue Test-Patterns, Tool-Wechsel |
| `frontend.md` | React GUI, FastAPI | Frontend-Entwicklung |
| `adversarial.md` | FGSM/PGD, Robustness | Phase 3/4 Implementierung |
| `todo.md` | Meilensteine, Tasks | Regelmäßig (Task abgeschlossen) |

## Formatierung
- **Tabellen** für Vergleiche und Übersichten nutzen.
- **Code-Blöcke** mit Sprachkennung (` ```python `, ` ```yaml `, ` ```text `).
- **Fett** für wichtige Begriffe und Warnungen.
- **Kursiv** für Erklärungen und Nebenbemerkungen.
- Bullet-Listen für Aufzählungen, nummerierte Listen für sequentielle Schritte.
