# Projekt-Wissensdatenbank (Obsidian Vault)

Dieser Ordner (`vault/`) ist die gemeinsame Wissensdatenbank des Projekts
**„Unmasking Deception"** als Obsidian-Tresor. Er liegt im Code-Repository,
d.h. jedes Teammitglied bekommt ihn automatisch mit `git clone` / `git pull`.

## Einrichtung (einmalig)

1. [Obsidian](https://obsidian.md) installieren (kostenlos).
2. In Obsidian: **„Tresor öffnen" → „Ordner als Tresor öffnen"** und den
   Ordner `vault/` in diesem Repository auswählen.
3. Fertig. Die geteilte Konfiguration (`.obsidian/`) wird mitgeliefert –
   Plugins, Graph-Einstellungen und Daily-Notes sind bereits eingerichtet.

## Struktur

```
vault/
  Research/deepfake-detection/   <- die eigentliche Wissensdatenbank
    00-Hub.md                    Einstieg / Navigation
    00-Literature-Hub.md         Literatur-Einstieg
    01-Plan.md                   Projektplan
    02-Index.md                  Index (von Tooling gepflegt)
    Sources/Papers/              Paper-Notizen (eine Datei pro Paper)
    Knowledge/                   Synthese: Claim Map, Method Taxonomy, Gaps ...
    Writing/                     Schreib-Entwürfe (Related Work etc.)
    Maps/literature.canvas       Literatur-Canvas (visuelle Übersicht)
    Experiments/ Results/        Experimente und Ergebnisse
    Daily/                       Tagesnotizen (YYYY-MM-DD)
    _system/                     registry.md / schema.md / lint-report.md
```

## Git-Workflow

- **Vor dem Arbeiten:** `git pull` (aktuellen Stand holen).
- **Kleine, thematische Commits** für Notizen, getrennt von Code-Commits.
  Conventional Commits, z.B. `docs(kb): add note on PGD attack`.
- **Nach dem Arbeiten:** committen und `git push`.
- Bei parallelem Editieren derselben Notiz kann es zu Merge-Konflikten
  kommen – Markdown lässt sich aber problemlos manuell auflösen.

## Konventionen

- Verlinkung über Wikilinks: `[[Sources/Papers/videomae-tong-2022]]`.
- Neue Quellen nach `Sources/Papers/`, Synthese nach `Knowledge/`,
  Entwürfe nach `Writing/`.
- Den `_system/`-Ordner nicht von Hand editieren – der wird automatisch
  gepflegt.

## Optional: Claude-Code-Anbindung

Wer Claude Code nutzt, kann das Repo einmalig an diesen Tresor binden,
damit Claude die Wissensdatenbank automatisch pflegt:

```bash
# in Claude Code, im Repo-Root:
/kb-init --vault_path ./vault
```

Die Bindungsdatei (`.claude/project-memory/`) enthält absolute Pfade und
wird **nicht** eingecheckt – jede Person bindet ihren eigenen Klon.

## Was nicht eingecheckt wird

- `.obsidian/workspace.json` (persönliches Fenster-Layout)
- `.obsidian/cache`, `.trash/`
- `.claude/project-memory/` (maschinenspezifische Bindung)
