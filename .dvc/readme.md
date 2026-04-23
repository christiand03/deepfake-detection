## Datensätze & Modelle herunterladen (mit DVC)

Dieses Projekt nutzt [DVC (Data Version Control)](https://dvc.org/), um große Datensätze (wie z.B. die Tensoren) zu verwalten. Die eigentlichen großen Dateien werden nicht in Git versioniert, sondern liegen sicher in einem freigegebenen Google Drive.

### Wichtig: Zugriffsrechte & Anmeldedaten anfragen
Da wir aus Sicherheitsgründen eine eigene Google-App für den DVC-Zugriff verwenden, musst du vor dem Start den Projekt-Administrator (Paul.Harmuth@mailbox.tu-dresden.de) kontaktieren. Du benötigst Folgendes:
1. **Lesezugriff** auf den verknüpften Google Drive-Ordner.
2. **Freischaltung deiner E-Mail:** Deine Google-Mail-Adresse muss vom Admin als "Testnutzer" für unsere App freigeschaltet werden.
3. **Google OAuth Client-ID & Secret:** Bitte den Admin um diese beiden geheimen Schlüssel für die Einrichtung.

---

### Anleitung: Daten abrufen

**1. Abhängigkeiten installieren**
Stelle sicher, dass DVC inklusive der Google Drive-Erweiterung installiert ist:
```bash
pip install -r requirements-dev.txt
# Alternativ manuell: pip install "dvc[gdrive]"
```

**2. DVC mit den Anmeldedaten konfigurieren (nur lokal!)**
Damit DVC sich bei Google authentifizieren kann, musst du die vom Admin erhaltene Client-ID und das Secret eintragen. 
*Wichtig: Durch das `--local` Flag werden diese Daten in der Datei `.dvc/config.local` gespeichert, welche von Git ignoriert wird. So landen die Schlüssel niemals aus Versehen im Code-Repository.*

Ersetze die Platzhalter (`<HIER_EINTRAGEN>`) durch die echten Werte vom Admin:
```bash
dvc remote modify --local gdrive gdrive_client_id <HIER_CLIENT_ID_EINTRAGEN>

dvc remote modify --local gdrive gdrive_client_secret <HIER_CLIENT_SECRET_EINTRAGEN>
```

**3. Daten herunterladen**
Führe nun im Hauptverzeichnis des Projekts diesen Befehl aus:
```bash
dvc pull
```

**4. Google-Authentifizierung im Browser**
Beim ersten `dvc pull` (oder `dvc push`) öffnet sich ein Link im Terminal:
- Klicke auf den Link (oder kopiere ihn in deinen Browser).
- Melde dich mit deinem **freigeschalteten Google-Konto** an.
- Bestätige den Zugriff (sollte eine Warnung kommen, dass die App nicht verifiziert ist, klicke auf *Erweitert* -> *Trotzdem weiter*).
- Kopiere den angezeigten Code zurück in das Terminal.

DVC lädt nun automatisch den `data/` Ordner herunter, der exakt zum aktuellen Code-Stand deines Git-Branches passt!

---

### Eigene Datenänderungen hochladen
Wenn du selbst Dateien im `data/` Ordner hinzufügst, löschst oder änderst, halte dich an diesen Workflow:

```bash
# 1. DVC mitteilen, die neuen Daten zu scannen
dvc add data

# 2. Die DVC-Metadatei (Lieferschein) in Git committen
git add data.dvc
git commit -m "feat: Neue Datensätze hinzugefügt"

# 3. Physische Daten ins Google Drive hochladen
dvc push

# 4. Git-Commit auf GitHub/GitLab pushen
git push
```
```
