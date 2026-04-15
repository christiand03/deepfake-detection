# Frontend & xAI Visualisierung GUI

Während der Fokus massiv auf der Backend-ML-Entwicklung liegt, bietet eine saubere GUI einen exzellenten Mehrwert (Demo-Charakter, Interaktion in Prüfungs-Präsentationen).

## 1. Technologiestack Frontend
- **Framework:** `React` gepaart mit `TypeScript` (Für saubere Datenströme der Tensor-Metadaten).
- **Bundler:** `Vite` (SOTA im Vergleich zum veralteten Webpack).
- **Styling:** `TailwindCSS` für rasantes UI-Prototyping.

## 2. Backend / API Schnittstelle
Das PyTorch/Python-Modell muss an die React-Application angebunden werden.
- **Framework:** `FastAPI`. 
- **Verarbeitung:** FastAPI stellt REST-Endpoints zur Verfügung, akzeptiert kleine Video-Uploads, leitet diese durch eine Inference-Pipeline in den geladenen Spatio-Temporal-Head und liefert als JSON:
  - `Confidence Score` (Fake / Real).
  - `Base64 encodiertes Bild` der gerenderten LRP-Heatmap.
  - `Metadaten` zu gefundenen Anomalien in Audio-Synchronität.

## 3. Die Visualisierungs-Dashboarding-Elemente
Im React-Frontend ist der visuelle Informationsfluss entscheidend:
- **Videoplayer:** Mit Overlay-Toggle (Einblenden der Attention-Heatmaps framegenau).
- **Interaktive Graphen:** Nutzen von `Plotly.js` oder `D3.js` um die Synchronisations-Verweildauer zwischen Audio und Lip-Sync darzustellen.
- **Kontext-Display:** Erklärende Textboxen basierend auf xAI-Output (z.B. "Die Inkonsistenz auf Frame 42 (Mundbereich) hat signifikant (LRP +4.2) zum Fake-Label beigetragen").

## Weiterführende Recherche
- "Deploying PyTorch models with FastAPI and React"
- "Vite + React + Tailwind Setup"
- "Video Overlay with Canvas in React"
