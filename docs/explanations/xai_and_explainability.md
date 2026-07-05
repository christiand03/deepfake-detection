# Explainable AI (xAI) – Glossar

## 1. Motivation & Grundkonzepte

### Explainable AI (xAI)

Explainable AI bezeichnet Methoden, die die Entscheidungsprozesse von Machine-Learning-Modellen für Menschen nachvollziehbar machen. Im Kontext der Deepfake-Erkennung gegen Desinformation reicht ein reines "FAKE/REAL"-Urteil nicht aus – der wissenschaftliche Beitrag dieses Projekts liegt darin zu beweisen, *welche* visuellen und akustischen Merkmale das Modell als Fälschungsindikator identifiziert. xAI ist deshalb kein nachträgliches Werkzeug, sondern der zentrale Forschungsbeitrag aller Projektphasen.

### Black-Box vs. interpretierbares Modell

Ein Black-Box-Modell produziert Vorhersagen, ohne seine interne Berechnungslogik offenzulegen; ein interpretierbares Modell erlaubt einen Einblick in die Entscheidungsgrundlage. Transformer-Modelle sind technisch Black-Boxes, obwohl ihre Attention-Gewichte prinzipiell sichtbar sind – hohe Aufmerksamkeit auf eine Region bedeutet jedoch nicht zwingend, dass diese Region kausal für das Urteil war. LRP-basierte xAI-Methoden transformieren einen Black-Box-Transformer in ein interpretierbares Modell, indem sie das Vorhersageergebnis mathematisch korrekt auf die Eingabe zurückführen.

### Saliency Map / Heatmap

Eine Saliency Map (oder Heatmap) ist eine räumliche Visualisierung, die jedem Eingabepixel einen Relevanzscore zuweist und angibt, wie stark dieses Pixel die Vorhersage des Modells beeinflusst hat. Positive Werte (typischerweise rot dargestellt) repräsentieren Evidenz *für* die vorhergesagte Klasse; negative Werte (blau) repräsentieren Evidenz *dagegen*. In diesem Projekt werden Heatmaps pro Videoframe berechnet und als eingefärbtes Overlay auf das Originalvideo gelegt, um zu zeigen, auf welche Gesichtsregionen das Modell bei seinem Urteil fokussiert war.

### Anomalie-Regionen

Die räumliche Heatmap eines Frames wird auf sechs anatomisch definierte Gesichtsregionen aggregiert: Mund, Linkes Auge, Rechtes Auge, Kiefer, Schulter und Hintergrund. Jede Region erhält einen skalaren Relevanzscore aus dem Mittelwert der absoluten LRP-Werte ihrer Pixel. Diese Regionszerlegung bildet die quantitative Grundlage für die Attention-Shift-Analyse in Phase 3 (Robustness) und Phase 4 (Adversarial).

## 2. Methoden

### Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) erzeugt klassenspezifische Saliency Maps für CNNs, indem Gradienten der Zielklasse in die letzte Faltungsschicht zurückgeführt und mit den dortigen Feature Maps gewichtet werden. Es ist die Standardmethode für konvolutionale Modelle, da sie die topographische Struktur der räumlichen Feature Maps ausnutzt. Für Transformer-Modelle ist Grad-CAM nicht direkt anwendbar, weil Transformer flache Token-Sequenzen ohne räumliche Feature Maps verwenden – weshalb LRP-basierte Methoden notwendig sind.

### Attention Rollout

Attention Rollout approximiert den Informationsfluss durch einen Transformer, indem die Attention-Gewichtsmatrizen aller Schichten sukzessive miteinander multipliziert werden, um die Aufmerksamkeit vom CLS-Token bis zu den Eingabe-Tokens zu propagieren. Es ist die rechnerisch günstigere der zwei xAI-Modi in diesem Projekt und liefert schnell eine Visualisierung, wo das Modell "hingeschaut" hat. Die Einschränkung besteht darin, dass Attention-Gewichte den Informationsfluss beschreiben, aber nicht direkt die kausale Relevanz für die finale Vorhersage messen.

### LRP (Layer-wise Relevance Propagation)

LRP ist eine xAI-Methode, die beim Vorhersage-Score des Modells beginnt und diesen unter Einhaltung von Erhaltungsregeln Schicht für Schicht rückwärts auf die Eingabe verteilt, sodass die Gesamtsumme der Relevanz erhalten bleibt. Das Ergebnis ist eine vorzeichenbehaftete Heatmap pro Eingabepixel: positive Werte zeigen Evidenz *für* die erklärte Klasse an, negative Werte Evidenz *dagegen*. LRP gilt als methodisch robuster als Attention-Rollout, weil es die tatsächliche Berechnung jeder Schicht berücksichtigt und nicht nur die Attention-Gewichte.

### AttnLRP (Achtibat et al., ICML 2024)

AttnLRP ist eine speziell für Transformer-Attention-Schichten entwickelte LRP-Variante, die das Problem adressiert, dass klassische LRP-Regeln nicht ohne Weiteres auf den Softmax-Aufmerksamkeitsmechanismus verallgemeinert werden können. Dazu wird die Input×Gradient-Formulierung an den Attention-Modulen eingesetzt, während an den übrigen Schichten (MLP, LayerNorm) standardmäßige LRP-Regeln gelten. Dieses Projekt verwendet AttnLRP als primäre xAI-Methode, angewandt sowohl auf VideoMAE (Videoheatmaps) als auch auf Wav2Vec 2.0 (Audio-Relevanz).

### Input × Gradient

Input × Gradient ist eine gradientenbasierte Attributionsmethode, bei der die Relevanz jedes Eingabefeatures als elementweises Produkt aus dem Eingabewert und seinem Gradienten bezüglich des Ziel-Logits berechnet wird: `relevance = x * ∂score/∂x`. Es handelt sich dabei um einen Spezialfall der LRP mit einer spezifischen Propagationsregel. AttnLRP verwendet diese Formulierung an den Attention-Modulen, weil sie differenzierbar ist und vorzeichenbehaftete Relevanzen liefert.

### Monkey-Patching (für LRP)

Monkey-Patching bedeutet in diesem Kontext, dass die Attention-Module eines vortrainierten Transformers zur Laufzeit durch LRP-kompatible Äquivalente ersetzt werden – ohne den originalen Modellcode oder die Gewichte zu verändern. Die Bibliothek `lxt` (LRP for Transformers) führt dieses Patching vor dem Backward-Pass durch. Dieser Ansatz ist notwendig, weil die Standard-HuggingFace-Implementierung der Attention-Berechnung nicht mit den LRP-Erhaltungsregeln kompatibel ist; ein Guard-Flag (`_VIDEOMAE_LRP_PATCHED`) stellt sicher, dass das Patching nur einmalig erfolgt.

### Attention Shift

Attention Shift misst, wie stark sich die LRP-Regionsscores zwischen zwei Bedingungen verschieben – beispielsweise zwischen einem unveränderten Video und einer degradierten oder adversariell gestörten Version. Eine Verschiebung der Relevanz von der Mundregion zum Hintergrund belegt, dass ein Angriff oder eine Degradierung die Aufmerksamkeit des Modells von der diskriminativen Gesichtsregion weggelenkt hat. Diese Metrik ist das zentrale quantitative Argument der Phase-3- und Phase-4-Analyse.

### Temporale Relevanz / Per-Frame Score

Da das Modell 16 aufeinanderfolgende Videoframes als einzelne Eingabe verarbeitet, muss die LRP-Relevanz anschließend auf einzelne Frames aufgeteilt werden, um zu bestimmen, *welcher Zeitpunkt* im Clip am verdächtigsten war. Der Relevanztensor wird kanalweise summiert und dann räumlich pro Frame gemittelt, sodass ein skalarer Per-Frame-Score entsteht. Diese zeitliche Aufschlüsselung ermöglicht die Frame-Timeline-Visualisierung im Frontend, bei der jeder Frame mit seiner individuellen Fake-Relevanz annotiert wird.

### Occlusion Sensitivity

Occlusion Sensitivity ist eine modellunabhängige Analysemethode: Ein Frame nach dem anderen wird auf den Mittelwert (oder null) gesetzt, und die Änderung der Fake-Wahrscheinlichkeit wird gemessen. Der Frame mit dem größten Einbruch in der Vorhersagekonfidenz trägt am meisten zur Erkennung bei. Diese Methode ist langsamer als LRP (16 Forward-Passes statt einem), dient aber als unabhängige Validierung der LRP-Ergebnisse.

## Weiterführende Recherche

- Achtibat, M. et al. (2024): *AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers* – ICML 2024, Grundlage der xAI-Implementierung.
- Bach, S. et al. (2015): *On Pixel-wise Explanations for Non-Linear Classifier Decisions by Layer-wise Relevance Propagation* – LRP-Originalmethodik.
- Abnar, S. & Zuidema, W. (2020): *Quantifying Attention Flow in Transformers* – Grundlage von Attention Rollout.
- Chefer, H. et al. (2021): *Transformer Interpretability Beyond Attention Visualization* – Alternative LRP-Ansätze für Transformer.
