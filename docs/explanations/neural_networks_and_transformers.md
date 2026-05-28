# Transformer-Architekturen & Modelle – Glossar

## 1. Transformer-Grundlagen

### Self-Attention / Scaled Dot-Product Attention

Self-Attention ist der zentrale Mechanismus des Transformer-Modells: Für jedes Token einer Sequenz werden Aufmerksamkeitswerte gegenüber allen anderen Tokens berechnet, sodass das Modell relevante Kontexte unabhängig von ihrer Position in der Sequenz erfassen kann. Die Berechnung erfolgt über drei Projektionen – Query (Q), Key (K) und Value (V) –, wobei die Ähnlichkeit zwischen Q und K das Gewicht bestimmt, mit dem V aggregiert wird: `Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V`. Im Gegensatz zu Faltungsschichten hat Self-Attention ein globales rezeptives Feld – jedes Token "sieht" jedes andere Token in einem einzigen Schritt.

### Multi-Head Attention

Multi-Head Attention führt mehrere parallele Self-Attention-Berechnungen ("Heads") in reduzierten Unterräumen durch und konkateniert deren Ausgaben. Jeder Head kann dabei unterschiedliche Aspekte der Eingabe erfassen, z. B. lokale Bewegungsmuster in einem Head und globale Gesichtsstruktur in einem anderen. In der Cross-Attention Fusion dieses Projekts werden 8 Heads parallel eingesetzt.

### Token / Patch

In Transformer-basierten Bildmodellen wird ein Bild in gleichmäßige räumliche Regionen (Patches) unterteilt, die jeweils durch eine lineare Projektion in einen Vektor (Token) umgewandelt werden. Das Modell verarbeitet diese Token-Sequenz analog zu Wörtern in einem Sprachmodell. Für VideoMAE entspricht ein Token einem 2×16×16-Pixel-Block über zwei aufeinanderfolgende Frames (Tubelet).

### Positional Encoding

Da der Transformer-Mechanismus keine inhärente Reihenfolge kennt, werden den Token-Vektoren Positionskodierungen addiert, die räumliche und zeitliche Ordnungsinformationen einbetten. Ohne Positional Encoding würde das Modell eine vertauschte Token-Sequenz identisch zu der unveränderten behandeln. In Videomodellen spannen Positional Encodings drei Dimensionen auf: Zeit, Höhe und Breite.

## 2. Backbone-Modelle

### VideoMAE (Video Masked Autoencoder)

VideoMAE ist ein selbst-supervisioniert vortrainierter Video-Transformer, der während des Vortrainings ~90 % der spatio-temporalen Patches maskiert und lernt, die fehlenden Inhalte aus den verbleibenden 10 % zu rekonstruieren. Dieses Pretraining-Ziel zwingt das Modell, tiefe visuelle Repräsentationen zu erlernen, anstatt sich auf oberflächliche statistische Abkürzungen zu stützen. In diesem Projekt wird der vortrainierte VideoMAE-Encoder (`MCG-NJU/videomae-base`) als visuelles Backbone verwendet und für die binäre Deepfake-Klassifikation feinabgestimmt.

### Tubelet Embedding

Ein Tubelet ist die spatio-temporale Patch-Einheit von VideoMAE: ein 2-Frame × 16×16-Pixel großer Block, der direkt aus dem Videovolumen extrahiert und linear in einen Token-Vektor eingebettet wird. Diese 3D-Einheit kodiert implizit Bewegungsinformationen zwischen zwei aufeinanderfolgenden Frames, was effizienter ist als die getrennte Verarbeitung jedes Frames. Bei 16 Frames und `tubelet_size=2` entstehen 8 temporale Schichten mit jeweils 196 räumlichen Tokens (14×14 Patches pro Frame bei 224×224 Pixeln).

### Wav2Vec 2.0

Wav2Vec 2.0 ist ein selbst-supervisionierter Audio-Transformer, der rohe Audiowaveformen ohne handkodierte Merkmale wie MFCCs oder Spektrogramme direkt verarbeitet. Ein CNN-Frontend konvertiert die Waveform zunächst in eine latente Repräsentation, die ein Transformer-Stack anschließend zu kontextuellen Merkmalsvektoren weiterverarbeitet. Das vortrainierte Modell (`facebook/wav2vec2-base`) wird als Audio-Backbone eingesetzt; das CNN-Frontend bleibt dabei eingefroren.

## 3. Fusionsarchitektur

### Cross-Attention Fusion

Cross-Attention Fusion verbindet zwei Modalitäten, indem die Tokens einer Modalität als Queries über die Tokens der anderen Modalität (als Keys und Values) attenden. In diesem Projekt wird die Fusion bidirektional durchgeführt: Video-Tokens fragen Audio-Tokens ab (Video→Audio) und umgekehrt (Audio→Video), sodass das Modell zeitliche Inkonsistenzen zwischen Lippenbewegungen und Ton erkennen kann. Die konkatenierte Ausgabe beider Richtungen wird durch Mean Pooling auf einen Clip-Level-Vektor reduziert, der anschließend dem Klassifikationskopf zugeführt wird.

### Backbone / Feature Extractor

Ein Backbone ist ein großes, vortrainiertes Modell, das als Merkmalsextraktor in einer nachgelagerten Aufgabe eingesetzt wird. Dabei werden die vorgelernten Gewichte entweder eingefroren (frozen) oder feinabgestimmt (fine-tuned), je nach verfügbarer Datenmenge und Ähnlichkeit der Aufgaben. In diesem Projekt liefern VideoMAE und Wav2Vec 2.0 die modalen Repräsentationen, während nur der `CrossAttentionFusion`-Head in der ersten Trainingsphase optimiert wird.

### Transfer Learning & Fine-Tuning

Transfer Learning überträgt Wissen aus einem großen Vortraining (z. B. allgemeines Videoverstehen) auf eine verwandte Zielaufgabe (Deepfake-Erkennung), indem die vortrainierten Gewichte als Startpunkt genutzt werden. Fine-Tuning aktualisiert diese Gewichte dann auf den Zieldaten, typischerweise mit einer deutlich niedrigeren Lernrate, um die erlernten Repräsentationen zu erhalten. Dieses Projekt verwendet einen zweistufigen Ablauf: zunächst wird nur der Fusionskopf trainiert (eingefrorene Backbones), danach werden alle Parameter gemeinsam optimiert.

### Frozen Backbone

Ein eingefrorenes (frozen) Backbone hat `requires_grad=False` für alle seine Parameter gesetzt, sodass diese beim Training nicht aktualisiert werden. Dies stabilisiert das Training in der ersten Phase, wenn der Fusionskopf noch nicht auf die Backbone-Ausgaben abgestimmt ist. Sobald der Fusionskopf konvergiert hat, werden die Backbones sukzessive freigegeben und end-to-end feinabgestimmt.

### HuggingFace Transformers

HuggingFace Transformers ist eine Python-Bibliothek mit vortrainierten Checkpoint-Archiven und standardisierten APIs für hunderte Transformer-Modelle. VideoMAE (`MCG-NJU/videomae-base`) und Wav2Vec 2.0 (`facebook/wav2vec2-base`) werden über `from_pretrained()` geladen und nutzen die einheitliche `AutoModel`-Schnittstelle. Die Bibliothek stellt auch die notwendigen Feature Extractor und Konfigurationsobjekte bereit.

## Weiterführende Recherche

- Vaswani, A. et al. (2017): *Attention Is All You Need* – Originalpaper des Transformer-Mechanismus.
- Tong, Z. et al. (2022): *VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training* – VideoMAE-Pretraining-Methodik.
- Baevski, A. et al. (2020): *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations* – Wav2Vec 2.0 Originalpaper.
- Dosovitskiy, A. et al. (2021): *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* – ViT-Grundlage für Bild-Transformer.
