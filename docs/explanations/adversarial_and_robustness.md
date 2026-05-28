# Adversarial Attacks & Robustness – Glossar

## 1. Adversarial Attacks

### Adversarial Example

Ein Adversarial Example ist eine Eingabe, die durch eine absichtlich berechnete, für Menschen kaum wahrnehmbare Störung so modifiziert wurde, dass ein Klassifikator mit hoher Wahrscheinlichkeit eine falsche Vorhersage trifft. Im Unterschied zu zufälligem Rauschen ist die Perturbation gezielt optimiert, um die Entscheidungsgrenze des Modells zu überschreiten. In diesem Projekt werden Adversarial Examples nicht als echter Angriff eingesetzt, sondern als Analysewerkzeug, um die Schwachstellen des Detektors wissenschaftlich zu charakterisieren.

### White-Box-Angriff

Bei einem White-Box-Angriff hat der Angreifer vollständigen Zugang zu Architektur, Gewichten und Gradienten des Zielmodells. Dies ist das stärkste Bedrohungsmodell und erzeugt die wirkungsvollsten Perturbationen, da der Gradient direkt optimiert werden kann. Beide Angriffsmethoden in diesem Projekt (FGSM und PGD) sind White-Box-Angriffe, was für die wissenschaftliche Worst-Case-Analyse gewählt wurde.

### FGSM (Fast Gradient Sign Method)

FGSM ist ein einstufiger Adversarial-Angriff: Der Gradient des Verlusts bezüglich der Eingabe wird berechnet, sein Vorzeichen extrahiert und mit dem Faktor ε skaliert auf die Eingabe addiert – formal: `x_adv = x + ε · sign(∇_x L(x, y))`. Es ist der recheneffizienteste Angriff (ein Forward- plus ein Backward-Pass) und dient als Baseline für den Vergleich mit stärkeren Methoden. FGSM ist ein Sonderfall von PGD mit einem einzigen Schritt der Schrittgröße ε.

### PGD (Projected Gradient Descent)

PGD ist ein iterativer Adversarial-Angriff, der FGSM-Schritte mit kleinerer Schrittgröße mehrfach wiederholt und das Ergebnis nach jedem Schritt durch Clipping in die ε-Kugel zurückprojiziert. Durch mehrere Iterationen findet PGD stärkere adversariale Beispiele als FGSM, auf Kosten eines höheren Rechenaufwands. In diesem Projekt ist PGD mit bis zu 100 Schritten konfigurierbar und wird für die Phase-4-Evaluation eingesetzt.

### ε (Epsilon) / L∞-Kugel

Epsilon definiert die maximale erlaubte Perturbation; die L∞-Norm schränkt dies so ein, dass *kein einzelner Pixelwert* um mehr als ε verändert werden darf. Die L∞-Kugel ist die Menge aller Vektoren, die von der Originaleingabe im L∞-Sinne um höchstens ε abweichen. Kleine Werte (z. B. ε = 0.01) erzeugen für das menschliche Auge unsichtbare Störungen; größere Werte (z. B. ε = 0.1) erzeugen sichtbares Rauschen. Die Projektion in die L∞-Kugel wird durch Clipping nach jedem PGD-Schritt erzwungen.

### Fooling Rate

Die Fooling Rate ist der Anteil korrekt klassifizierter Clips, die nach einem Adversarial-Angriff ihre Vorhersage ändern. Ein Wert von 80 % bei z.B. ε = 0.05 bedeutet, dass der Angriff den Klassifikator bei 80 % der Clips täuscht, die er zuvor korrekt erkannte. Die Fooling Rate ist die primäre Metrik des Phase-4-Batch-Sweeps und wird in W&B geloggt.

### Confidence Drop

Der Confidence Drop ist die durchschnittliche Abnahme der vom Modell vorhergesagten Wahrscheinlichkeit für die ursprüngliche Klasse nach einem Angriff oder einer Degradierung. Er ist eine weichere Metrik als die Fooling Rate, da er misst, wie stark ein Angriff das Modell destabilisiert – auch wenn er die Vorhersage nicht vollständig umkehrt. Beide Metriken zusammen beschreiben die Form der adversarialen Robustheitskurve.

### Universal Adversarial Perturbation (UAP)

Eine Universal Adversarial Perturbation ist ein einziges eingabeunabhängiges Rauschbild δ*, das – wenn es zu *beliebigen* Clips des Datensatzes addiert wird – mit hoher Wahrscheinlichkeit eine Fehlklassifikation auslöst. Eine UAP wird einmalig über den gesamten Datensatz optimiert, ohne auf einen spezifischen Clip zugeschnitten zu sein. Ihr Vorhandensein wäre ein Beleg für systematische Schwachstellen im spatio-temporalen Merkmalsraum des Modells, die durch LRP-Heatmaps visualisiert werden können.

### Adversarial Training / Adversarial Fine-Tuning

Adversarial Training ist eine Verteidigungsstrategie, bei der PGD-generierte adversariale Beispiele in den Trainingsdatensatz gemischt werden, um das Modell auf gestörte Eingaben robuster zu machen. Das Modell lernt so Repräsentationen, die gegenüber gradientenbasierten Perturbationen stabiler sind – auf Kosten eines leichten Rückgangs der Clean-Accuracy. In diesem Projekt ist Adversarial Fine-Tuning als Phase-4-Erweiterung geplant: Training mit einer 1:1-Mischung aus sauberen und adversarialen Batches (ε = 0.03, 7 PGD-Schritte).

## 2. Robustness (Social-Media-Pipeline)

### H.264 / CRF (Constant Rate Factor)

H.264 ist der dominierende Videocodec, den nahezu alle sozialen Netzwerke zur Videokompression verwenden. Der Constant Rate Factor (CRF) steuert die Qualität: CRF 18 entspricht nahezu verlustfreier Kodierung, CRF 51 maximaler Kompression mit starken Artefakten. In diesem Projekt werden Videos mittels FFmpeg bei CRF-Werten zwischen 18 und 51 rekodiert, um die Kompressionsartefakte von Plattformen wie YouTube oder TikTok zu simulieren.

### AAC-Codec / Audio-Bitrate

AAC (Advanced Audio Coding) ist das Standard-Audiokompressionsformat aller gängigen Videoplattformen. Bei niedrigen Bitraten (z. B. 32 kbps) entfernt AAC hohe Frequenzanteile, was für Wav2Vec 2.0 relevante akustische Merkmale wie Stimmformanten und Zischlaute beschädigt. Die Robustheitspipeline untersucht, ob der Audioarm des Modells früher versagt als der Videoarm, wenn typische plattformübliche Audiokompression angewendet wird.

### Gaußsches Rauschen

Gaußsches Rauschen ist ein additives Rauschmodell, bei dem jedes Pixel eine unabhängige Störung aus einer Normalverteilung mit Standardabweichung σ erhält. Es simuliert Kamerasensorrauschen bei schlechten Lichtverhältnissen oder Übertragungsfehler. In der Robustheitspipeline wird es über den FFmpeg-Filter `noise=alls={σ}:allf=t+u` appliziert; der Parameter `noise_sigma` steuert die Stärke.

### Framerate-Reduktion (FPS)

Durch die Reduktion der Bildrate (z. B. von 25 fps auf 5 fps) via FFmpeg `fps`-Filter werden temporale Informationen zwischen Frames gelöscht, die für die Erkennung von Bewegungsartefakten essenziell sind. Soziale Netzwerke re-encodieren Videos häufig mit reduzierten Frameraten, um Speicher- und Bandbreitenkosten zu senken. Die Framerate ist neben CRF und Rauschen eine der drei Degradierungsachsen im Robustheitssweep.

### Social-Media-Pipeline

Die Social-Media-Pipeline bezeichnet die Abfolge verlustbehafteter Transformationen, die ein Video beim Hochladen auf eine Plattform wie TikTok, WhatsApp oder YouTube durchläuft: H.264-Videokompression, AAC-Audiokompression, mögliche Frameratenanpassung sowie Up- und Downscaling. Ziel der Phase-3-Analyse ist es, herauszufinden, ab welchem Degradierungsgrad die Deepfake-Erkennung nicht mehr zuverlässig funktioniert.

### Breaking Point

Als Breaking Point wird der Degradierungsschwellenwert bezeichnet, ab dem die Modellperformance signifikant einbricht – etwa wenn die AUC-ROC unter einen akzeptablen Grenzwert fällt. Er wird durch die Robustheitskurve identifiziert und beantwortet die zentrale Forschungsfrage von Phase 3: Wie resilient ist der Detektor unter realen Bedingungen sozialer Medien?

### Robustheitskurve

Die Robustheitskurve ist ein 2D-Plot der Modellperformance (AUC, Accuracy oder Fooling Rate) als Funktion eines Degradierungsparameters (CRF, FPS oder ε). Sie wird von den Offline-Sweep-Skripten generiert und an W&B geloggt. Die Kurve ermöglicht den direkten Vergleich zwischen Video- und Audioarm sowie zwischen Phase 3 (unbeabsichtigte Degradierung) und Phase 4 (gezielter Adversarial-Angriff).

## Weiterführende Recherche

- Goodfellow, I. et al. (2015): *Explaining and Harnessing Adversarial Examples* – FGSM-Originalpaper.
- Madry, A. et al. (2018): *Towards Deep Learning Models Resistant to Adversarial Attacks* – PGD-Originalmethodik und Adversarial Training.
- Moosavi-Dezfooli, S.-M. et al. (2017): *Universal Adversarial Perturbations* – UAP-Konzept und Algorithmus.
- Carlini, N. & Wagner, D. (2017): *Towards Evaluating the Robustness of Neural Networks* – Überblick über Angriffsmethoden und Robustheitsevaluation.
