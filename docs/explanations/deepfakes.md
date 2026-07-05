# Deepfake-Technologien – Glossar

## 1. Deepfake-Varianten

### Talking-Head Deepfake

Ein Talking-Head Deepfake bezeichnet eine KI-generierte Videomanipulation, bei der ausschließlich Kopf und Gesicht einer sprechenden Person gefälscht werden, während der Körper und der Hintergrund unverändert bleiben. Im Mittelpunkt stehen dabei Lippenbewegungen, Gesichtsausdruck und Augenbewegungen, die so generiert werden, dass sie zu einem manipulierten Audiotrack passen. Dieses Szenario bildet das primäre Bedrohungsmodell dieses Projekts, da Talking-Head-Videos besonders anfällig für gezielte Falschinformationskampagnen sind.

### Lip-Sync Deepfake (Audio-Driven)

Ein Lip-Sync Deepfake ersetzt den Mundbereich einer Zielperson durch eine audio-gesteuerte Gesichtssynthese – die generierten Lippenbewegungen sind präzise auf einen anderen (manipulierten) Audiotrack abgestimmt. Die Manipulation beschränkt sich zeitlich auf die Mundregion, während andere Gesichtsmerkmale wie Augen und Stirn intakt bleiben. Dieser Fälschungstyp ist besonders schwer zu erkennen, weil er bei schnellem Ansehen kaum auffällt – er ist jedoch über die Cross-Attention-Analyse der Mundregion detektierbar.

### Face-Swap vs. Reenactment

Face-Swap Deepfakes ersetzen die gesamte Identität einer Person in einem Video durch das Gesicht einer anderen Person; Reenactment Deepfakes hingegen steuern Mimik und Lippenbewegungen der Zielperson über ein Driving Signal, ohne die Identität zu ändern. Beide Varianten hinterlassen charakteristische Artefakte an den Übergangsbereichen – entweder an der Gesichtskontur (Face-Swap) oder am Mundrand (Reenactment). Das verwendete Datensatz AV-Deepfake1M enthält beide Manipulationstypen, jeweils mit separaten Labels für Video- und Audiofälschung.

### GAN (Generative Adversarial Network)

Ein GAN besteht aus zwei konkurrierenden neuronalen Netzen: einem Generator, der synthetische Inhalte erzeugt, und einem Diskriminator, der echte von gefälschten Beispielen unterscheiden soll. Durch dieses adversariale Training verbessert der Generator kontinuierlich die Qualität seiner Ausgaben, bis der Diskriminator sie nicht mehr zuverlässig erkennt. Die meisten modernen Deepfake-Systeme nutzen GAN-Varianten (z. B. StarGAN, First Order Motion Model), obwohl diffusionsbasierte Methoden zunehmend an Bedeutung gewinnen.

### Biometrische Konsistenz / Audio-Visual Synchrony

Ein Gesicht, das spricht, erzeugt normalerweise eine hochpräzise Korrelation zwischen Lippenbewegungen und akustischen Phonemen – diese sogenannte Audio-Visual Synchrony ist ein evolutionär erlerntes biometrisches Merkmal. Deepfake-Generierungsmodelle können diese Synchronität häufig nicht perfekt reproduzieren und hinterlassen messbare Inkonsistenzen, die als Detektionsartefakte genutzt werden. Das multimodale Modell dieses Projekts nutzt Cross-Attention explizit dazu, Abweichungen zwischen Audio- und Video-Token zu erkennen, die auf eine solche Inkonsistenz hinweisen.

## 2. Datensatz

### AV-Deepfake1M

AV-Deepfake1M ist der primäre Trainingsdatensatz dieses Projekts mit über einer Million Videosegmenten, die auf realen Interviews und Reden basieren und teilweise mit KI-Methoden manipuliert wurden. Jedes Segment ist durch eine JSON-Sidecar-Datei mit den Feldern `label_video`, `label_audio` und `modify_type` annotiert, was eine differenzierte Unterscheidung zwischen Video-Only-, Audio-Only- und kombinierten AV-Fälschungen ermöglicht. Die Daten sind nach Identitäten strukturiert, sodass jede Person exklusiv in einem der Splits (Train, Val, Test) vorkommt und Identity-Leakage verhindert wird.

## Weiterführende Recherche

- Wang, S.-Y. et al. (2020): *CNN-generated images are surprisingly easy to spot… for now* – frühe Analyse von GAN-Artefakten.
- Rossler, A. et al. (2019): *FaceForensics++: Learning to Detect Manipulated Facial Images* – Benchmark für Gesichtsfälschungen.
- Cai, Z. et al. (2023): *AV-Deepfake1M: A Large-Scale LLM-Driven Audio-Visual Deepfake Dataset* – Beschreibung des verwendeten Datensatzes.
- Tolosana, R. et al. (2020): *Deepfakes and Beyond: A Survey of Face Manipulation and Fake Detection* – umfassende Übersicht.
