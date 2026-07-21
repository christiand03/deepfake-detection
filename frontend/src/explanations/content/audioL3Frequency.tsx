/**
 * Explanation content — L3 Frequency × Time (Audio).
 * Quellen: docs/xai_pipeline_reference.md §7.3; FrequencyHeatmap.tsx / FrequencyBandChart.tsx.
 */

import type { Explanation } from '../types'
import {
  BivariateLrpNote,
  Callout,
  Chip,
  ColorScaleLegend,
  DeadzoneNote,
  KeyValueList,
  P,
  RelevanceScaleNote,
  Term,
  UL,
} from '../ui/widgets'

export const audioL3Frequency: Explanation = {
  id: 'audio-l3-frequency',
  title: 'L3 — Frequency × Time',
  subtitle:
    'Die Relevanz aufgeteilt in drei Frequenzbänder über die Zeit — welcher Artefakttyp, wann.',
  method: 'Band-Ablation (Confidence) · energie-gewichtete Relevanz',
  cvr: 'both',
  sections: [
    {
      kind: 'what',
      body: (
        <>
          <P>
            Ein Band × Zeit-Grid (drei Zeilen, Spalten = 0,64-s-Chunks); bei älteren
            Caches ein 3-Balken-Chart. Die Bänder:
          </P>
          <KeyValueList
            items={[
              { k: 'Low', v: '0–500 Hz — Grundfrequenz / Prosodie.' },
              { k: 'Mid', v: '500 Hz–4 kHz — Formanten / Vokale.' },
              { k: 'High', v: '4–8 kHz — Frikative / Vocoder-Artefakte.' },
            ]}
          />
        </>
      ),
    },
    {
      kind: 'purpose',
      body: (
        <P>
          Beantwortet <Term>„Welche Art von Artefakt erkennt das Modell — und
          wann?"</Term>. Ersetzt für Nicht-Audio-Experten das rohe Mel-Spektrogramm.
        </P>
      ),
    },
    {
      kind: 'method',
      body: (
        <>
          <P>
            <Term>Confidence</Term>: Band-Ablation — ein Band wird per Butterworth-
            Filter entfernt und das Modell neu bewertet; der Abfall des Fake-Margins
            ist der Score (nur auf Fake-Chunks gegated).
          </P>
          <P>
            <Term>Relevance</Term>: energie-gewichteter Mittel der bivariaten
            Dual-Seed-Relevanz je Band — unabhängig von der Lautstärke, damit das
            energiearme High-Band nicht auf ~0 kollabiert.
          </P>
          <BivariateLrpNote compact unit="je Frequenzband (nur Relevance-Sicht)" />
        </>
      ),
    },
    {
      kind: 'legend',
      body: <ColorScaleLegend />,
    },
    {
      kind: 'normalization',
      body: (
        <>
          <DeadzoneNote present={false}>
            <P>
              Die Magnitude (Deckkraft der Zellen) steht auf einer{' '}
              <Term>absoluten Skala</Term> — sie wird nicht auf 1 normiert. Lokale
              Fälschungen mitteln sich über den ganzen Clip zu schwachen Bändern, und
              genau diese Schwäche soll sichtbar bleiben, statt künstlich aufgeblasen zu
              werden. Die <Term>Confidence</Term>-Sicht ist dagegen auf das stärkste
              Band normiert und wirkt kräftiger.
            </P>
          </DeadzoneNote>
          <Callout variant="info" title="Farb-Gamma: nur der Farbton, nicht die Werte">
            <P>
              Der Farbton der Zellen entsteht aus der Richtung über eine Gamma-Kurve:{' '}
              <Chip>sign(dir) · min(0,85, |dir|^1,6 · 4)</Chip>. Das{' '}
              <Chip>Gamma 1,6</Chip> dämpft schwach gerichtete Zellen Richtung
              neutral-weiß, statt sie zwischen Rot und Blau flackern zu lassen; der{' '}
              <Chip>Gain 4</Chip> hebt klar gerichtete Zellen in kräftiges Rot/Blau, der{' '}
              <Chip>Cap 0,85</Chip> hält sie unter dem dunklen Skalen-Ende. Das wirkt
              nur auf den Farbton — die Deckkraft (Magnitude) bleibt unverändert.
            </P>
          </Callout>
        </>
      ),
    },
    {
      kind: 'interpret',
      body: (
        <>
          <RelevanceScaleNote
            frame={
              <>
                Jede Zelle zeigt die Relevanz eines Frequenzbands in einem
                Zeitfenster. Der Wert wird nicht auf einen Maximalwert von 1 normiert,
                sondern bleibt auf seiner absoluten Skala und ist deshalb meist klein.
                Eine angezeigte Zahl wie <Chip>+0,30</Chip> ist die Relevanz des
                Bandes, kein Prozentanteil des Audios.
              </>
            }
          />
          <UL>
            <li>
              Eine rote Zelle → dieses Frequenzband trug in diesem Chunk Fake-Evidenz.
            </li>
            <li>
              In der Confidence-Sicht ist ein <Term>reales</Term> Clip-Grid leer
              (keine Fake-Chunks).
            </li>
            <li>High-Band-Aktivität deutet oft auf Vocoder-/Synthese-Artefakte.</li>
          </UL>
        </>
      ),
    },
    {
      kind: 'pitfalls',
      body: (
        <Callout variant="warn" title="Leeres Grid ist kein Fehler">
          Ein leeres Confidence-Grid heißt „keine Fake-Chunks gefunden", kein Bug. Und
          die Relevance-Sicht ist von Natur aus schwach — nicht mit der kräftigeren
          Confidence-Sicht verwechseln.
        </Callout>
      ),
    },
    {
      kind: 'links',
      body: (
        <UL>
          <li>
            <Term color="#a855f7">L1 Waveform</Term>: gleiche Zeitachse, ohne
            Frequenz-Aufteilung.
          </li>
          <li>
            <Term color="#a855f7">Audio-Frequency-Shift (Phase 3/4)</Term>: dieselben
            Bänder im Vorher/Nachher.
          </li>
        </UL>
      ),
    },
  ],
}
