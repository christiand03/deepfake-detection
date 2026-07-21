/**
 * Explanation content — L1 Waveform (Audio).
 * Quellen: docs/xai_pipeline_reference.md §3.4, §7.1; WaveformRelevanceLayer.tsx.
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

export const audioL1Waveform: Explanation = {
  id: 'audio-l1-waveform',
  title: 'L1 — Waveform',
  subtitle:
    'Die Audio-Wellenform als grauer Hintergrund, darüber ein farbiges Relevance/Confidence-Band entlang der Zeit.',
  method: 'Wav2Vec 2.0 · AttnLRP Dual-Seed · bivariat pro 0,64-s-Chunk',
  cvr: 'both',
  sections: [
    {
      kind: 'what',
      body: (
        <P>
          Unten ist die graue Waveform des Audiosignals als Zeitreferenz, darüber
          ein farbiges Band, das die Relevanz (bzw. Confidence, je nach Toggle) über
          die Zeit anzeigt. Ein cyan Playhead läuft mit.
        </P>
      ),
    },
    {
      kind: 'purpose',
      body: (
        <P>
          Beantwortet <Term>„WANN im Audio reagiert das Modell?"</Term> — das
          akustische Gegenstück zur Video-Heatmap, mit Sekunden als Achse.
        </P>
      ),
    },
    {
      kind: 'method',
      body: (
        <>
          <BivariateLrpNote unit="pro 0,64-s-Chunk (10240 Samples @ 16 kHz, eine Modellentscheidung)" />
          <P>
            Für Audio werden beide Seeds an der{' '}
            <Term>CNN→Transformer-Grenze</Term> von Wav2Vec 2.0 berechnet (nicht am
            Roh-Wave). Magnitude und Direction werden pro{' '}
            <Chip>0,64-s-Chunk</Chip> clip-global gemittelt; eine Hue-Schärfung
            dämpft schwach gerichtete Chunks, statt sie flackern zu lassen.
          </P>
        </>
      ),
    },
    {
      kind: 'legend',
      body: (
        <>
          <ColorScaleLegend />
          <KeyValueList
            items={[
              { k: 'Farbe', v: 'Richtung: rot = fake-, blau = real-stützend.' },
              { k: 'Deckkraft', v: 'Stärke des Signals im Chunk.' },
              { k: 'Graue Kurve', v: 'Lautstärke-Waveform als Zeitreferenz.' },
            ]}
          />
        </>
      ),
    },
    {
      kind: 'normalization',
      body: (
        <DeadzoneNote present>
          <P>
            Bei L1 wirkt die Dead-Zone nur auf die <Term>Farbe</Term>, nicht auf die
            Deckkraft. Der Farbton entsteht aus der Richtung über eine Gamma-Kurve:{' '}
            <Chip>sign(d) · min(0,85, |d|^1,5 · 4,0)</Chip>. Das <Chip>Gamma 1,5</Chip>{' '}
            staucht kleine Richtungswerte — schwach gerichtete Chunks rutschen Richtung
            null und bleiben neutral-weiß, statt zwischen blassem Rot und Blau zu
            flackern. Der <Chip>Gain 4,0</Chip> hebt dagegen klar gerichtete Chunks in
            kräftiges Rot/Blau, der <Chip>Cap 0,85</Chip> hält sie unter dem dunklen
            Skalen-Ende. Die <Term>Deckkraft</Term> bleibt davon unberührt: Sie zeigt
            weiterhin die echte, unveränderte Magnitude (Engagement).
          </P>
        </DeadzoneNote>
      ),
    },
    {
      kind: 'resolution',
      body: (
        <P>
          Ein Farb-Chunk ≈ 0,64 s — dieselbe X-Achse wie die L3-Frequenz-Heatmap.
        </P>
      ),
    },
    {
      kind: 'interpret',
      body: (
        <>
          <RelevanceScaleNote
            frame={
              <>
                L1 zeigt keinen Zahlenwert. Die Relevanz steckt in der Deckkraft des
                Farbbands: Je kräftiger ein 0,64-Sekunden-Abschnitt eingefärbt ist,
                desto stärker hat das Modell dort gearbeitet. Die Farbe zeigt die
                Richtung (rot = fake, blau = real).
              </>
            }
          />
          <UL>
            <li>Farbige Abschnitte → hier reagiert das Modell; die Farbe sagt wohin.</li>
            <li>Blasse Abschnitte → kaum Engagement.</li>
            <li>Die graue Kurve nur zur zeitlichen Orientierung nutzen.</li>
          </UL>
        </>
      ),
    },
    {
      kind: 'pitfalls',
      body: (
        <Callout variant="warn" title="Chunk-Aggregat">
          Die Farbe ist der Mittelwert über einen 0,64-s-Chunk, keine einzelne
          Sample-Aussage. Nahe-neutrale Stellen in der Relevanzansicht sind nicht zwingend „echt", es ist
          lediglich die Abwesenheit einer eindeutigen Manipulation.
        </Callout>
      ),
    },
    {
      kind: 'links',
      body: (
        <UL>
          <li>
            <Term color="#a855f7">L2 Word-Tokens</Term>: dieselbe Relevanz, aber pro
            gesprochenem Wort.
          </li>
          <li>
            <Term color="#a855f7">L3 Frequency</Term>: dieselbe Zeitachse, aufgeteilt
            nach Frequenzbändern.
          </li>
          <li>
            <Term color="#a855f7">Heatmap-Overlay</Term>: das visuelle Gegenstück.
          </li>
        </UL>
      ),
    },
  ],
}
