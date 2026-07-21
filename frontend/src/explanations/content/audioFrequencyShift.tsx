/**
 * Explanation content — Audio-Frequency-Shift (Phase 3/4).
 * Quelle: docs/xai_pipeline_reference.md §8.2; AudioFrequencyShift.tsx.
 */

import type { Explanation } from '../types'
import {
  BivariateLrpNote,
  Callout,
  Chip,
  P,
  RelevanceScaleNote,
  Term,
  UL,
} from '../ui/widgets'

export const audioFrequencyShift: Explanation = {
  id: 'audio-frequency-shift',
  title: 'Audio-Frequency-Shift',
  subtitle:
    'Vorher/Nachher der drei Audio-Bänder (Low/Mid/High) unter Audio-Kompression oder Angriff, plus ein Confidence-Delta.',
  method: 'Band-Werte before/after · dieselbe Shift-Marke wie Attention-Shift',
  cvr: 'relevance',
  sections: [
    {
      kind: 'what',
      body: (
        <P>
          Eine Confidence-Delta-Zeile für das Audio plus je ein Shift-Balken für{' '}
          <Term>Low</Term>, <Term>Mid</Term> und <Term>High</Term> — dieselbe Marke
          wie beim Attention-Shift, nur auf Frequenzbändern statt Gesichtsregionen.
        </P>
      ),
    },
    {
      kind: 'purpose',
      body: (
        <P>
          Beantwortet <Term>„Verschiebt Kompression oder Angriff die Frequenz-Evidenz
          des Audio-Modells?"</Term>.
        </P>
      ),
    },
    {
      kind: 'method',
      body: (
        <>
          <BivariateLrpNote compact unit="je Frequenzband, vorher/nachher (Relevance-Bänder)" />
          <P>
            Die bivariaten Band-Werte vor und nach der Störung werden auf
            Shift-Zeilen abgebildet: <Term>magnitude = |Wert|</Term>{' '}
            (Balkenlänge/Seite), <Term>direction = Wert</Term> (Farbe/Verdict-Lean).
          </P>
        </>
      ),
    },
    {
      kind: 'normalization',
      body: (
        <Callout variant="info" title="Feste, nicht auto-skalierte Achse">
          Länge und Farbe nutzen eine feste, nicht aus den Daten abgeleitete Skala:
          Eine Änderung der Bandstärke von ±1 bildet auf den vollen Balkenausschlag
          ab, eine Richtungs-Änderung von ±1 auf die volle Farbsättigung; größere
          Werte werden auf ±1 geklemmt. So bleiben die Balken über alle Analysen
          vergleichbar.
        </Callout>
      ),
    },
    {
      kind: 'interpret',
      body: (
        <>
          <RelevanceScaleNote
            frame={
              <>
                Die Balken zeigen eine <Term>Veränderung</Term> (vorher → nachher),
                hier je Frequenzband. Der Hover nennt den Bandwert vorher und nachher,
                der Balken stellt die Differenz dar. Die Skala ist fest: Ein voller
                Ausschlag entspricht einer Änderung von 1. Ein Bandwert wie{' '}
                <Chip>0,3</Chip> ist die Stärke des Bandes im Clip, kein Anteil am
                Audio.
              </>
            }
          />
          <UL>
            <li>Ein Balken zeigt, ob ein Band nach der Störung mehr/weniger und fake-/real-lastiger wird.</li>
            <li>Das Confidence-Delta darüber fasst den Gesamteffekt aufs Audio-Urteil zusammen.</li>
          </UL>
        </>
      ),
    },
    {
      kind: 'pitfalls',
      body: (
        <Callout variant="warn" title="Änderung, nicht Absolutwert">
          Die zentrierte Position bedeutet „keine Veränderung", nicht „keine
          Relevanz". Die absoluten Bänder zeigt Audio-Layer L3.
        </Callout>
      ),
    },
    {
      kind: 'links',
      body: (
        <UL>
          <li>
            <Term color="#a855f7">L3 Frequency</Term>: die absoluten Bänder des
            unveränderten Clips.
          </li>
          <li>
            <Term color="#a855f7">Attention-Shift</Term>: dasselbe Prinzip für die
            Video-Regionen.
          </li>
        </UL>
      ),
    },
  ],
}
