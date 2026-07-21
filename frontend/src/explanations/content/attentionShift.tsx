/**
 * Explanation content — Attention-Shift-Table (Phase 3/4).
 * Quelle: docs/xai_pipeline_reference.md §8.1; AttentionShiftTable.tsx.
 */

import type { Explanation } from '../types'
import {
  BivariateLrpNote,
  Callout,
  Chip,
  KeyValueList,
  P,
  RelevanceScaleNote,
  Term,
  UL,
} from '../ui/widgets'

export const attentionShift: Explanation = {
  id: 'attention-shift',
  title: 'Attention-Shift',
  subtitle:
    'Pro Region ein Balken, der die Veränderung der Aufmerksamkeit vor → nach einer Störung vergleicht.',
  method: 'LRP-Region-Scores · Vorher/Nachher · feste Skala',
  cvr: 'relevance',
  sections: [
    {
      kind: 'what',
      body: (
        <>
          <P>
            Eine Zeile je Region mit zwei Punkten: <Term>● before</Term> in der Mitte,{' '}
            <Term>● after</Term> an der Balkenspitze. Der Balken trägt zwei Kanäle:
          </P>
          <KeyValueList
            items={[
              {
                k: 'Länge + Seite',
                v: 'Änderung der Aufmerksamkeit (magnitude). Links = weniger, rechts = mehr als vorher.',
              },
              {
                k: 'Farbe',
                v: 'Änderung der Richtung: rot = Richtung fake, blau = Richtung real.',
              },
            ]}
          />
        </>
      ),
    },
    {
      kind: 'purpose',
      body: (
        <P>
          Beantwortet <Term>„Wie verschiebt eine Degradation oder ein Angriff, wohin
          das Modell schaut und wie viel sich die Aufmerksamkeit verändert?"</Term>.
        </P>
      ),
    },
    {
      kind: 'method',
      body: (
        <>
          <BivariateLrpNote compact unit="je Region, vorher/nachher (magnitude & direction)" />
          <P>
            Die Region-Scores stammen aus derselben bivariaten Aggregation wie das
            Gesichts-Schema, einmal auf der Clean- und einmal auf der gestörten
            Heatmap. Angezeigt wird die Differenz (nachher − vorher) beider Kanäle.
          </P>
        </>
      ),
    },
    {
      kind: 'normalization',
      body: (
        <Callout variant="info" title="Feste, nicht auto-skalierte Achse">
          Länge und Farbe nutzen eine feste, nicht aus den Daten abgeleitete Skala:
          Eine Magnitude-Änderung von ±1 bildet auf den vollen Balkenausschlag ab,
          eine Richtungs-Änderung von ±1 auf die volle Farbsättigung; größere Werte
          werden auf ±1 geklemmt. So bleiben die Balken über alle Analysen
          vergleichbar, und ein Diagramm winziger Änderungen wird nicht künstlich groß
          gezogen.
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
                keinen Absolutwert. Der Hover nennt die Relevanz der Region vorher und
                nachher (z. B. <Chip>0,21 → 0,35</Chip>), der Balken stellt die
                Differenz dar. Die Skala ist fest: Ein voller Ausschlag entspricht
                einer Änderung von 1.
              </>
            }
          />
          <UL>
            <li>Langer Balken nach rechts + rot → die Region hat nach der Störung fake-Aufmerksamkeit gewonnen.</li>
            <li>Kurzer Balken → kaum Veränderung.</li>
            <li>Regionen sind nach Größe der Änderung sortiert (oben = größte).</li>
          </UL>
        </>
      ),
    },
    {
      kind: 'pitfalls',
      body: (
        <Callout variant="warn" title="Zeigt Änderung, nicht Absolutwert">
          Ein zentrierter Punkt heißt „keine Veränderung", nicht „keine Relevanz".
          Für die absoluten Region-Werte ist das Gesichts-Schema da. Bei starker Rotation
          muss die Rotation-Warning mit beachtet werden.
        </Callout>
      ),
    },
    {
      kind: 'interaction',
      body: <P><Chip>Hover</Chip> zeigt die exakten before → after → Δ-Werte beider Kanäle.</P>,
    },
    {
      kind: 'links',
      body: (
        <UL>
          <li>
            <Term color="#a855f7">Region-Relevance</Term>: die absolute Quelle dieser
            Änderungen.
          </li>
          <li>
            <Term color="#a855f7">Crop-Comparison</Term>: dieselbe Verschiebung als
            Bild.
          </li>
          <li>
            <Term color="#a855f7">Audio-Frequency-Shift</Term>: dasselbe Prinzip für
            die Audio-Bänder.
          </li>
        </UL>
      ),
    },
  ],
}
