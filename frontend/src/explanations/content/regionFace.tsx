/**
 * Explanation content — Region-Relevance (Gesichts-Schema, ganzer Clip).
 * Quellen: docs/xai_pipeline_reference.md §6.3; FaceSchematic.tsx.
 */

import type { Explanation } from '../types'
import {
  BivariateLrpNote,
  Callout,
  Chip,
  ColorScaleLegend,
  KeyValueList,
  P,
  RelevanceScaleNote,
  Term,
  UL,
} from '../ui/widgets'

export const regionFace: Explanation = {
  id: 'region-face',
  title: 'Region-Relevance (Gesichts-Schema)',
  subtitle:
    'Ein schematisches Gesicht, in sechs Regionen unterteilt, jede eingefärbt nach ihrer Relevanz über den ganzen Clip.',
  method: 'AttnLRP-Region-Aggregation · bivariate Füllung · Whole-Clip',
  cvr: 'relevance',
  sections: [
    {
      kind: 'what',
      body: (
        <>
          <P>
            Ein front-orientiertes Gesichts-Schema, aufgeteilt in{' '}
            <Term>Forehead, Left/Right Eye, Nose, Mouth, Jaw</Term>. Jede Region ist
            mit derselben bivariaten seismic-Kodierung gefüllt wie überall in der
            App: <Chip>Alpha = Relevanz-Stärke</Chip>,{' '}
            <Chip color="#ff7070">Farbe = Richtung</Chip>. Die am stärksten beachtete
            Region bekommt eine helle Kontur.
          </P>
        </>
      ),
    },
    {
      kind: 'purpose',
      body: (
        <P>
          Beantwortet <Term>„Welche Gesichtsregionen beachtet das Modell über den gesamten Clip hinweg am meisten?"</Term>{' '}.
        </P>
      ),
    },
    {
      kind: 'method',
      body: (
        <>
          <BivariateLrpNote compact unit="je anatomischer Region über den ganzen Clip" />
          <P>
            Die bivariate Heatmap jedes Frames wird auf die sechs Gesichtsregionen
            aggregiert: pro Region der Mittelwert der Relevanz. Die Füllstärke folgt
            der Höhe dieser Relevanz, die Farbe der Richtung (fake oder real).
          </P>
        </>
      ),
    },
    {
      kind: 'legend',
      body: (
        <>
          <ColorScaleLegend leftLabel="fake-lean" rightLabel="real-lean" deadZone={false} />
          <KeyValueList
            items={[
              { k: 'Alpha', v: 'Aufmerksamkeit der Region (auf die stärkste normiert).' },
              { k: 'Farbe', v: 'Verdict-Lean der Region (fake vs. real).' },
              { k: 'Helle Kontur', v: 'Die über den ganzen Clip meist-beachtete Region.' },
            ]}
          />
        </>
      ),
    },
    {
      kind: 'normalization',
      body: (
        <>
          <P>
            Die Einfärbung ist auf die stärkste Region normiert: Die am meisten
            beachtete Region füllt voll aus, die übrigen erscheinen im Verhältnis dazu
            blasser. „Voll gefüllt" heißt also „am stärksten in diesem Clip", nicht
            absolut stark. Deshalb kann eine Region voll eingefärbt sein, obwohl ihr
            Relevanzwert im Tooltip klein wirkt.
          </P>
          <P>
            Zusätzlich läuft die Füllung durch eine leichte{' '}
            <Chip>Gamma</Chip>-Kurve: Die <Chip>Alpha</Chip>-Deckkraft folgt nicht
            linear der Relevanz, sondern wird angehoben (
            <Term>Alpha = Relevanz^0,6</Term>), damit schwach beachtete Regionen
            sichtbar getönt bleiben statt ins Schwarze zu kippen. Die Farbsättigung
            ist für die wenigen großen Regionen zusätzlich leicht verstärkt. Beides
            ist reine Darstellungs-Balance — die Werte im Tooltip bleiben roh und
            unverändert.
          </P>
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
                Es gibt einen Wert je Gesichtsregion, gemittelt über den ganzen
                Clip. Der Tooltip zeigt daneben einen echten
                Prozentwert („% of total"): den Anteil der Region an der gesamten
                Aufmerksamkeit, alle Regionen zusammen 100 %.
              </>
            }
          />
          <UL>
            <li>
              Die umrandete, hellste Region → dort hat das Modell am meisten gearbeitet.
            </li>
            <li>Die Farbe der Region → ob sie fake- oder real-stützend war.</li>
            <li>Blasse Regionen → wurden kaum beachtet.</li>
          </UL>
        </>
      ),
    },
    {
      kind: 'gain',
      body: (
        <KeyValueList
          items={[
            { k: 'Fokus', v: 'Welche Gesichtsregion die Entscheidung über den Clip trägt.' },
            { k: 'Plausibilität', v: 'Ob das Modell auf sinnvolle Stellen (Mund/Augen) schaut.' },
            { k: 'Verteilung', v: 'Wie sich die Aufmerksamkeit auf die Regionen verteilt (% im Tooltip).' },
          ]}
        />
      ),
    },
    {
      kind: 'pitfalls',
      body: (
        <Callout variant="warn" title="Whole-Clip-Aggregat">
          <P>
            Zeigt den Durchschnitt über den ganzen Clip, keinen einzelnen Moment.
            Für zeitliche Details sind die Timelines da, für Vorher/Nachher der
            Attention-Shift.
          </P>
          <P>
            Die <Term>Richtung (Lean)</Term> wird über die ganze Region und alle
            Frames gemittelt. Eine nur lokal begrenzte Manipulation mittelt sich
            dabei vollständig weg: Die Farbe verblasst zu neutral, obwohl der Clip
            fake ist. Aussagekräftig bleibt dann die{' '}
            <Term>Stärke</Term> (welche Region am meisten beachtet wurde), nicht die
            Richtung. Ein wirklicher Fake/Real Lean würde nur bei end-to-end Video Fakes zum Vorschein kommen.
          </P>
        </Callout>
      ),
    },
    {
      kind: 'trust',
      body: (
        <P>
          Bei stark gedrehtem Kopf wird die Regionen-Zuordnung unzuverlässig (die
          Rotation-Warning erscheint dann).
        </P>
      ),
    },
    {
      kind: 'interaction',
      body: (
        <UL>
          <li>Über den <Chip>FACE MAP</Chip>-Tab am linken Player-Rand einblendbar.</li>
          <li><Chip>Hover</Chip> über eine Region zeigt Stärke, Anteil und Lean.</li>
        </UL>
      ),
    },
    {
      kind: 'links',
      body: (
        <UL>
          <li>
            <Term color="#a855f7">Heatmap-Overlay</Term>: die Pixel-Quelle dieser
            Regionen-Aggregation.
          </li>
          <li>
            <Term color="#a855f7">Attention-Shift (Phase 3/4)</Term>: dieselben
            Region-Scores im Vorher/Nachher-Vergleich.
          </li>
          <li>
            <Term color="#a855f7">Rotation-Warning</Term>: markiert, wann die Regionen
            unzuverlässig sind.
          </li>
        </UL>
      ),
    },
  ],
}
