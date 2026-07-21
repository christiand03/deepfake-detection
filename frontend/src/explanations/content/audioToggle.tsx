/**
 * Explanation content — Relevance/Confidence-Toggle (Audio-Panel).
 * Quellen: docs/xai_pipeline_reference.md §7; AudioLayers.tsx (B4).
 */

import type { Explanation } from '../types'
import { BivariateLrpNote, Callout, Chip, KeyValueList, P, Term, UL } from '../ui/widgets'

export const audioToggle: Explanation = {
  id: 'audio-toggle',
  title: 'Relevance / Confidence-Toggle',
  subtitle:
    'Ein Schalter, der alle drei Audio-Layer gemeinsam zwischen Relevance und Confidence umstellt.',
  method: 'einmal berechnet · clientseitig umschaltbar',
  cvr: 'both',
  sections: [
    {
      kind: 'what',
      body: (
        <P>
          Ein panel-weiter Umschalter im Kopf des Audio-Panels. Er wechselt L1, L2
          und L3 <Term>gleichzeitig</Term> zwischen zwei komplementären Sichten.
        </P>
      ),
    },
    {
      kind: 'purpose',
      body: (
        <>
          <P>Die beiden Sichten beantworten verschiedene Fragen:</P>
          <KeyValueList
            items={[
              {
                k: 'RELEVANCE',
                v: (
                  <>
                    Wie stark und in welche Richtung ein Abschnitt die Entscheidung beeinflusst hat. Verwendet bivariate LRP.
                  </>
                ),
              },
              {
                k: 'CONFIDENCE',
                v: (
                  <>
                    Wie fake das Modell einen Abschnitt einstuft.
                  </>
                ),
              },
            ]}
          />
        </>
      ),
    },
    {
      kind: 'method',
      body: (
        <>
          <P>
            Beide Größen werden bei der Analyse <Term>einmal</Term> berechnet und
            liegen pro Chunk vor; der Toggle schaltet nur clientseitig um. Die
            Confidence-Sicht mappt die Fake-Prob <Chip>p → 2p − 1</Chip> auf dieselbe
            seismic-Skala.
          </P>
          <BivariateLrpNote compact unit="pro Audio-Chunk (nur Relevance-Sicht)" />
        </>
      ),
    },
    {
      kind: 'interpret',
      body: (
        <UL>
          <li>
            <Term>Relevance</Term> nutzen, um zu verstehen, <em>warum</em> das Modell
            reagiert.
          </li>
          <li>
            <Term>Confidence</Term> nutzen, um zu sehen, <em>wie sicher</em> ein
            Abschnitt eingestuft wird.
          </li>
        </UL>
      ),
    },
    {
      kind: 'pitfalls',
      body: (
        <Callout variant="warn" title="Nicht austauschbar">
          Relevance und Confidence sind komplementär, nicht dasselbe. Eine Relevanz
          nahe null heißt <em>„wenig Engagement"</em>, nicht „real" — dafür ist die
          Confidence-Sicht da.
        </Callout>
      ),
    },
    {
      kind: 'links',
      body: (
        <UL>
          <li>Wirkt auf <Term color="#a855f7">L1 Waveform</Term>, <Term color="#a855f7">L2 Word-Tokens</Term> und <Term color="#a855f7">L3 Frequency</Term> gemeinsam.</li>
        </UL>
      ),
    },
  ],
}
