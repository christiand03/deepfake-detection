/**
 * Explanation content — Verdict-Gauges.
 * Quellen: docs/frontend_roadmap.md C1; VerdictGauge.tsx / VerdictPanel.tsx.
 */

import type { Explanation } from '../types'
import { Callout, Chip, KeyValueList, P, Term, UL } from '../ui/widgets'

export const verdictGauges: Explanation = {
  id: 'verdict-gauges',
  title: 'Verdict-Gauges',
  subtitle:
    'Die finale Confidence mit dem REAL/FAKE-Urteil des Modells.',
  method: 'Max-Pool über alle Chunks („most suspicious chunk")',
  cvr: 'confidence',
  sections: [
    {
      kind: 'what',
      body: (
        <>
          <P>
            Das zusammengefasste Gesamt-Verdict. Je nach Modus:
          </P>
          <KeyValueList
            items={[
              { k: 'Unimodal', v: 'zwei Gauges nebeneinander — VISUAL und AUDIO.' },
              { k: 'Multimodal', v: 'ein fusionierter Gauge aus Video + Audio.' },
            ]}
          />
        </>
      ),
    },
    {
      kind: 'purpose',
      body: (
        <P>
          Beantwortet <Term>„WAS ist das Urteil für den Clip und WIE sicher ist das Modell?"</Term>{' '}
          — die kompakte Confidence-Sicht, komplementär zum WO/WARUM von Heatmap und
          Region-Relevance.
        </P>
      ),
    },
    {
      kind: 'method',
      body: (
        <P>
          Der Verdict ist ein <Term>Max-Pool</Term> über alle Chunks des
          Clips (der „most suspicious chunk"). Angezeigt wird die Confidence dieses
          Urteils. Das ermöglicht einen Clip als <Chip color="#ff7070">Fake</Chip>
          zu werten, wenn bereits ein einzelner Chunk manipuliert wurde.
        </P>
      ),
    },
    {
      kind: 'legend',
      body: (
        <KeyValueList
          items={[
            { k: 'Rot', v: 'Verdict FAKE' },
            { k: 'Blau', v: 'Verdict REAL' },
            { k: 'Bogenlänge', v: 'Confidence in Prozent (0–100 %)' },
          ]}
        />
      ),
    },
    {
      kind: 'interpret',
      body: (
        <UL>
          <li>Nahe 100 % und rot → das Modell ist sich sehr sicher: FAKE.</li>
          <li>Nahe 50 % → unsicher, das Urteil ist schwach fundiert.</li>
          <li>
            Bei unimodal: VISUAL und AUDIO können auseinandergehen — je nachdem welche Modalität
            manipuliert wurde.
          </li>
        </UL>
      ),
    },
    {
      kind: 'pitfalls',
      body: (
        <>
          <Callout variant="warn" title="Max-Pool ist absichtlich empfindlich">
            Ein einzelner stark manipulierter Chunk kippt den ganzen Clip auf FAKE —
            gewollt, weil Fälschungen lokal sein können. Für die zeitliche Verteilung
            die Confidence-Timeline ansehen.
          </Callout>
          <Callout variant="warn" title="Confidence ≠ Ort">
            Der Gauge sagt <em>wie sicher</em>, nicht <em>wo</em> oder <em>warum</em>.
            Das liefern Heatmap, Region-Relevance und die Timelines.
          </Callout>
        </>
      ),
    },
    {
      kind: 'links',
      body: (
        <UL>
          <li>
            <Term color="#a855f7">Chunk-Confidence-Timeline</Term>: die per-Chunk-
            Detailsicht hinter der einen gepoolten Zahl.
          </li>
          <li>
            <Term color="#a855f7">Heatmap / Region-Relevance</Term>: das
            komplementäre WO/WARUM.
          </li>
          <li>
            <Chip>Phase 3/4</Chip>: dieselbe Confidence im Vorher/Nachher unter
            Degradation bzw. Angriff.
          </li>
        </UL>
      ),
    },
  ],
}
