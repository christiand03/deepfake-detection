/**
 * Explanation content — Crop-Comparison-Player (Phase 3, Clean → Degraded).
 * Quelle: CropComparisonPlayer.tsx.
 */

import type { Explanation } from '../types'
import { BivariateLrpNote, Callout, Chip, P, Term, UL } from '../ui/widgets'

export const robustnessCropCompare: Explanation = {
  id: 'robustness-crop-compare',
  title: 'Crop-Comparison — Clean → Degraded',
  subtitle:
    'Zwei synchron laufende Video-Player mit Heatmap-Overlay: links das Original, rechts der degradierte Clip.',
  method: 'Crop-Space-Heatmap · gekoppelte Player · Video-Opacity-Slider',
  cvr: 'relevance',
  sections: [
    {
      kind: 'what',
      body: (
        <P>
          Zwei Gesichts-Crop-Videos nebeneinander, jeweils mit der bivariaten
          AttnLRP-Heatmap darüber. Ein Slider steuert die Video-Deckkraft beider
          Player gemeinsam (bis 0 % = nur die Heatmaps sichtbar).
        </P>
      ),
    },
    {
      kind: 'purpose',
      body: (
        <P>
          Beantwortet <Term>„WOHIN wandert die Relevanz, wenn der Clip degradiert
          wird?"</Term> — der direkte visuelle Vorher/Nachher-Vergleich.
        </P>
      ),
    },
    {
      kind: 'method',
      body: (
        <>
          <P>
            Die Heatmap wird <Term>genau wie im Phase-1-Player berechnet und
            normalisiert</Term>. Für die Deutung von Deckkraft, Farbe und Skala gilt
            daher dessen Erklärung. Einziger Unterschied: Hier wird der Gesichts-Crop
            (224) gezeigt, nicht das Vollbild.
          </P>
          <BivariateLrpNote compact unit="dieselbe Heatmap wie im Video-Panel, clean und degradiert" />
          <P>
            Beide Player sind in <Term>Lockstep</Term> (Play/Pause/Seek/Rate
            gespiegelt, mit Drift-Korrektur), sodass der Vergleich immer frame-aligned
            ist.
          </P>
        </>
      ),
    },
    {
      kind: 'interpret',
      body: (
        <UL>
          <li>Die Heatmap-Patches links und rechts direkt vergleichen.</li>
          <li>Das Video ausblenden, um nur die Relevanz-Patches zu sehen.</li>
          <li>Verschiebt/zerstreut sich die Heatmap rechts, hat die Degradation die Evidenz verändert.</li>
        </UL>
      ),
    },
    {
      kind: 'pitfalls',
      body: (
        <Callout variant="warn" title="Crop-Space, nicht Vollbild">
          Gezeigt wird der Gesichts-Crop, nicht der ganze Frame. Die Heatmap
          ist auf ~4 Hz gedrosselt, genauso wie auch in Phase 1, damit diese nicht flackert und gut lesbar ist.
        </Callout>
      ),
    },
    {
      kind: 'interaction',
      body: (
        <UL>
          <li><Chip>Video-Opacity-Slider</Chip> blendet das Video stufenlos aus.</li>
          <li>Beide Player sind synchronisiert; ein Region-Overlay ist für den Attention Shift per Button austauschbar.</li>
        </UL>
      ),
    },
    {
      kind: 'links',
      body: (
        <UL>
          <li>
            <Term color="#a855f7">Attention-Shift</Term>: quantifiziert dieselbe
            Regionen-Verschiebung auf Clip-Ebene.
          </li>
          <li>
            <Term color="#a855f7">Confidence-Delta</Term>: der numerische Effekt auf
            das Urteil.
          </li>
        </UL>
      ),
    },
  ],
}
