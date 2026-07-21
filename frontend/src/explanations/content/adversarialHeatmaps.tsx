/**
 * Explanation content — Adversarial Heatmaps (Phase 4, Clean → Attacked).
 * Quellen: AdversarialPanel.tsx; CropComparisonPlayer.tsx.
 */

import type { Explanation } from '../types'
import { BivariateLrpNote, Callout, Chip, P, Term, UL } from '../ui/widgets'

export const adversarialHeatmaps: Explanation = {
  id: 'adversarial-heatmaps',
  title: 'Adversarial Heatmaps — Clean → Attacked',
  subtitle:
    'Zwei synchrone Video-Player mit Heatmap: links der saubere, rechts der angegriffene Clip.',
  method: 'FGSM / PGD White-Box · Crop-Space-Heatmap · gekoppelte Player',
  cvr: 'relevance',
  sections: [
    {
      kind: 'what',
      body: (
        <P>
          Derselbe Vorher/Nachher-Player wie in Phase 3, hier für einen{' '}
          <Term>White-Box-Angriff</Term> (FGSM/PGD): links CLEAN, rechts ATTACKED,
          jeweils mit bivariater AttnLRP-Heatmap und gemeinsamem
          Video-Opacity-Slider.
        </P>
      ),
    },
    {
      kind: 'purpose',
      body: (
        <P>
          Beantwortet: <Term>„Wie verändert ein gezielter Angriff die Relevanz des
          Modells?"</Term>.
        </P>
      ),
    },
    {
      kind: 'method',
      body: (
        <>
          <P>
            Die Heatmaps für den sauberen und den angegriffenen Clip werden{' '}
            <Term>genau wie im Phase-1-Player berechnet und normalisiert</Term>. Für
            die Deutung von Deckkraft, Farbe und Skala gilt daher dessen Erklärung.
            Einziger Unterschied: Hier wird der Gesichts-Crop (224) gezeigt, nicht das
            Vollbild.
          </P>
          <BivariateLrpNote compact unit="dieselbe Heatmap wie im Video-Panel, clean und attacked" />
          <P>Beide Player laufen frame-aligned im Lockstep.</P>
        </>
      ),
    },
    {
      kind: 'interpret',
      body: (
        <UL>
          <li>Vergleichen, wo die Relevanz sauber vs. angegriffen sitzt.</li>
          <li>Ein erfolgreicher Angriff zerstreut oder verschiebt die Patches oft sichtbar.</li>
        </UL>
      ),
    },
    {
      kind: 'pitfalls',
      body: (
        <Callout variant="warn" title="White-Box-Worst-Case">
          Der Angriff kennt das Modell vollständig — das ist der ungünstigste Fall,
          keine Aussage über natürliche Robustheit. Gezeigt wird der Crop-Space, nicht
          das Vollbild.
        </Callout>
      ),
    },
    {
      kind: 'interaction',
      body: <P><Chip>Video-Opacity-Slider</Chip> und synchrone Wiedergabe wie im Robustness-Player.</P>,
    },
    {
      kind: 'links',
      body: (
        <UL>
          <li>
            <Term color="#a855f7">Adversarial Confidence</Term>: der numerische Effekt
            (Flip) desselben Angriffs.
          </li>
          <li>
            <Term color="#a855f7">Attention-Shift</Term>: die Verschiebung in Zahlen.
          </li>
          <li>
            <Term color="#a855f7">Robustness Crop-Comparison</Term>: das Pendant für
            natürliche Degradation.
          </li>
        </UL>
      ),
    },
  ],
}
