/**
 * Explanation content — Adversarial Verdict/Confidence (Phase 4).
 * Quelle: AdversarialPanel.tsx (VerdictCompare).
 */

import type { Explanation } from '../types'
import { Callout, Chip, KeyValueList, P, Term, UL } from '../ui/widgets'

export const adversarialConfidence: Explanation = {
  id: 'adversarial-confidence',
  title: 'Adversarial Verdict — Clean vs. Attacked',
  subtitle:
    'Vergleicht Confidence und Urteil vor und nach dem Angriff — und zeigt, ob eine winzige Störung das Urteil kippt.',
  method: 'FGSM (single-step) / PGD (multi-step) · L∞-Budget ε · gleiches Modell beidseitig',
  cvr: 'confidence',
  sections: [
    {
      kind: 'what',
      body: (
        <P>
          Zwei Confidence-Boxen (<Chip>CLEAN</Chip> vs. <Chip>ATTACKED</Chip>). Kippt
          das Urteil, erscheint ein <Term color="#f59e0b">FLIPPED</Term>-Badge.
        </P>
      ),
    },
    {
      kind: 'purpose',
      body: (
        <P>
          Beantwortet <Term>„Kann eine kaum wahrnehmbare Störung das Urteil
          umdrehen?"</Term>.
        </P>
      ),
    },
    {
      kind: 'method',
      body: (
        <>
          <P>
            Beide Seiten kommen aus <Term>demselben Modell</Term>. Die Störung:
          </P>
          <KeyValueList
            items={[
              { k: 'FGSM', v: 'ein einziger Gradienten-Vorzeichen-Schritt (schnell).' },
              { k: 'PGD', v: 'mehrere projizierte Gradienten-Schritte (stärker).' },
              { k: 'ε (Epsilon)', v: 'das L∞-Budget — wie groß die Störung maximal sein darf.' },
            ]}
          />
        </>
      ),
    },
    {
      kind: 'legend',
      body: (
        <KeyValueList
          items={[
            { k: 'Rot / Blau', v: 'Verdict FAKE / REAL je Box.' },
            { k: 'Amber-Rand + FLIPPED', v: 'Das Urteil ist durch den Angriff gekippt.' },
          ]}
        />
      ),
    },
    {
      kind: 'interpret',
      body: (
        <UL>
          <li>Ein FLIP bei kleinem ε → das Modell ist brüchig.</li>
          <li>Nur ein Confidence-Rückgang ohne FLIP → geschwächt, aber Urteil hält.</li>
        </UL>
      ),
    },
    {
      kind: 'pitfalls',
      body: (
        <Callout variant="warn" title="Kein natürlicher Robustheits-Beweis">
          Ein White-Box-Angriff ist der Worst Case mit vollem Modellwissen. Ein Flip
          hier heißt nicht, dass der Detektor im Alltag versagt — dafür ist Phase 3
          (Robustness) da.
        </Callout>
      ),
    },
    {
      kind: 'links',
      body: (
        <UL>
          <li>
            <Term color="#a855f7">Adversarial Heatmaps</Term>: wie derselbe Angriff die
            Relevanz verändert.
          </li>
          <li>
            <Term color="#a855f7">Robustness Confidence-Delta</Term>: das Pendant für
            natürliche Degradation.
          </li>
        </UL>
      ),
    },
  ],
}
