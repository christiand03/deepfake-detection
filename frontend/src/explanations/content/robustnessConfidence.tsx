/**
 * Explanation content — Robustness Confidence-Delta + Breaking-Point (Phase 3).
 * Quelle: RobustnessPanel.tsx.
 */

import type { Explanation } from '../types'
import { Callout, Chip, KeyValueList, P, Term, UL } from '../ui/widgets'

export const robustnessConfidence: Explanation = {
  id: 'robustness-confidence',
  title: 'Robustness — Confidence-Delta',
  subtitle:
    'Vergleicht die Confidence vor und nach einer Social-Media-Degradation und misst, wie stark die Erkennung nachlässt.',
  method: 'ffmpeg-Degradation · Re-Scoring · Severity-Schwellen',
  cvr: 'confidence',
  sections: [
    {
      kind: 'what',
      body: (
        <>
          <P>
            Zwei Confidence-Boxen (<Chip>CLEAN</Chip> → <Chip>DEGRADED</Chip>) plus
            eine Robustness-Analyse mit Confidence-Abfall, Severity und Kurzfazit.
          </P>
        </>
      ),
    },
    {
      kind: 'purpose',
      body: (
        <P>
          Beantwortet <Term>„Übersteht der Detektor reale Qualitätsverluste?"</Term> —
          Kompression (CRF), Framerate-Drops, Rauschen, Downscale-Upscale und optional
          Audio-Bitrate.
        </P>
      ),
    },
    {
      kind: 'method',
      body: (
        <P>
          Der Clip wird per ffmpeg mit den gewählten Parametern degradiert und mit{' '}
          <Term>demselben Modell</Term> neu bewertet. Der Abfall = Clean − Degraded;
          Severity-Schwellen: <Chip>&gt;50 % kritisch</Chip>,{' '}
          <Chip>&gt;25 % moderat</Chip>, sonst niedrig. Kippt das Urteil, erscheint
          ein <Term color="#f59e0b">FLIPPED</Term>-Badge.
        </P>
      ),
    },
    {
      kind: 'legend',
      body: (
        <KeyValueList
          items={[
            { k: 'Rot / Blau', v: 'Verdict FAKE / REAL je Box.' },
            { k: 'Amber-Rand', v: 'Urteil gekippt (FLIPPED) oder Gesicht verloren.' },
            { k: 'Severity', v: 'Grün niedrig · Amber moderat · Rot kritisch.' },
          ]}
        />
      ),
    },
    {
      kind: 'interpret',
      body: (
        <UL>
          <li>Großer Abfall → der Detektor ist unter dieser Degradation fragil.</li>
          <li>Gain möglich → das Modell ist sich unter Degradation manchmal sicherer.</li>
          <li>FLIPPED → die Degradation hat das Urteil umgeworfen (Breaking Point).</li>
        </UL>
      ),
    },
    {
      kind: 'pitfalls',
      body: (
        <>
          <Callout variant="warn" title="Abfall ≠ Fehlklassifikation">
            Ein Confidence-Abfall bedeutet nicht automatisch ein falsches Urteil —
            erst ein FLIP kippt die Klasse.
          </Callout>
          <Callout variant="warn" title="Face-lost = Detektionsstufe, nicht Classifier">
            Ist das Gesicht bei starker Degradation nicht mehr auffindbar, wird auf dem
            Clean-Crop bewertet und ein Hinweis gezeigt: die Face-Detection ist
            gebrochen, nicht der Classifier.
          </Callout>
          <Callout variant="warn" title="Confidence-Gain ist nicht automatisch gut">
            <P>
              Confidence misst die <Term>Sicherheit</Term> des Modells, nicht die{' '}
              <Term>Korrektheit</Term>. Ein Anstieg unter Degradation ist selten „mehr
              Robustheit" — die häufigen Ursachen:
            </P>
            <UL>
              <li>
                Die Degradation entfernt gerade die feinen Details, bei denen das
                Modell (berechtigt) zögerte. Mit weniger Information legt es sich{' '}
                <em>entschiedener</em> fest — nicht unbedingt <em>richtiger</em>.
              </li>
              <li>
                Kompressions- oder Rausch-Artefakte können Synthese-Artefakten ähneln
                und die Fake-Confidence künstlich anheben.
              </li>
            </UL>
            <P>
              Ein Gain belegt daher eher <Term>Sensibilität</Term> gegenüber der
              Degradation. Echte Robustheit zeigt sich nur in einer <Term>stabilen</Term>{' '}
              Confidence von Clean → Degraded, nicht in einer höheren Zahl.
            </P>
          </Callout>
        </>
      ),
    },
    {
      kind: 'links',
      body: (
        <UL>
          <li>
            <Term color="#a855f7">Crop-Comparison</Term>: die Heatmap unter derselben
            Degradation.
          </li>
          <li>
            <Term color="#a855f7">Attention-Shift</Term> und{' '}
            <Term color="#a855f7">Audio-Frequency-Shift</Term>: wohin sich die Evidenz
            verschiebt.
          </li>
        </UL>
      ),
    },
  ],
}
