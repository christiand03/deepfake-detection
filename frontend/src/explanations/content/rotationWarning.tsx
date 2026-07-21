/**
 * Explanation content — Rotation-Warning (Kopf gedreht).
 * Quelle: RotationWarning.tsx.
 */

import type { Explanation } from '../types'
import { Chip, P, Term, UL } from '../ui/widgets'

export const rotationWarning: Explanation = {
  id: 'rotation-warning',
  title: 'Rotation-Warning',
  subtitle:
    'Ein Hinweis, der erscheint, wenn das Gesicht im Clip stärker gedreht ist.',
  method: 'MediaPipe FaceMesh · Yaw-Erkennung',
  cvr: 'neither',
  sections: [
    {
      kind: 'what',
      body: (
        <P>
          Ein kleiner Warnhinweis an den region-basierten Visualisierungen
          (Gesichts-Schema, Attention-Shift). Er taucht nur auf, wenn das Backend den
          Clip als stark gedreht markiert (<Chip>faceRotationWarning</Chip>).
        </P>
      ),
    },
    {
      kind: 'method',
      body: (
        <P>
          Die Regionen stammen aus einem <Term>MediaPipe-FaceMesh</Term>, das ein
          frontal ausgerichtetes Gesicht benötigt. Bei starker Rotation wird die verdeckte
          Gesichtshälfte halluziniert, sodass die Regionen-Partition nicht mehr zum
          sichtbaren Gesicht passt.
        </P>
      ),
    },
    {
      kind: 'interpret',
      body: (
        <P>
          Wenn der Hinweis sichtbar ist, dann ist die per-Region-Zuordnung mit hoher Wahrscheinlichkeit degradiert.
          Es ist empfohlen einmal die Regionszuordnung zu überprüfen.
        </P>
      ),
    },
    {
      kind: 'links',
      body: (
        <UL>
          <li>
            <Term color="#a855f7">Region-Relevance</Term> und{' '}
            <Term color="#a855f7">Attention-Shift</Term>: die betroffenen Visuals.
          </li>
        </UL>
      ),
    },
  ],
}
