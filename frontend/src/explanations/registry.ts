/**
 * Registry: maps a `VisualId` to its authored `Explanation`.
 *
 * Add a content module under `content/` and register it here. A `VisualId` that
 * is absent resolves to `undefined`; `ExplanationButton` then renders nothing, so
 * buttons and content can be wired independently while F1 is filled in.
 */

import type { Explanation, VisualId } from './types'
import { heatmapOverlay } from './content/heatmapOverlay'
import { chunkTimelines } from './content/chunkTimelines'
import { regionFace } from './content/regionFace'
import { rotationWarning } from './content/rotationWarning'
import { verdictGauges } from './content/verdictGauges'
import { audioToggle } from './content/audioToggle'
import { audioL1Waveform } from './content/audioL1Waveform'
import { audioL2Words } from './content/audioL2Words'
import { audioL3Frequency } from './content/audioL3Frequency'
import { robustnessConfidence } from './content/robustnessConfidence'
import { robustnessCropCompare } from './content/robustnessCropCompare'
import { attentionShift } from './content/attentionShift'
import { audioFrequencyShift } from './content/audioFrequencyShift'
import { adversarialHeatmaps } from './content/adversarialHeatmaps'
import { adversarialConfidence } from './content/adversarialConfidence'

const ALL: Explanation[] = [
  heatmapOverlay,
  chunkTimelines,
  regionFace,
  rotationWarning,
  verdictGauges,
  audioToggle,
  audioL1Waveform,
  audioL2Words,
  audioL3Frequency,
  robustnessConfidence,
  robustnessCropCompare,
  attentionShift,
  audioFrequencyShift,
  adversarialHeatmaps,
  adversarialConfidence,
]

const REGISTRY: Partial<Record<VisualId, Explanation>> = Object.fromEntries(
  ALL.map(e => [e.id, e]),
)

export function getExplanation(id: VisualId): Explanation | undefined {
  return REGISTRY[id]
}
