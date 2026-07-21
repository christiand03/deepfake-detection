/**
 * AudioFrequencyShift — xAI comparison of Wav2Vec2 frequency-band relevance
 * before and after AAC audio compression.
 *
 * Used by RobustnessPanel (Phase 3) to visualise how social-media audio
 * compression degrades the model's per-band detection signal (Low / Mid / High).
 * Band magnitudes are mapped to AttentionShift rows and rendered via the shared
 * AttentionShiftTable component.
 */

import type { AttentionShift, AudioRobustness } from '../../types/analysis'

import { AttentionShiftTable } from './AttentionShiftTable'
import { ExplanationButton } from '../../explanations/ui/ExplanationButton'

export function AudioFrequencyShift({ audio }: { audio: AudioRobustness }) {
  const confDelta = audio.degradedConfidence - audio.baseConfidence
  const confUp = confDelta > 0
  const confColor = confUp ? '#ef4444' : '#3b82f6'

  // Map the three frequency bands to bivariate AttentionShift rows (I4). The band
  // values are SIGNED (they encode fake/real direction), so magnitude = |value|
  // (detection strength / attention share) and direction = value (verdict lean).
  const band = (base: number, degraded: number, region: string): AttentionShift => ({
    region,
    magnitudeBefore: Math.abs(base),
    magnitudeAfter: Math.abs(degraded),
    directionBefore: base,
    directionAfter: degraded,
  })
  const shifts: AttentionShift[] = [
    band(audio.baseFrequencyBands.low, audio.degradedFrequencyBands.low, 'Low'),
    band(audio.baseFrequencyBands.mid, audio.degradedFrequencyBands.mid, 'Mid'),
    band(audio.baseFrequencyBands.high, audio.degradedFrequencyBands.high, 'High'),
  ]

  return (
    <div style={{ marginTop: 12 }}>
      {/* Section header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          fontSize: 9,
          fontFamily: 'monospace',
          color: '#4d5470',
          letterSpacing: '0.12em',
          marginBottom: 6,
        }}
      >
        <span>AUDIO COMPRESSION (AAC {audio.bitrate} kbps)</span>
        <ExplanationButton id="audio-frequency-shift" label="Audio-Frequency-Shift erklären" size={15} />
      </div>

      {/* Confidence delta */}
      <div
        style={{
          fontSize: 10,
          fontFamily: 'monospace',
          color: '#8b92a8',
          marginBottom: 10,
        }}
      >
        audio confidence:{' '}
        <span style={{ color: '#a0a8c0' }}>
          {(audio.baseConfidence * 100).toFixed(1)}%
        </span>
        {' → '}
        <span style={{ color: confColor }}>
          {(audio.degradedConfidence * 100).toFixed(1)}%
        </span>
        <span style={{ color: confColor, fontWeight: 700 }}>
          {' '}
          ({confUp ? '+' : ''}
          {(confDelta * 100).toFixed(1)}%)
        </span>
      </div>

      <AttentionShiftTable shifts={shifts} />
    </div>
  )
}
