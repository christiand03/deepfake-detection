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

export function AudioFrequencyShift({ audio }: { audio: AudioRobustness }) {
  const confDelta = audio.degradedConfidence - audio.baseConfidence
  const confUp = confDelta > 0
  const confColor = confUp ? '#ef4444' : '#3b82f6'

  // Map the three frequency bands to AttentionShift rows.
  // Math.abs() is used so bar widths are always positive regardless of sign
  // (band sign encodes fake/real direction; magnitude captures detection strength).
  const shifts: AttentionShift[] = [
    {
      region: 'Low',
      before: Math.abs(audio.baseFrequencyBands.low),
      after: Math.abs(audio.degradedFrequencyBands.low),
    },
    {
      region: 'Mid',
      before: Math.abs(audio.baseFrequencyBands.mid),
      after: Math.abs(audio.degradedFrequencyBands.mid),
    },
    {
      region: 'High',
      before: Math.abs(audio.baseFrequencyBands.high),
      after: Math.abs(audio.degradedFrequencyBands.high),
    },
  ]

  return (
    <div style={{ marginTop: 12 }}>
      {/* Section header */}
      <div
        style={{
          fontSize: 9,
          fontFamily: 'monospace',
          color: '#4d5470',
          letterSpacing: '0.12em',
          marginBottom: 6,
        }}
      >
        AUDIO COMPRESSION (AAC {audio.bitrate} kbps)
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
