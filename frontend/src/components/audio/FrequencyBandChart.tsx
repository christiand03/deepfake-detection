/**
 * FrequencyBandChart — Layer 3 of the audio xAI stack.
 *
 * Three horizontal bars showing AttnLRP relevance aggregated per frequency
 * band. Positive values indicate fake-supporting evidence (red), negative
 * values indicate real-supporting evidence (blue).
 *
 * Bands:
 *   Low   0–500 Hz    — Prosody / fundamental frequency
 *   Mid   500–4 kHz   — Phoneme formants
 *   High  4–8 kHz     — Consonants / fricatives
 */

import { motion } from 'framer-motion'
import { seismicToRgb } from '../../lib/seismicColormap'
import type { FrequencyBands } from '../../types/analysis'

interface FrequencyBandChartProps {
  bands: FrequencyBands
}

const BAND_DEFS = [
  { key: 'low' as const, label: 'Low', range: '0–500 Hz', desc: 'Prosody' },
  { key: 'mid' as const, label: 'Mid', range: '500–4 kHz', desc: 'Phonemes' },
  { key: 'high' as const, label: 'High', range: '4–8 kHz', desc: 'Consonants' },
]

function bandColor(value: number): string {
  const [r, g, b] = seismicToRgb(value)
  const alpha = 0.8 + 0.2 * Math.abs(value)
  return `rgba(${r},${g},${b},${alpha.toFixed(2)})`
}

function bandGlow(value: number): string {
  if (value > 0) return `rgba(239,68,68,0.35)`
  if (value < 0) return `rgba(59,130,246,0.35)`
  return 'transparent'
}

export function FrequencyBandChart({ bands }: FrequencyBandChartProps) {
  return (
    <div>
      <div
        style={{
          fontSize: 10,
          fontFamily: 'monospace',
          color: '#4d5470',
          letterSpacing: '0.12em',
          marginBottom: 8,
        }}
      >
        LAYER 3 — FREQUENCY BAND RELEVANCE
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {BAND_DEFS.map((band, i) => {
          const value = bands[band.key]
          const pct = Math.abs(value) * 100
          const color = bandColor(value)
          const glow = bandGlow(value)
          const isPositive = value > 0

          return (
            <div key={band.key}>
              {/* Label row */}
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  marginBottom: 4,
                  alignItems: 'baseline',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
                  <span
                    style={{
                      fontSize: 11,
                      fontFamily: 'monospace',
                      fontWeight: 600,
                      color: color,
                    }}
                  >
                    {band.label}
                  </span>
                  <span
                    style={{
                      fontSize: 9,
                      fontFamily: 'monospace',
                      color: '#4d5470',
                    }}
                  >
                    {band.range} · {band.desc}
                  </span>
                </div>
                <span
                  style={{
                    fontSize: 11,
                    fontFamily: 'monospace',
                    color: color,
                    fontWeight: 600,
                  }}
                >
                  {isPositive ? '+' : ''}{value.toFixed(2)}
                </span>
              </div>

              {/* Bar track (bidirectional from centre) */}
              <div
                style={{
                  height: 10,
                  borderRadius: 5,
                  backgroundColor: '#1b1f2e',
                  position: 'relative',
                  overflow: 'hidden',
                }}
              >
                {/* Centre marker */}
                <div
                  style={{
                    position: 'absolute',
                    left: '50%',
                    top: 0,
                    bottom: 0,
                    width: 1,
                    backgroundColor: '#2a2f42',
                    zIndex: 1,
                  }}
                />

                {/* Animated bar */}
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${pct / 2}%` }}
                  transition={{
                    duration: 0.85,
                    delay: i * 0.1,
                    type: 'spring',
                    stiffness: 90,
                    damping: 16,
                  }}
                  style={{
                    position: 'absolute',
                    top: 0,
                    bottom: 0,
                    ...(isPositive
                      ? { left: '50%' }
                      : { right: '50%' }),
                    backgroundColor: color,
                    boxShadow: `0 0 8px ${glow}`,
                    borderRadius: isPositive ? '0 5px 5px 0' : '5px 0 0 5px',
                  }}
                />
              </div>

              {/* Direction labels */}
              {i === 0 && (
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    marginTop: 2,
                    fontSize: 8,
                    fontFamily: 'monospace',
                    color: '#2a2f42',
                  }}
                >
                  <span style={{ color: '#3b4cc0' }}>◀ REAL</span>
                  <span style={{ color: '#b40426' }}>FAKE ▶</span>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
