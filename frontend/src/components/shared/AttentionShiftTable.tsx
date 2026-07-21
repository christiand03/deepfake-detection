/**
 * AttentionShiftTable — shared bivariate xAI component (roadmap I4).
 *
 * One horizontal bar per region/band, encoding BOTH AttnLRP channels of the
 * clean → perturbed change in a single mark:
 *   • MAGNITUDE change (relevance / attention share) → bar length + side.
 *     Centre = no change; left = LESS attention than before, right = MORE.
 *     The two dots (● centre reference = "before", ● tip = "after") sit farther
 *     apart the larger the magnitude change.
 *   • VERDICT change (signed direction, R_fake − R_real) → bar/dot COLOUR.
 *     Red = moved toward FAKE, blue = toward REAL, white = neutral (unchanged
 *     from before); colour intensity grows with the size of the verdict change.
 *
 * Hover a bar for the exact before → after → Δ values of both channels.
 * Used by AdversarialPanel (Phase 4 video + audio bands), RobustnessPanel
 * (Phase 3 video) and AudioFrequencyShift (Phase 3 audio).
 */

import { motion } from 'framer-motion'
import { useState } from 'react'

import { relevanceToRgb } from '../../lib/seismicColormap'
import type { AttentionShift } from '../../types/analysis'
import { RotationWarning } from './RotationWarning'
import { ExplanationButton } from '../../explanations/ui/ExplanationButton'
import type { VisualId } from '../../explanations/types'

const HALF_WIDTH = 46 // % of the lane each side of centre (leaves a small margin)

// FIXED, absolute full-scale for the bar length + colour — NOT per-chart
// auto-scaling. The magnitude and direction channels are percentile-normalised
// clip-global, so a per-region mean lies in [0, 1] / [-1, 1] and its
// clean → perturbed change lies in [-1, 1] / [-2, 2]. Mapping a change of
// ±MAG_FULL_SCALE to the lane extreme (and ±DIR_FULL_SCALE to full colour
// saturation) keeps bars COMPARABLE across every analysis and stops a chart of
// tiny changes from being stretched to look big. Tune these two constants to set
// how much change counts as "full scale"; do not derive them from the data.
const MAG_FULL_SCALE = 1.0
const DIR_FULL_SCALE = 1.0

const clampUnit = (v: number) => Math.max(-1, Math.min(1, v))

function fmt(v: number): string {
  return v.toFixed(2)
}

function signed(v: number): string {
  return `${v >= 0 ? '+' : ''}${v.toFixed(2)}`
}

export function AttentionShiftTable({
  shifts,
  warn = false,
  explainId,
}: {
  shifts: AttentionShift[]
  /**
   * Face is near profile → the per-region partition behind these bars is
   * unreliable (shows a caution). Only meaningful for the VIDEO-region table;
   * audio-band usages leave it off.
   */
  warn?: boolean
  /** When set, shows an explanation button in the header (F1). */
  explainId?: VisualId
}) {
  const [hover, setHover] = useState<string | null>(null)

  const rows = shifts.map(s => ({
    region: s.region,
    magnitudeBefore: s.magnitudeBefore,
    magnitudeAfter: s.magnitudeAfter,
    directionBefore: s.directionBefore,
    directionAfter: s.directionAfter,
    dMag: s.magnitudeAfter - s.magnitudeBefore,
    dDir: s.directionAfter - s.directionBefore,
  }))
  // Biggest changes on top so the eye fixates on what moved most.
  const sorted = [...rows].sort((a, b) => Math.abs(b.dMag) - Math.abs(a.dMag))

  return (
    <div>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          fontSize: 9,
          fontFamily: 'monospace',
          color: '#4d5470',
          letterSpacing: '0.12em',
          marginBottom: 8,
        }}
      >
        <span>ATTENTION SHIFT (LRP)</span>
        {explainId && <ExplanationButton id={explainId} size={15} />}
      </div>

      {warn && (
        <div style={{ marginBottom: 8 }}>
          <RotationWarning compact />
        </div>
      )}

      {/* Axis hint */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '76px 1fr',
          gap: 8,
          fontSize: 8,
          fontFamily: 'monospace',
          color: '#4d5470',
          marginBottom: 4,
        }}
      >
        <span />
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>← less</span>
          <span style={{ color: '#2a2f42' }}>no change</span>
          <span>more →</span>
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
        {sorted.map((r, i) => {
          const [cr, cg, cb] = relevanceToRgb(clampUnit(r.dDir / DIR_FULL_SCALE))
          const color = `rgb(${cr},${cg},${cb})`
          const tipPct = 50 + clampUnit(r.dMag / MAG_FULL_SCALE) * HALF_WIDTH
          const barLeft = Math.min(50, tipPct)
          const barWidth = Math.abs(tipPct - 50)
          return (
            <div
              key={r.region}
              style={{
                display: 'grid',
                gridTemplateColumns: '76px 1fr',
                gap: 8,
                alignItems: 'center',
              }}
            >
              <span
                style={{
                  fontSize: 10,
                  fontFamily: 'monospace',
                  color: '#8b92a8',
                  textAlign: 'right',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                }}
              >
                {r.region}
              </span>

              {/* Lane */}
              <div
                onMouseEnter={() => setHover(r.region)}
                onMouseLeave={() => setHover(cur => (cur === r.region ? null : cur))}
                style={{
                  position: 'relative',
                  height: 20,
                  backgroundColor: '#141720',
                  borderRadius: 4,
                  cursor: 'default',
                }}
              >
                {/* Centre "no change" guide */}
                <div
                  style={{
                    position: 'absolute',
                    left: '50%',
                    top: 3,
                    bottom: 3,
                    width: 1,
                    backgroundColor: '#2a2f42',
                    transform: 'translateX(-0.5px)',
                  }}
                />
                {/* Magnitude bar (centre → tip) */}
                <motion.div
                  initial={{ width: 0, left: '50%' }}
                  animate={{ width: `${barWidth}%`, left: `${barLeft}%` }}
                  transition={{ duration: 0.55, delay: i * 0.04, ease: 'easeOut' }}
                  style={{
                    position: 'absolute',
                    top: '50%',
                    height: 6,
                    transform: 'translateY(-50%)',
                    backgroundColor: color,
                    opacity: 0.5,
                    borderRadius: 3,
                  }}
                />
                {/* Before dot (centre reference) */}
                <div
                  style={{
                    position: 'absolute',
                    left: '50%',
                    top: '50%',
                    width: 7,
                    height: 7,
                    borderRadius: '50%',
                    backgroundColor: '#0d0f14',
                    border: '1.5px solid #8b92a8',
                    transform: 'translate(-50%, -50%)',
                  }}
                />
                {/* After dot (tip), coloured by verdict change */}
                <motion.div
                  initial={{ left: '50%' }}
                  animate={{ left: `${tipPct}%` }}
                  transition={{ duration: 0.55, delay: i * 0.04, ease: 'easeOut' }}
                  style={{
                    position: 'absolute',
                    top: '50%',
                    width: 11,
                    height: 11,
                    borderRadius: '50%',
                    backgroundColor: color,
                    boxShadow: `0 0 6px ${color}`,
                    transform: 'translate(-50%, -50%)',
                  }}
                />

                {/* Hover popup */}
                {hover === r.region && (
                  <div
                    style={{
                      position: 'absolute',
                      bottom: '100%',
                      left: '50%',
                      transform: 'translateX(-50%)',
                      marginBottom: 6,
                      padding: '5px 8px',
                      whiteSpace: 'nowrap',
                      fontSize: 10,
                      lineHeight: 1.5,
                      fontFamily: 'monospace',
                      color: '#e8eaf0',
                      backgroundColor: '#1b1f2e',
                      border: '1px solid #2a2f42',
                      borderRadius: 5,
                      pointerEvents: 'none',
                      zIndex: 3,
                    }}
                  >
                    <div style={{ color: '#a0a8c0', marginBottom: 2 }}>{r.region}</div>
                    <div>
                      <span style={{ color: '#8b92a8' }}>relevance </span>
                      {fmt(r.magnitudeBefore)} → {fmt(r.magnitudeAfter)}{' '}
                      <span style={{ color: '#8b92a8' }}>(Δ {signed(r.dMag)})</span>
                    </div>
                    <div>
                      <span style={{ color: '#8b92a8' }}>verdict </span>
                      {signed(r.directionBefore)} → {signed(r.directionAfter)}{' '}
                      <span style={{ color }}>(Δ {signed(r.dDir)})</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )
        })}

        {/* Legend */}
        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: '4px 12px',
            marginTop: 6,
            fontSize: 8,
            fontFamily: 'monospace',
            color: '#4d5470',
          }}
        >
          <span>● before (centre = no change)</span>
          <span>● after — distance = magnitude change</span>
          <span style={{ color: '#ef4444' }}>▮ → fake</span>
          <span style={{ color: '#8b92a8' }}>▮ neutral</span>
          <span style={{ color: '#3b82f6' }}>▮ → real</span>
        </div>
      </div>
    </div>
  )
}
