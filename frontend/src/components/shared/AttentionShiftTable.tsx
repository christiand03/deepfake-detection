/**
 * AttentionShiftTable — shared xAI component.
 *
 * Renders a bar-chart comparison of AttnLRP region scores before and after
 * a perturbation (adversarial attack or social-media degradation).
 * Used by both AdversarialPanel (Phase 4) and RobustnessPanel (Phase 3).
 */

import { motion } from 'framer-motion'

import type { AttentionShift } from '../../types/analysis'

export function AttentionShiftTable({ shifts }: { shifts: AttentionShift[] }) {
  const maxVal = Math.max(...shifts.flatMap(s => [s.before, s.after]), 0.01)

  return (
    <div>
      <div
        style={{
          fontSize: 9,
          fontFamily: 'monospace',
          color: '#4d5470',
          letterSpacing: '0.12em',
          marginBottom: 8,
        }}
      >
        ATTENTION SHIFT (LRP)
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {shifts.map(s => {
          const delta = s.after - s.before
          const isUp = delta > 0
          const deltaColor = isUp ? '#ef4444' : '#3b82f6'
          return (
            <div
              key={s.region}
              style={{
                display: 'grid',
                gridTemplateColumns: '72px 1fr 1fr 52px',
                gap: 6,
                alignItems: 'center',
              }}
            >
              {/* Region label */}
              <span
                style={{
                  fontSize: 10,
                  fontFamily: 'monospace',
                  color: '#8b92a8',
                  textAlign: 'right',
                }}
              >
                {s.region}
              </span>
              {/* Before bar */}
              <div
                style={{
                  height: 8,
                  backgroundColor: '#1b1f2e',
                  borderRadius: 4,
                  overflow: 'hidden',
                  border: '1px solid #2a2f42',
                }}
              >
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${(s.before / maxVal) * 100}%` }}
                  transition={{ duration: 0.7, ease: 'easeOut' }}
                  style={{
                    height: '100%',
                    backgroundColor: 'rgba(139,146,168,0.5)',
                    borderRadius: 4,
                  }}
                />
              </div>
              {/* After bar */}
              <div
                style={{
                  height: 8,
                  backgroundColor: '#1b1f2e',
                  borderRadius: 4,
                  overflow: 'hidden',
                  border: `1px solid ${deltaColor}44`,
                }}
              >
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${(s.after / maxVal) * 100}%` }}
                  transition={{ duration: 0.7, delay: 0.1, ease: 'easeOut' }}
                  style={{
                    height: '100%',
                    backgroundColor: deltaColor,
                    borderRadius: 4,
                    opacity: 0.75,
                  }}
                />
              </div>
              {/* Delta */}
              <span
                style={{
                  fontSize: 10,
                  fontFamily: 'monospace',
                  color: deltaColor,
                  fontWeight: 700,
                  textAlign: 'right',
                }}
              >
                {isUp ? '+' : ''}
                {delta.toFixed(2)}
              </span>
            </div>
          )
        })}

        {/* Legend */}
        <div
          style={{
            display: 'flex',
            gap: 12,
            marginTop: 4,
            fontSize: 8,
            fontFamily: 'monospace',
            color: '#2a2f42',
          }}
        >
          <span style={{ color: '#4d5470' }}>■ before</span>
          <span style={{ color: '#ef4444' }}>■ after (↑ fake signal)</span>
          <span style={{ color: '#3b82f6' }}>■ after (↓ real signal)</span>
        </div>
      </div>
    </div>
  )
}
