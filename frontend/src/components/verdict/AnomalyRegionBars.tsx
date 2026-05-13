/**
 * AnomalyRegionBars — staggered spring-animated horizontal bar chart.
 *
 * Each entry shows a spatial face region (Mouth, Left Eye, Jaw, Forehead)
 * with its LRP contribution score. Bars animate in with a spring on mount
 * using Framer Motion, staggered by 80 ms per row.
 */

import { motion } from 'framer-motion'

interface RegionEntry {
  region: string
  score: number
}

interface AnomalyRegionBarsProps {
  regions: RegionEntry[]
  verdict: 'FAKE' | 'REAL'
}

export function AnomalyRegionBars({ regions, verdict }: AnomalyRegionBarsProps) {
  const barColor = verdict === 'FAKE' ? '#ef4444' : '#3b82f6'
  const glowColor = verdict === 'FAKE' ? 'rgba(239,68,68,0.4)' : 'rgba(59,130,246,0.4)'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {regions.map((r, i) => (
        <motion.div
          key={r.region}
          initial={{ opacity: 0, x: -8 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.35, delay: i * 0.08 }}
          style={{ display: 'flex', alignItems: 'center', gap: 10 }}
        >
          {/* Region label */}
          <span
            style={{
              width: 72,
              textAlign: 'right',
              fontSize: 11,
              fontFamily: 'monospace',
              color: '#8b92a8',
              flexShrink: 0,
              letterSpacing: '0.04em',
            }}
          >
            {r.region}
          </span>

          {/* Bar track */}
          <div
            style={{
              flex: 1,
              height: 6,
              borderRadius: 3,
              backgroundColor: '#232738',
              overflow: 'hidden',
            }}
          >
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${r.score * 100}%` }}
              transition={{
                duration: 0.9,
                delay: i * 0.09,
                type: 'spring',
                stiffness: 100,
                damping: 18,
              }}
              style={{
                height: '100%',
                borderRadius: 3,
                backgroundColor: barColor,
                boxShadow: `0 0 6px ${glowColor}`,
              }}
            />
          </div>

          {/* Score value */}
          <span
            style={{
              width: 36,
              fontSize: 11,
              fontFamily: 'monospace',
              color: barColor,
              textAlign: 'right',
              flexShrink: 0,
            }}
          >
            {r.score.toFixed(2)}
          </span>
        </motion.div>
      ))}
    </div>
  )
}
