/**
 * VerdictGauge — animated SVG half-circle arc gauge.
 *
 * Framer Motion animates stroke-dashoffset from full (empty arc) to the
 * confidence-proportional length on result arrival. Glows in red (FAKE) or
 * blue (REAL) matching the seismic / verdict palette.
 */

import { motion } from 'framer-motion'

interface VerdictGaugeProps {
  confidence: number
  verdict: 'FAKE' | 'REAL'
  isScanning: boolean
}

const CX = 100
const CY = 92
const R = 72
// Half-circle circumference: π * R
const CIRC = Math.PI * R // ≈ 226.19

const ARC_D = `M ${CX - R} ${CY} A ${R} ${R} 0 0 1 ${CX + R} ${CY}`

export function VerdictGauge({ confidence, verdict, isScanning }: VerdictGaugeProps) {
  const arcColor = verdict === 'FAKE' ? '#ef4444' : '#3b82f6'
  const dashOffset = CIRC - confidence * CIRC

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 10,
      }}
    >
      <svg
        width="200"
        height="110"
        viewBox="0 0 200 110"
        style={{ width: '100%', height: 'auto', overflow: 'visible' }}
      >
        {/* Tick marks */}
        {[0, 0.25, 0.5, 0.75, 1].map(pct => {
          const angle = Math.PI + pct * Math.PI // 180° → 360°
          const x1 = CX + (R + 4) * Math.cos(angle)
          const y1 = CY + (R + 4) * Math.sin(angle)
          const x2 = CX + (R + 10) * Math.cos(angle)
          const y2 = CY + (R + 10) * Math.sin(angle)
          return (
            <line
              key={pct}
              x1={x1.toFixed(1)}
              y1={y1.toFixed(1)}
              x2={x2.toFixed(1)}
              y2={y2.toFixed(1)}
              stroke="#2a2f42"
              strokeWidth="1.5"
            />
          )
        })}

        {/* Track arc */}
        <path
          d={ARC_D}
          fill="none"
          stroke="#232738"
          strokeWidth="12"
          strokeLinecap="round"
        />

        {/* Animated fill arc */}
        <motion.path
          d={ARC_D}
          fill="none"
          stroke={arcColor}
          strokeWidth="12"
          strokeLinecap="round"
          strokeDasharray={`${CIRC} ${CIRC}`}
          initial={{ strokeDashoffset: CIRC }}
          animate={{
            strokeDashoffset: isScanning ? CIRC : dashOffset,
          }}
          transition={{ duration: 1.2, ease: [0.34, 1.56, 0.64, 1] }}
          style={{
            filter: isScanning ? 'none' : `drop-shadow(0 0 8px ${arcColor}99)`,
          }}
        />

        {/* Confidence % */}
        <motion.text
          x={CX}
          y={CY - 6}
          textAnchor="middle"
          fontSize="34"
          fontWeight="700"
          fill={isScanning ? '#2a2f42' : arcColor}
          fontFamily="monospace"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.4, delay: isScanning ? 0 : 0.7 }}
        >
          {isScanning ? '—' : `${Math.round(confidence * 100)}%`}
        </motion.text>

        {/* Sub-label */}
        <text
          x={CX}
          y={CY + 14}
          textAnchor="middle"
          fontSize="9"
          fill="#4d5470"
          fontFamily="monospace"
          letterSpacing="0.18em"
        >
          CONFIDENCE
        </text>

        {/* Scale labels */}
        <text x={CX - R - 4} y={CY + 16} fontSize="9" fill="#4d5470" fontFamily="monospace" textAnchor="middle">
          0%
        </text>
        <text x={CX + R + 4} y={CY + 16} fontSize="9" fill="#4d5470" fontFamily="monospace" textAnchor="middle">
          100%
        </text>
      </svg>

      {/* Verdict badge */}
      <motion.div
        initial={{ opacity: 0, scale: 0.92 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.35, delay: isScanning ? 0 : 0.9 }}
        style={{
          paddingInline: 20,
          paddingBlock: 5,
          borderRadius: 24,
          fontSize: 12,
          fontFamily: 'monospace',
          fontWeight: 700,
          letterSpacing: '0.2em',
          backgroundColor: isScanning
            ? 'transparent'
            : verdict === 'FAKE'
              ? 'rgba(239,68,68,0.12)'
              : 'rgba(59,130,246,0.12)',
          border: `1px solid ${isScanning ? '#2a2f42' : arcColor}`,
          color: isScanning ? '#4d5470' : arcColor,
        }}
      >
        {isScanning ? 'ANALYZING' : verdict}
      </motion.div>
    </div>
  )
}
