/**
 * FrameTimeline — SVG sparkline of per-frame confidence scores.
 *
 * Renders a filled area chart with a vertical playhead line that tracks
 * the current frame index. Coloring: values above the 0.5 threshold are
 * tinted red (fake evidence), values below are tinted blue (real).
 */

import { useRef } from 'react'

interface FrameTimelineProps {
  scores: number[]
  currentFrame: number
  verdict: 'FAKE' | 'REAL'
}

const HEIGHT = 44
const THRESHOLD = 0.5

export function FrameTimeline({ scores, currentFrame, verdict }: FrameTimelineProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  if (scores.length === 0) return null

  const W = 600 // viewBox width
  const max = Math.max(...scores, 1)
  const isFake = verdict === 'FAKE'
  const lineColor = isFake ? '#ef4444' : '#3b82f6'
  const fillColor = isFake ? 'rgba(239,68,68,0.15)' : 'rgba(59,130,246,0.15)'

  // Build SVG polyline points
  const pts = scores.map((s, i) => {
    const x = (i / (scores.length - 1)) * W
    const y = HEIGHT - (s / max) * (HEIGHT - 4) - 2
    return `${x.toFixed(1)},${y.toFixed(1)}`
  })

  const polyPoints = pts.join(' ')
  const firstPt = pts[0].split(',')
  const lastPt = pts[pts.length - 1].split(',')

  // Closed polygon for fill (adds bottom-left and bottom-right corners)
  const fillPoints = [
    `${firstPt[0]},${HEIGHT}`,
    ...pts,
    `${lastPt[0]},${HEIGHT}`,
  ].join(' ')

  // Playhead X position
  const playheadX = ((currentFrame / Math.max(scores.length - 1, 1)) * W).toFixed(1)
  const playheadScore = scores[currentFrame] ?? 0
  const playheadY = HEIGHT - (playheadScore / max) * (HEIGHT - 4) - 2

  return (
    <div
      ref={containerRef}
      className="w-full rounded-lg px-3 py-2"
      style={{ backgroundColor: '#141720', border: '1px solid #2a2f42' }}
    >
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs font-mono" style={{ color: '#4d5470', letterSpacing: '0.1em' }}>
          FRAME CONFIDENCE
        </span>
        <span
          className="text-xs font-mono"
          style={{ color: lineColor }}
        >
          frame {currentFrame + 1}/{scores.length}
          {' · '}
          {(playheadScore * 100).toFixed(1)}%
        </span>
      </div>

      <svg
        viewBox={`0 0 ${W} ${HEIGHT}`}
        preserveAspectRatio="none"
        style={{ width: '100%', height: HEIGHT, display: 'block', overflow: 'visible' }}
      >
        {/* Threshold line */}
        <line
          x1="0" y1={(HEIGHT - THRESHOLD * (HEIGHT - 4) - 2).toFixed(1)}
          x2={W} y2={(HEIGHT - THRESHOLD * (HEIGHT - 4) - 2).toFixed(1)}
          stroke="#2a2f42"
          strokeWidth="1"
          strokeDasharray="4 3"
        />

        {/* Fill */}
        <polygon points={fillPoints} fill={fillColor} />

        {/* Line */}
        <polyline
          points={polyPoints}
          fill="none"
          stroke={lineColor}
          strokeWidth="1.5"
          strokeLinejoin="round"
          strokeLinecap="round"
        />

        {/* Playhead */}
        <line
          x1={playheadX} y1="0"
          x2={playheadX} y2={HEIGHT}
          stroke="#00e5ff"
          strokeWidth="1.5"
          opacity="0.8"
        />
        <circle
          cx={playheadX}
          cy={playheadY.toFixed(1)}
          r="3"
          fill="#00e5ff"
        />
      </svg>
    </div>
  )
}
