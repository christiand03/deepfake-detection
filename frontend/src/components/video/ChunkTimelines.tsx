/**
 * ChunkTimelines — two stacked per-chunk timelines under the video player (A1).
 *
 *   1. CHUNK CONFIDENCE  — the model's REAL↔FAKE classification of each 16-frame
 *      chunk (per-window fake probability, midline at 0.5). Each segment is
 *      coloured by ITS OWN class (> 0.5 red / fake, < 0.5 blue / real), NOT by the
 *      single clip verdict — so a short manipulation shows as FAKE only where it
 *      occurs and the rest stays REAL.
 *   2. CHUNK RELEVANCE   — a hybrid: bar HEIGHT = influence magnitude
 *      (mean |relevance| per chunk), bar COLOUR = direction (sign of the net
 *      relevance: red = fake-supporting, blue = real-supporting).
 *
 * Both charts span the full clip width, so a shared cyan playhead (positioned by
 * the current video time) lines them up. The two source arrays may differ in
 * length (confidence is per forward-pass chunk, relevance is per heatmap window);
 * each is mapped independently across the same width.
 */

import { useState } from 'react'
import { relevanceToRgb } from '../../lib/seismicColormap'
import { ExplanationButton } from '../../explanations/ui/ExplanationButton'

interface ChunkTimelinesProps {
  confidence: number[]
  relevanceMagnitude: number[]
  relevanceSign: number[]
  /** Current frame index from useVideoSync (0-based). */
  currentFrame: number
  /** Total video frames (perFrameScores length) for the playhead time fraction. */
  totalFrames: number
}

const FRAMES_PER_CHUNK = 16
const CHART_H = 40
const W = 600 // viewBox width
// Linear display gain for the relevance bars: mean(|relevance|) per chunk is small
// in absolute terms (averaged over all pixels) even after clip-global normalisation,
// so scale it up uniformly for visibility. Applied equally to every bar, so the
// relative differences between chunks stay faithful.
const RELEVANCE_DISPLAY_GAIN = 4

function rgb(value: number): string {
  const [r, g, b] = relevanceToRgb(value)
  return `rgb(${r},${g},${b})`
}

/** Confidence timeline: per-segment-coloured line around a 0.5 midline. */
function ConfidenceChart({ scores }: { scores: number[] }) {
  const mid = CHART_H / 2
  // y(0)=bottom, y(1)=top; 0.5 → midline.
  const y = (s: number) => CHART_H - s * (CHART_H - 4) - 2
  // Slot-centred (matches the relevance bars and the per-window time centre), so
  // point i sits directly above bar i of the relevance timeline below.
  const slot = scores.length > 0 ? W / scores.length : W
  const x = (i: number) => (i + 0.5) * slot

  return (
    <svg
      viewBox={`0 0 ${W} ${CHART_H}`}
      preserveAspectRatio="none"
      style={{ width: '100%', height: CHART_H, display: 'block', overflow: 'visible' }}
    >
      {/* Midline (decision threshold) */}
      <line x1="0" y1={mid} x2={W} y2={mid} stroke="#2a2f42" strokeWidth="1" strokeDasharray="4 3" />

      {/* Per-segment coloured line: each segment tinted by its mean class. */}
      {scores.map((s, i) => {
        if (i === 0) return null
        const prev = scores[i - 1]
        const seg = (prev + s) / 2 // > 0.5 fake (red), < 0.5 real (blue)
        // Map probability 0..1 → signed -1..1 for the seismic ramp.
        const color = rgb(seg * 2 - 1)
        return (
          <line
            key={i}
            x1={x(i - 1).toFixed(1)}
            y1={y(prev).toFixed(1)}
            x2={x(i).toFixed(1)}
            y2={y(s).toFixed(1)}
            stroke={color}
            strokeWidth="2"
            strokeLinecap="round"
          />
        )
      })}

      {/* Per-chunk markers */}
      {scores.map((s, i) => (
        <circle key={i} cx={x(i).toFixed(1)} cy={y(s).toFixed(1)} r="2" fill={rgb(s * 2 - 1)} />
      ))}
    </svg>
  )
}

/** Relevance hybrid: bar height = magnitude, colour = sign (direction). */
function RelevanceChart({
  magnitude,
  sign,
}: {
  magnitude: number[]
  sign: number[]
}) {
  const n = magnitude.length
  const slot = n > 0 ? W / n : W
  const barW = Math.max(2, slot * 0.6)

  return (
    <svg
      viewBox={`0 0 ${W} ${CHART_H}`}
      preserveAspectRatio="none"
      style={{ width: '100%', height: CHART_H, display: 'block', overflow: 'visible' }}
    >
      <line x1="0" y1={CHART_H - 1} x2={W} y2={CHART_H - 1} stroke="#2a2f42" strokeWidth="1" />
      {magnitude.map((m, i) => {
        // Magnitude is the clip-globally-normalised mean(|relevance|) per chunk:
        // absolute and comparable across chunks (a calm chunk stays low), but small
        // since it averages over all pixels. A fixed linear gain lifts it into the
        // chart's height range WITHOUT per-chunk renormalisation (ratios preserved).
        const h = Math.max(1, Math.min(1, m * RELEVANCE_DISPLAY_GAIN) * (CHART_H - 4))
        const cx = (i + 0.5) * slot
        const s = sign[i] ?? 0
        // Full-saturation direction colour; height already encodes magnitude.
        const color = rgb(s)
        return (
          <rect
            key={i}
            x={(cx - barW / 2).toFixed(1)}
            y={(CHART_H - h - 1).toFixed(1)}
            width={barW.toFixed(1)}
            height={h.toFixed(1)}
            rx="1"
            fill={color}
            opacity="0.9"
          />
        )
      })}
    </svg>
  )
}

function Playhead({ fraction }: { fraction: number }) {
  const x = (Math.max(0, Math.min(1, fraction)) * W).toFixed(1)
  return (
    <svg
      viewBox={`0 0 ${W} ${CHART_H}`}
      preserveAspectRatio="none"
      style={{
        width: '100%',
        height: CHART_H,
        display: 'block',
        position: 'absolute',
        inset: 0,
        pointerEvents: 'none',
      }}
    >
      <line x1={x} y1="0" x2={x} y2={CHART_H} stroke="#00e5ff" strokeWidth="1.5" opacity="0.85" />
    </svg>
  )
}

function ChartRow({
  label,
  legend,
  children,
  fraction,
  n,
  renderTooltip,
}: {
  label: string
  legend: React.ReactNode
  children: React.ReactNode
  fraction: number
  /** Number of chunks/windows, for mapping the cursor x → chunk index. */
  n: number
  /** Tooltip content for the hovered chunk index. */
  renderTooltip: (index: number) => React.ReactNode
}) {
  const [hover, setHover] = useState<{ i: number; xPct: number } | null>(null)

  function onMove(e: React.MouseEvent<HTMLDivElement>) {
    const rect = e.currentTarget.getBoundingClientRect()
    if (rect.width === 0 || n === 0) return
    const f = (e.clientX - rect.left) / rect.width
    const i = Math.max(0, Math.min(n - 1, Math.floor(f * n)))
    // Snap the guide/tooltip to the chunk's slot centre (matches the bars/points).
    setHover({ i, xPct: ((i + 0.5) / n) * 100 })
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs font-mono" style={{ color: '#4d5470', letterSpacing: '0.1em' }}>
          {label}
        </span>
        <span className="text-xs font-mono">{legend}</span>
      </div>
      <div
        style={{ position: 'relative' }}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      >
        {children}
        <Playhead fraction={fraction} />
        {hover && (
          <>
            {/* Hover guide */}
            <div
              style={{
                position: 'absolute',
                top: 0,
                bottom: 0,
                left: `${hover.xPct}%`,
                width: 1,
                backgroundColor: '#8b92a8',
                opacity: 0.5,
                pointerEvents: 'none',
              }}
            />
            {/* Tooltip */}
            <div
              style={{
                position: 'absolute',
                bottom: '100%',
                left: `${hover.xPct}%`,
                transform: 'translateX(-50%)',
                marginBottom: 4,
                padding: '3px 7px',
                whiteSpace: 'nowrap',
                fontSize: 10,
                fontFamily: 'monospace',
                color: '#e8eaf0',
                backgroundColor: '#1b1f2e',
                border: '1px solid #2a2f42',
                borderRadius: 5,
                pointerEvents: 'none',
                zIndex: 2,
              }}
            >
              {renderTooltip(hover.i)}
            </div>
          </>
        )}
      </div>
    </div>
  )
}

export function ChunkTimelines({
  confidence,
  relevanceMagnitude,
  relevanceSign,
  currentFrame,
  totalFrames,
}: ChunkTimelinesProps) {
  if (confidence.length === 0 && relevanceMagnitude.length === 0) return null

  const fraction = totalFrames > 1 ? currentFrame / (totalFrames - 1) : 0
  const currentChunk = Math.floor(currentFrame / FRAMES_PER_CHUNK)
  const chunkConf = confidence[Math.min(currentChunk, confidence.length - 1)] ?? 0
  const isFake = chunkConf > 0.5

  const realFakeLegend = (
    <>
      <span style={{ color: '#5e91ee', marginRight: 8 }}>■ real</span>
      <span style={{ color: '#ff7070' }}>■ fake</span>
    </>
  )

  return (
    <div
      className="w-full rounded-lg px-3 py-3 flex flex-col gap-3"
      style={{ backgroundColor: '#141720', border: '1px solid #2a2f42' }}
    >
      <div className="flex items-center justify-between">
        <span className="text-xs font-mono" style={{ color: '#4d5470', letterSpacing: '0.12em' }}>
          PER-CHUNK TIMELINES
        </span>
        <ExplanationButton id="chunk-timelines" label="Chunk-Timelines erklären" size={16} />
      </div>

      {confidence.length > 0 && (
        <ChartRow
          label="CHUNK CONFIDENCE"
          legend={
            <span style={{ color: isFake ? '#ff7070' : '#5e91ee' }}>
              chunk {Math.min(currentChunk + 1, confidence.length)}/{confidence.length} ·{' '}
              {isFake ? 'FAKE' : 'REAL'} {(Math.abs(chunkConf - 0.5) * 2 * 100).toFixed(0)}%
            </span>
          }
          fraction={fraction}
          n={confidence.length}
          renderTooltip={i => {
            const p = confidence[i] ?? 0
            const fake = p > 0.5
            return (
              <>
                chunk {i + 1}/{confidence.length} ·{' '}
                <span style={{ color: fake ? '#ff7070' : '#5e91ee' }}>{fake ? 'FAKE' : 'REAL'}</span>{' '}
                {(Math.abs(p - 0.5) * 2 * 100).toFixed(0)}% · p(fake) {p.toFixed(3)}
              </>
            )
          }}
        >
          <ConfidenceChart scores={confidence} />
        </ChartRow>
      )}

      {relevanceMagnitude.length > 0 && (
        <ChartRow
          label="CHUNK RELEVANCE"
          legend={realFakeLegend}
          fraction={fraction}
          n={relevanceMagnitude.length}
          renderTooltip={i => {
            const m = relevanceMagnitude[i] ?? 0
            const s = relevanceSign[i] ?? 0
            const fake = s > 0
            // % of the bar's display scale (height = min(1, value × gain)), so the
            // readout matches the bar even where high values saturate at 100%.
            const pct = Math.min(100, Math.round(m * RELEVANCE_DISPLAY_GAIN * 100))
            return (
              <>
                chunk {i + 1}/{relevanceMagnitude.length} · rel. strength {m.toFixed(3)} ({pct}% of
                scale) ·{' '}
                <span style={{ color: fake ? '#ff7070' : '#5e91ee' }}>
                  {fake ? 'fake' : 'real'}-supporting
                </span>
              </>
            )
          }}
        >
          <RelevanceChart magnitude={relevanceMagnitude} sign={relevanceSign} />
        </ChartRow>
      )}
    </div>
  )
}
