/**
 * WordTokenChart — Layer 2 of the audio xAI stack.
 *
 * Recharts BarChart with one bar per word segment.
 * Each bar is filled with the seismic colour mapped from the word's relevance
 * score. The currently-playing word (by video time) is highlighted with an
 * animated cyan ring rendered via a custom Bar shape.
 */

import { useMemo } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Cell,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import { relevanceToRgb } from '../../lib/seismicColormap'
import { useActiveWordIndex } from '../../hooks/useActiveWordIndex'
import type { WordSegment } from '../../types/analysis'

interface WordTokenChartProps {
  wordSegments: WordSegment[]
  videoRef: React.RefObject<HTMLVideoElement | null>
}

interface BarEntry {
  word: string
  value: number
  fill: string
}

function seismicFill(relevance: number): string {
  const [r, g, b] = relevanceToRgb(relevance)
  const alpha = 0.7 + 0.3 * Math.abs(relevance)
  return `rgba(${r},${g},${b},${alpha.toFixed(2)})`
}

// Custom bar shape that draws an animated cyan ring on the active word
function ActiveBarShape(props: Record<string, unknown>) {
  const {
    x,
    y,
    width,
    height,
    fill,
    isActive,
  } = props as {
    x: number
    y: number
    width: number
    height: number
    fill: string
    isActive: boolean
  }

  // For negative values Recharts flips y/height; normalise
  const rectY = height < 0 ? y + height : y
  const rectH = Math.abs(height)

  return (
    <g>
      <rect x={x} y={rectY} width={width} height={Math.max(rectH, 1)} fill={fill} rx={2} />
      {isActive && (
        <>
          <rect
            x={x - 2}
            y={rectY - 2}
            width={width + 4}
            height={rectH + 4}
            fill="none"
            stroke="#00e5ff"
            strokeWidth={1.5}
            rx={3}
            opacity={0.9}
          />
          {/* Glow rect */}
          <rect
            x={x - 4}
            y={rectY - 4}
            width={width + 8}
            height={rectH + 8}
            fill="none"
            stroke="#00e5ff"
            strokeWidth={0.5}
            rx={4}
            opacity={0.3}
          />
        </>
      )}
    </g>
  )
}

export function WordTokenChart({ wordSegments, videoRef }: WordTokenChartProps) {
  // Re-renders only when the active word changes (not on every time tick), so
  // the chart stays stable (no per-frame jitter); the index is still checked
  // every animation frame, so fast words are not skipped.
  const activeIdx = useActiveWordIndex(videoRef, wordSegments)

  // Memoised so a re-render never hands Recharts a fresh array reference, which
  // would otherwise restart the bar animation.
  const data: BarEntry[] = useMemo(
    () =>
      wordSegments.map(w => ({
        word: w.word,
        value: w.relevance,
        fill: seismicFill(w.relevance),
      })),
    [wordSegments],
  )

  return (
    <div>
      <div
        style={{
          fontSize: 10,
          fontFamily: 'monospace',
          color: '#4d5470',
          letterSpacing: '0.12em',
          marginBottom: 6,
        }}
      >
        LAYER 2 — WORD-LEVEL RELEVANCE
      </div>

      <ResponsiveContainer width="100%" height={150}>
        <BarChart
          data={data}
          margin={{ top: 6, right: 0, bottom: 0, left: -20 }}
          barCategoryGap="18%"
        >
          <XAxis
            dataKey="word"
            interval={0}
            height={48}
            axisLine={{ stroke: '#2a2f42' }}
            tickLine={false}
            tick={(props: object) => {
              // Custom tick: every word, rotated -45° (so long words don't
              // overlap), with the currently-spoken word highlighted in cyan.
              const { x, y, payload } = props as {
                x: number
                y: number
                payload: { value: string; index: number }
              }
              const isActive = payload.index === activeIdx
              return (
                <text
                  x={x}
                  y={y}
                  dy={3}
                  textAnchor="end"
                  transform={`rotate(-45, ${x}, ${y})`}
                  fontSize={9}
                  fontFamily="monospace"
                  fontWeight={isActive ? 700 : 400}
                  fill={isActive ? '#00e5ff' : '#8b92a8'}
                >
                  {payload.value}
                </text>
              )
            }}
          />
          <YAxis
            domain={[-1, 1]}
            tick={{ fontSize: 9, fontFamily: 'monospace', fill: '#4d5470' }}
            axisLine={{ stroke: '#2a2f42' }}
            tickLine={false}
            tickCount={3}
          />
          <ReferenceLine y={0} stroke="#2a2f42" strokeWidth={1} />
          <Tooltip
            cursor={{ fill: 'rgba(0,229,255,0.04)' }}
            contentStyle={{
              backgroundColor: '#1b1f2e',
              border: '1px solid #2a2f42',
              borderRadius: 6,
              fontFamily: 'monospace',
              fontSize: 11,
              color: '#e8eaf0',
            }}
            // Force the value row white too — otherwise Recharts colours it with the
            // bar's seismic fill, which is near-black for near-zero/strong relevance.
            itemStyle={{ color: '#e8eaf0' }}
            formatter={(v) => [typeof v === 'number' ? v.toFixed(3) : '0.000', 'Relevance']}
          />
          <Bar
            dataKey="value"
            isAnimationActive={false}
            shape={(props: object) => {
              const barProps = props as { index?: number } & Record<string, unknown>
              return (
                <ActiveBarShape
                  {...barProps}
                  isActive={(barProps.index ?? -1) === activeIdx}
                />
              )
            }}
          >
            {data.map((entry, index) => (
              <Cell key={index} fill={entry.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Always reserve this line's height (even when no word is active) so the
          card never resizes when the active-word readout appears/disappears —
          otherwise Layer 3 below would jitter between words. */}
      <div
        style={{
          height: 16,
          lineHeight: '16px',
          marginTop: 2,
          fontSize: 10,
          fontFamily: 'monospace',
          color: '#00e5ff',
          letterSpacing: '0.08em',
          overflow: 'hidden',
          whiteSpace: 'nowrap',
        }}
      >
        {activeIdx >= 0 && (
          <>
            ▶ "{wordSegments[activeIdx].word}"
            {' — relevance '}
            {wordSegments[activeIdx].relevance.toFixed(3)}
          </>
        )}
      </div>
    </div>
  )
}
