/**
 * AudioLayers — right-panel audio xAI section; always visible.
 *
 * Shows an idle placeholder when no audio result is available.
 * Once analysis completes with audio, renders three stacked layers:
 *   L1 — WaveformRelevanceLayer  (Canvas)
 *   L2 — WordTokenChart          (Recharts BarChart, scrollable)
 *   L3 — FrequencyBandChart      (custom SVG horizontal bars)
 *
 * The shared `videoRef` is passed straight to L1/L2; each syncs to playback
 * itself (imperative rAF playhead in L1, `useActiveWordIndex` in L2) so the
 * panel does not re-render on every animation frame.
 */

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { WaveformRelevanceLayer } from './WaveformRelevanceLayer'
import { WordTokenChart } from './WordTokenChart'
import { FrequencyBandChart } from './FrequencyBandChart'
import { FrequencyHeatmap } from './FrequencyHeatmap'
import type {
  AnalysisResult,
  AudioView,
  ClipMeta,
  FrequencyBands,
} from '../../types/analysis'

interface AudioLayersProps {
  result: AnalysisResult | null
  clip: ClipMeta | null
  videoRef: React.RefObject<HTMLVideoElement | null>
}

/**
 * Lift the signed Confidence-view bands ({low,mid,high} floats) into the
 * bivariate shape FrequencyBandChart now expects: magnitude = |value| (bar
 * width), direction = value (side/colour). The bar then reads identically to
 * before in the Confidence view, while the Relevance view passes through the
 * genuinely-bivariate frequencyBandsRelevance unchanged.
 */
function toBivariateBands(b: FrequencyBands) {
  const v = (x: number) => ({ magnitude: Math.abs(x), direction: x })
  return { low: v(b.low), mid: v(b.mid), high: v(b.high) }
}

// Panel-wide Relevance/Confidence toggle (B4). Switches all three layers at once.
function ViewToggle({
  view,
  onChange,
}: {
  view: AudioView
  onChange: (v: AudioView) => void
}) {
  const options: { value: AudioView; label: string }[] = [
    { value: 'relevance', label: 'RELEVANCE' },
    { value: 'confidence', label: 'CONFIDENCE' },
  ]
  return (
    <div
      style={{
        display: 'flex',
        gap: 2,
        padding: 2,
        borderRadius: 6,
        backgroundColor: '#0d0f14',
        border: '1px solid #2a2f42',
      }}
    >
      {options.map(opt => {
        const active = view === opt.value
        return (
          <button
            key={opt.value}
            type="button"
            onClick={() => onChange(opt.value)}
            style={{
              fontSize: 9,
              fontFamily: 'monospace',
              letterSpacing: '0.1em',
              padding: '3px 8px',
              borderRadius: 4,
              border: 'none',
              cursor: 'pointer',
              color: active ? '#0d0f14' : '#8b92a8',
              backgroundColor: active ? '#00e5ff' : 'transparent',
              fontWeight: active ? 700 : 400,
              transition: 'background-color 0.15s, color 0.15s',
            }}
          >
            {opt.label}
          </button>
        )
      })}
    </div>
  )
}

// Layer sub-header label
function LayerLabel({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        fontSize: 9,
        fontFamily: 'monospace',
        letterSpacing: '0.12em',
        color: '#4d5470',
        marginBottom: 6,
      }}
    >
      {children}
    </div>
  )
}

// Dark inner card for each layer
function LayerCard({
  children,
  maxHeight,
}: {
  children: React.ReactNode
  maxHeight?: number
}) {
  return (
    <div
      style={{
        backgroundColor: '#0d0f14',
        borderRadius: 8,
        padding: '12px 14px',
        border: '1px solid #2a2f42',
        ...(maxHeight ? { maxHeight, overflowY: 'auto' as const } : {}),
      }}
    >
      {children}
    </div>
  )
}

export function AudioLayers({ result, clip, videoRef }: AudioLayersProps) {
  const audio = result?.audio ?? null
  const duration = clip?.duration ?? 1
  const [view, setView] = useState<AudioView>('relevance')

  return (
    <div
      className="rounded-lg p-4"
      style={{ backgroundColor: '#141720', border: '1px solid #2a2f42' }}
    >
      {/* Section header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          marginBottom: 12,
        }}
      >
        <span
          style={{
            fontSize: 10,
            fontFamily: 'monospace',
            letterSpacing: '0.18em',
            color: '#4d5470',
          }}
        >
          AUDIO ANALYSIS
        </span>
        <div style={{ flex: 1, height: 1, backgroundColor: '#1e2233' }} />
        {audio !== null && <ViewToggle view={view} onChange={setView} />}
        <span style={{ fontSize: 10, fontFamily: 'monospace', color: '#2a2f42' }}>
          Wav2Vec 2.0 · AttnLRP
        </span>
      </div>

      <AnimatePresence mode="wait">
        {audio === null ? (
          <motion.div
            key="idle"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: 8,
              padding: '20px 0',
            }}
          >
            {/* Dim waveform bars */}
            <svg width="44" height="32" viewBox="0 0 44 32" fill="none">
              <rect x="0" y="12" width="4" height="8" rx="2" fill="#1b1f2e" />
              <rect x="6" y="6" width="4" height="20" rx="2" fill="#1b1f2e" />
              <rect x="12" y="2" width="4" height="28" rx="2" fill="#1b1f2e" />
              <rect x="18" y="8" width="4" height="16" rx="2" fill="#1b1f2e" />
              <rect x="24" y="4" width="4" height="24" rx="2" fill="#1b1f2e" />
              <rect x="30" y="10" width="4" height="12" rx="2" fill="#1b1f2e" />
              <rect x="36" y="14" width="4" height="4" rx="2" fill="#1b1f2e" />
            </svg>
            <span style={{ fontSize: 10, fontFamily: 'monospace', color: '#2a2f42' }}>
              Run analysis to see audio xAI
            </span>
          </motion.div>
        ) : (
          <motion.div
            key="content"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.4, ease: 'easeOut' }}
            style={{ display: 'flex', flexDirection: 'column', gap: 12 }}
          >
            {/* L1 — Waveform Relevance */}
            <div>
              <LayerLabel>
                L1 — WAVEFORM {view === 'confidence' ? 'CONFIDENCE' : 'RELEVANCE'}
              </LayerLabel>
              <LayerCard>
                <WaveformRelevanceLayer
                  audio={audio}
                  videoRef={videoRef}
                  duration={duration}
                  view={view}
                />
              </LayerCard>
            </div>

            {/* L2 — Word Tokens */}
            <div>
              <LayerLabel>L2 — WORD TOKENS</LayerLabel>
              {/* No maxHeight on the L2 card: the chart height is fixed
                  (independent of word count), so the card sizes exactly to its
                  content. A cap a few px below content made a scrollbar appear,
                  which shrank the ResponsiveContainer width and caused a
                  re-layout feedback loop (the visible "shaking"). */}
              {audio.wordSegments.length > 0 ? (
                <LayerCard>
                  <WordTokenChart
                    wordSegments={audio.wordSegments}
                    videoRef={videoRef}
                    view={view}
                  />
                </LayerCard>
              ) : (
                <LayerCard>
                  <div
                    style={{
                      fontSize: 10,
                      fontFamily: 'monospace',
                      color: '#4d5470',
                      letterSpacing: '0.08em',
                      padding: '6px 0',
                    }}
                  >
                    Word-level alignment unavailable
                  </div>
                </LayerCard>
              )}
            </div>

            {/* L3 — Frequency × Time (heatmap; falls back to the 3-bar chart for
                older caches / multimodal that don't provide the grids). */}
            <div>
              <LayerLabel>L3 — FREQUENCY × TIME</LayerLabel>
              <LayerCard>
                {audio.frequencyGridConfidence ? (
                  <FrequencyHeatmap
                    audio={audio}
                    view={view}
                    videoRef={videoRef}
                    duration={duration}
                  />
                ) : (
                  <FrequencyBandChart
                    bands={
                      view === 'confidence'
                        ? toBivariateBands(audio.frequencyBands)
                        : audio.frequencyBandsRelevance ??
                          toBivariateBands(audio.frequencyBands)
                    }
                    view={view}
                  />
                )}
              </LayerCard>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
