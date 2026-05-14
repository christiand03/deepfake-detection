/**
 * AudioLayers — right-panel audio xAI section; always visible.
 *
 * Shows an idle placeholder when no audio result is available.
 * Once analysis completes with audio, renders three stacked layers:
 *   L1 — WaveformRelevanceLayer  (Canvas)
 *   L2 — WordTokenChart          (Recharts BarChart, scrollable)
 *   L3 — FrequencyBandChart      (custom SVG horizontal bars)
 *
 * Syncs to the video element via `videoRef` → `useVideoTime`.
 */

import { motion, AnimatePresence } from 'framer-motion'
import { WaveformRelevanceLayer } from './WaveformRelevanceLayer'
import { WordTokenChart } from './WordTokenChart'
import { FrequencyBandChart } from './FrequencyBandChart'
import { useVideoTime } from '../../hooks/useVideoTime'
import type { AnalysisResult, ClipMeta } from '../../types/analysis'

interface AudioLayersProps {
  result: AnalysisResult | null
  clip: ClipMeta | null
  videoRef: React.RefObject<HTMLVideoElement | null>
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
  const currentTime = useVideoTime(videoRef)
  const audio = result?.audio ?? null
  const duration = clip?.duration ?? 1

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
              <LayerLabel>L1 — WAVEFORM RELEVANCE</LayerLabel>
              <LayerCard>
                <WaveformRelevanceLayer
                  audio={audio}
                  currentTime={currentTime}
                  duration={duration}
                />
              </LayerCard>
            </div>

            {/* L2 — Word Tokens */}
            {audio.wordSegments.length > 0 && (
              <div>
                <LayerLabel>L2 — WORD TOKENS</LayerLabel>
                <LayerCard maxHeight={160}>
                  <WordTokenChart
                    wordSegments={audio.wordSegments}
                    currentTime={currentTime}
                  />
                </LayerCard>
              </div>
            )}

            {/* L3 — Frequency Bands */}
            <div>
              <LayerLabel>L3 — FREQUENCY BANDS</LayerLabel>
              <LayerCard>
                <FrequencyBandChart bands={audio.frequencyBands} />
              </LayerCard>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
