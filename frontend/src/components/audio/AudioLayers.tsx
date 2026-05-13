/**
 * AudioLayers — bottom panel of the dashboard; visible when result.audio !== null.
 *
 * Contains three labelled analysis layers:
 *   Layer 1 — WaveformRelevanceLayer  (Canvas)
 *   Layer 2 — WordTokenChart          (Recharts BarChart)
 *   Layer 3 — FrequencyBandChart      (custom SVG horizontal bars)
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

export function AudioLayers({ result, clip, videoRef }: AudioLayersProps) {
  const currentTime = useVideoTime(videoRef)

  const audio = result?.audio ?? null
  const duration = clip?.duration ?? 1

  const isVisible = audio !== null

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          key="audio-panel"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 8 }}
          transition={{ duration: 0.45, ease: 'easeOut' }}
          style={{ borderTop: '1px solid #2a2f42' }}
        >
          {/* Header */}
          <div
            style={{
              padding: '12px 20px 0',
              display: 'flex',
              alignItems: 'center',
              gap: 10,
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
            <div
              style={{
                flex: 1,
                height: 1,
                backgroundColor: '#1e2233',
              }}
            />
            <span
              style={{
                fontSize: 10,
                fontFamily: 'monospace',
                color: '#2a2f42',
              }}
            >
              Wav2Vec 2.0 · AttnLRP
            </span>
          </div>

          {/* Three-column grid for the layers */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr',
              gap: 0,
              padding: '12px 20px',
            }}
          >
            {/* Layer 1 — waveform + relevance */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.3, delay: 0.05 }}
              style={{
                backgroundColor: '#141720',
                borderRadius: 8,
                padding: '12px 14px',
                border: '1px solid #2a2f42',
                marginBottom: 10,
              }}
            >
              <WaveformRelevanceLayer
                audio={audio}
                currentTime={currentTime}
                duration={duration}
              />
            </motion.div>

            {/* Layers 2 + 3 side by side */}
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr 320px',
                gap: 10,
              }}
            >
              {/* Layer 2 — word token chart */}
              {audio.wordSegments.length > 0 && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.3, delay: 0.12 }}
                  style={{
                    backgroundColor: '#141720',
                    borderRadius: 8,
                    padding: '12px 14px',
                    border: '1px solid #2a2f42',
                  }}
                >
                  <WordTokenChart
                    wordSegments={audio.wordSegments}
                    currentTime={currentTime}
                  />
                </motion.div>
              )}

              {/* Layer 3 — frequency band chart */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.3, delay: 0.2 }}
                style={{
                  backgroundColor: '#141720',
                  borderRadius: 8,
                  padding: '12px 14px',
                  border: '1px solid #2a2f42',
                }}
              >
                <FrequencyBandChart bands={audio.frequencyBands} />
              </motion.div>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
