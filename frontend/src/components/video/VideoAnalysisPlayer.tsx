/**
 * VideoAnalysisPlayer — stacks <video> + <HeatmapCanvas> + scan overlay.
 *
 * The scan animation is a moving horizontal gradient line that plays while
 * state === 'scanning'. On completion the heatmap fades in.
 */

import { forwardRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { HeatmapCanvas } from './HeatmapCanvas'
import type { ClipMeta } from '../../types/analysis'

interface VideoAnalysisPlayerProps {
  clip: ClipMeta
  heatmapFrames: string[] | null
  frameIndex: number
  isScanning: boolean
  /** 0–1 heatmap overlay opacity */
  heatmapOpacity?: number
}

const VIDEO_W = 640
const VIDEO_H = 360

export const VideoAnalysisPlayer = forwardRef<HTMLVideoElement, VideoAnalysisPlayerProps>(
  function VideoAnalysisPlayer(
    { clip, heatmapFrames, frameIndex, isScanning, heatmapOpacity = 0.55 },
    ref,
  ) {
    const [videoLoaded, setVideoLoaded] = useState(false)

    return (
      <div
        className="relative w-full rounded-lg overflow-hidden"
        style={{
          aspectRatio: '16/9',
          backgroundColor: '#0a0c10',
          border: '1px solid #2a2f42',
        }}
      >
        {/* Video element */}
        <video
          ref={ref}
          src={clip.videoSrc}
          poster={clip.posterSrc}
          controls
          onLoadedData={() => setVideoLoaded(true)}
          style={{
            width: '100%',
            height: '100%',
            display: 'block',
            objectFit: 'cover',
          }}
        />

        {/* Heatmap overlay — fades in after analysis */}
        <AnimatePresence>
          {heatmapFrames && heatmapFrames.length > 0 && (
            <motion.div
              key="heatmap"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.6 }}
              style={{ position: 'absolute', inset: 0 }}
            >
              <HeatmapCanvas
                heatmapFrames={heatmapFrames}
                frameIndex={frameIndex}
                opacity={heatmapOpacity}
                width={VIDEO_W}
                height={VIDEO_H}
              />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Scanning overlay */}
        <AnimatePresence>
          {isScanning && (
            <motion.div
              key="scan"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              style={{
                position: 'absolute',
                inset: 0,
                backgroundColor: 'rgba(13,15,20,0.6)',
                borderRadius: 8,
                overflow: 'hidden',
              }}
            >
              {/* Pulsing scan line */}
              <motion.div
                initial={{ top: '-4px' }}
                animate={{ top: 'calc(100% + 4px)' }}
                transition={{ duration: 1.6, repeat: Infinity, ease: 'linear' }}
                style={{
                  position: 'absolute',
                  left: 0,
                  right: 0,
                  height: 3,
                  background:
                    'linear-gradient(90deg, transparent 0%, #00e5ff 20%, #00e5ff 80%, transparent 100%)',
                  boxShadow: '0 0 12px 3px rgba(0,229,255,0.6)',
                }}
              />

              {/* Label */}
              <div
                style={{
                  position: 'absolute',
                  inset: 0,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <div
                  style={{
                    fontFamily: 'monospace',
                    fontSize: 13,
                    letterSpacing: '0.15em',
                    color: '#00e5ff',
                    opacity: 0.9,
                  }}
                >
                  ANALYZING…
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Placeholder when video not yet loaded */}
        {!videoLoaded && !isScanning && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#4d5470',
            }}
          >
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
              <polygon points="5,3 19,12 5,21" />
            </svg>
          </div>
        )}
      </div>
    )
  },
)
