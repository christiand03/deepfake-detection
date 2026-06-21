/**
 * CropComparisonPlayer — Phase-3/4 whole-clip before/after viewer (I2).
 *
 * Replaces the old single "Frame #8" still. Shows two face-crop videos side by
 * side (left = before, right = after) with the crop-space AttnLRP heatmap (224)
 * overlaid on each. A single slider controls the VIDEO opacity of both players
 * (default 100 % … 0 % = video off, only the heatmaps) so the user can fade the
 * footage out and focus on the relevance patches.
 *
 * The two players are kept in lockstep (play / pause / seek / rate mirrored, plus
 * drift correction) so the comparison is always frame-aligned. The heatmap frame
 * is synced to each video's own playback position (its frame count equals the
 * video's frame count, so currentTime/duration maps to an index).
 */

import { useEffect, useRef, useState } from 'react'

interface CropPlayerSide {
  label: string
  /** Served /media URL of the face-crop video, or null when unavailable. */
  videoUrl?: string | null
  /** Crop-space heatmap frames (data URIs), one per video frame. */
  heatmapFrames: string[]
  /** Border accent colour for the player. */
  accent: string
}

interface CropComparisonPlayerProps {
  title: string
  left: CropPlayerSide
  right: CropPlayerSide
}

function CropPlayer({
  side,
  videoOpacity,
  videoRef,
}: {
  side: CropPlayerSide
  videoOpacity: number
  videoRef: React.RefObject<HTMLVideoElement | null>
}) {
  const imgRef = useRef<HTMLImageElement>(null)
  const lastIdx = useRef(-1)

  // Sync the heatmap frame to the video position via rAF (no React re-render per
  // frame). Imperative img.src swap mirrors HeatmapCanvas.
  useEffect(() => {
    lastIdx.current = -1
    const video = videoRef.current
    const img = imgRef.current
    const frames = side.heatmapFrames
    if (!video || !img || frames.length === 0) return
    let raf = 0
    const tick = () => {
      const d = video.duration
      if (d && Number.isFinite(d) && d > 0) {
        const idx = Math.min(
          frames.length - 1,
          Math.max(0, Math.round((video.currentTime / d) * (frames.length - 1))),
        )
        if (idx !== lastIdx.current) {
          lastIdx.current = idx
          img.src = frames[idx]
        }
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [side.heatmapFrames, videoRef])

  return (
    <div style={{ flex: 1, minWidth: 0 }}>
      <div
        style={{
          position: 'relative',
          width: '100%',
          aspectRatio: '1 / 1',
          borderRadius: 6,
          overflow: 'hidden',
          border: `1px solid ${side.accent}`,
          backgroundColor: '#0a0c10',
        }}
      >
        {side.videoUrl ? (
          <video
            ref={videoRef}
            src={side.videoUrl}
            controls
            loop
            muted
            playsInline
            style={{
              position: 'absolute',
              inset: 0,
              width: '100%',
              height: '100%',
              objectFit: 'contain',
              opacity: videoOpacity,
              backgroundColor: '#000',
            }}
          />
        ) : (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 9,
              fontFamily: 'monospace',
              color: '#4d5470',
              textAlign: 'center',
              padding: 8,
            }}
          >
            video unavailable
          </div>
        )}
        <img
          ref={imgRef}
          src={side.heatmapFrames[0] ?? undefined}
          alt=""
          style={{
            position: 'absolute',
            inset: 0,
            width: '100%',
            height: '100%',
            objectFit: 'contain',
            pointerEvents: 'none',
          }}
        />
      </div>
      <div
        style={{
          fontSize: 11,
          fontFamily: 'monospace',
          fontWeight: 700,
          color: '#e8eaf0',
          marginTop: 5,
          textAlign: 'center',
          letterSpacing: '0.12em',
        }}
      >
        {side.label}
      </div>
    </div>
  )
}

export function CropComparisonPlayer({ title, left, right }: CropComparisonPlayerProps) {
  const [videoOpacity, setVideoOpacity] = useState(1)
  const leftVideoRef = useRef<HTMLVideoElement | null>(null)
  const rightVideoRef = useRef<HTMLVideoElement | null>(null)

  // Keep both players in lockstep so the before/after comparison is always
  // frame-aligned: play / pause / seek / rate on EITHER video is mirrored to the
  // other, and the left video drives a light drift correction during playback.
  // A single re-entrancy guard prevents the mirrored events from looping back.
  useEffect(() => {
    const a = leftVideoRef.current
    const b = rightVideoRef.current
    if (!a || !b) return
    let syncing = false

    const guard = (fn: () => void) => {
      if (syncing) return
      syncing = true
      try {
        fn()
      } finally {
        syncing = false
      }
    }

    const bind = (src: HTMLVideoElement, dst: HTMLVideoElement) => {
      const onPlay = () =>
        guard(() => {
          if (Math.abs(dst.currentTime - src.currentTime) > 0.05) dst.currentTime = src.currentTime
          void dst.play().catch(() => {})
        })
      const onPause = () =>
        guard(() => {
          dst.pause()
          dst.currentTime = src.currentTime
        })
      const onSeeked = () =>
        guard(() => {
          if (Math.abs(dst.currentTime - src.currentTime) > 0.03) dst.currentTime = src.currentTime
        })
      const onRate = () => guard(() => (dst.playbackRate = src.playbackRate))
      src.addEventListener('play', onPlay)
      src.addEventListener('pause', onPause)
      src.addEventListener('seeked', onSeeked)
      src.addEventListener('ratechange', onRate)
      return () => {
        src.removeEventListener('play', onPlay)
        src.removeEventListener('pause', onPause)
        src.removeEventListener('seeked', onSeeked)
        src.removeEventListener('ratechange', onRate)
      }
    }

    // Drift correction: nudge the right video toward the left during playback.
    const onDrift = () =>
      guard(() => {
        if (!a.paused && Math.abs(b.currentTime - a.currentTime) > 0.2) {
          b.currentTime = a.currentTime
        }
      })
    a.addEventListener('timeupdate', onDrift)

    const unbindAB = bind(a, b)
    const unbindBA = bind(b, a)
    return () => {
      a.removeEventListener('timeupdate', onDrift)
      unbindAB()
      unbindBA()
    }
  }, [left.videoUrl, right.videoUrl])

  return (
    <div>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'baseline',
          marginBottom: 6,
        }}
      >
        <span
          style={{
            fontSize: 9,
            fontFamily: 'monospace',
            color: '#4d5470',
            letterSpacing: '0.12em',
          }}
        >
          {title}
        </span>
        <span style={{ fontSize: 8, fontFamily: 'monospace', color: '#2a2f42' }}>SYNCED</span>
      </div>

      <div style={{ display: 'flex', gap: 8 }}>
        <CropPlayer side={left} videoOpacity={videoOpacity} videoRef={leftVideoRef} />
        <CropPlayer side={right} videoOpacity={videoOpacity} videoRef={rightVideoRef} />
      </div>

      {/* Video-opacity slider — fades the footage out to reveal the heatmaps. */}
      <div style={{ marginTop: 8 }}>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            fontSize: 9,
            fontFamily: 'monospace',
            color: '#4d5470',
            marginBottom: 3,
          }}
        >
          <span>VIDEO OPACITY</span>
          <span style={{ color: '#00e5ff' }}>{Math.round(videoOpacity * 100)}%</span>
        </div>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={videoOpacity}
          onChange={e => setVideoOpacity(Number(e.target.value))}
          style={{ width: '100%', accentColor: '#00e5ff', cursor: 'pointer' }}
        />
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            fontSize: 8,
            fontFamily: 'monospace',
            color: '#2a2f42',
            marginTop: 2,
          }}
        >
          <span>Heatmap only</span>
          <span>Video</span>
        </div>
      </div>
    </div>
  )
}
