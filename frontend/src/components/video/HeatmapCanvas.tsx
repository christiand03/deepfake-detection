/**
 * HeatmapCanvas — overlays the per-frame AttnLRP heatmap onto the video.
 *
 * Uses an <img> with objectFit: contain so its scaling and position match the
 * <video> element exactly — no manual coordinate calculation required. The
 * heatmap PNG is RGBA with transparent pixels outside the face crop, so only
 * the face region is coloured.
 *
 * Frames are preloaded into an Image cache when a new analysis arrives so
 * that scrubbing through the timeline does not cause flicker.
 */

import { useEffect, useRef } from 'react'

interface HeatmapCanvasProps {
  heatmapFrames: string[]
  frameIndex: number
  opacity: number
}

export function HeatmapCanvas({ heatmapFrames, frameIndex, opacity }: HeatmapCanvasProps) {
  const imgRef = useRef<HTMLImageElement>(null)
  const preloadCache = useRef<Map<number, HTMLImageElement>>(new Map())

  // Preload all frames when a new analysis result arrives.
  useEffect(() => {
    preloadCache.current.clear()
    heatmapFrames.forEach((src, i) => {
      const preImg = new Image()
      preImg.src = src
      preloadCache.current.set(i, preImg)
    })
  }, [heatmapFrames])

  // Switch frames imperatively to avoid React re-render flicker.
  useEffect(() => {
    const el = imgRef.current
    if (!el) return
    const src = heatmapFrames[frameIndex]
    if (src) el.src = src
  }, [frameIndex, heatmapFrames])

  return (
    <img
      ref={imgRef}
      src={heatmapFrames[0] ?? undefined}
      alt=""
      style={{
        position: 'absolute',
        inset: 0,
        width: '100%',
        height: '100%',
        objectFit: 'contain',
        opacity,
        // 'normal' overlays the patch colour directly (clearly visible on dark
        // backgrounds); 'screen' only lightens and washes colours out on dark,
        // blue-toned video. The magnitude-based alpha keeps edges seamless.
        mixBlendMode: 'normal',
        pointerEvents: 'none',
      }}
    />
  )
}
