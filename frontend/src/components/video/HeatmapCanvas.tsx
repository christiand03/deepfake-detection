/**
 * HeatmapCanvas — draws the per-frame AttnLRP overlay onto a <canvas>
 * that is positioned absolutely over the <video> element.
 *
 * Each frame's heatmap is a base64/SVG data URI. On each timeupdate the
 * current frame is loaded into an Image and composited at 50% opacity using
 * the "multiply" (or normal) blend mode so the video underneath is visible.
 */

import { useEffect, useRef } from 'react'

interface HeatmapCanvasProps {
  heatmapFrames: string[]
  frameIndex: number
  opacity: number
  /** Matches the natural size of the video element */
  width: number
  height: number
}

export function HeatmapCanvas({
  heatmapFrames,
  frameIndex,
  opacity,
  width,
  height,
}: HeatmapCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgCache = useRef<Map<number, HTMLImageElement>>(new Map())

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const src = heatmapFrames[frameIndex]
    if (!src) {
      ctx.clearRect(0, 0, width, height)
      return
    }

    // Use cached image if available
    const cached = imgCache.current.get(frameIndex)
    if (cached) {
      ctx.clearRect(0, 0, width, height)
      ctx.globalAlpha = opacity
      ctx.drawImage(cached, 0, 0, width, height)
      return
    }

    const img = new Image()
    img.onload = () => {
      imgCache.current.set(frameIndex, img)
      ctx.clearRect(0, 0, width, height)
      ctx.globalAlpha = opacity
      ctx.drawImage(img, 0, 0, width, height)
    }
    img.src = src
  }, [frameIndex, heatmapFrames, opacity, width, height])

  // Clear cache when frames change (new analysis)
  useEffect(() => {
    imgCache.current.clear()
  }, [heatmapFrames])

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{
        position: 'absolute',
        inset: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        borderRadius: 8,
        mixBlendMode: 'screen',
      }}
    />
  )
}
