/**
 * useVideoSync — subscribes to a <video> element's timeupdate event and
 * returns the current frame index for heatmap overlay synchronisation.
 *
 * The frame index is floored so HeatmapCanvas can use it as an array index
 * into the heatmapFrames array without needing to clamp itself.
 */

import { useEffect, useState } from 'react'

export function useVideoSync(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  fps: number,
  totalFrames: number,
): number {
  const [frameIndex, setFrameIndex] = useState(0)

  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    function onTimeUpdate() {
      const raw = Math.floor(video!.currentTime * fps)
      setFrameIndex(Math.max(0, Math.min(raw, totalFrames - 1)))
    }

    video.addEventListener('timeupdate', onTimeUpdate)
    return () => video.removeEventListener('timeupdate', onTimeUpdate)
  }, [videoRef, fps, totalFrames])

  return frameIndex
}
