/**
 * useVideoTime — subscribes to a <video> element's timeupdate event and
 * returns the current playback time in seconds.
 *
 * Used by audio layer components to sync visualisations to video playback
 * without coupling them to the frame-index used by the heatmap overlay.
 */

import { useEffect, useState } from 'react'

export function useVideoTime(
  videoRef: React.RefObject<HTMLVideoElement | null>,
): number {
  const [currentTime, setCurrentTime] = useState(0)

  useEffect(() => {
    let cleanupFn: (() => void) | null = null

    function attach(): boolean {
      const video = videoRef.current
      if (!video) return false
      function onTimeUpdate() {
        setCurrentTime(video!.currentTime)
      }
      video.addEventListener('timeupdate', onTimeUpdate)
      cleanupFn = () => video.removeEventListener('timeupdate', onTimeUpdate)
      return true
    }

    if (attach()) {
      return () => cleanupFn?.()
    }

    // Video element not yet in DOM — poll until it appears
    const id = setInterval(() => {
      if (attach()) clearInterval(id)
    }, 50)

    return () => {
      clearInterval(id)
      cleanupFn?.()
    }
  }, [videoRef])

  return currentTime
}
