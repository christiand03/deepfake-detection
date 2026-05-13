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
    const video = videoRef.current
    if (!video) return

    function onTimeUpdate() {
      setCurrentTime(video!.currentTime)
    }

    video.addEventListener('timeupdate', onTimeUpdate)
    return () => video.removeEventListener('timeupdate', onTimeUpdate)
  }, [videoRef])

  return currentTime
}
