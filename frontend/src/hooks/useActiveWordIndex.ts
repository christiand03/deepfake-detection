/**
 * useActiveWordIndex — index of the word segment currently being spoken,
 * tracked via a requestAnimationFrame loop on the <video> element.
 *
 * Unlike consuming a 60 Hz currentTime value, this only triggers a React
 * re-render when the ACTIVE INDEX actually changes (a few times per second), so
 * the consuming chart stays stable (no per-frame jitter) while still catching
 * short words — the index is recomputed on every animation frame.
 *
 * Returns -1 when no word is active (silence / outside any segment).
 */

import { useEffect, useRef, useState } from 'react'
import type { WordSegment } from '../types/analysis'

export function useActiveWordIndex(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  wordSegments: WordSegment[],
): number {
  const [activeIdx, setActiveIdx] = useState(-1)
  // Keep the latest segments in a ref so the rAF loop is set up once and never
  // restarts when a new analysis arrives.
  const segmentsRef = useRef(wordSegments)
  useEffect(() => {
    segmentsRef.current = wordSegments
  }, [wordSegments])

  useEffect(() => {
    let rafId = 0
    let last = -1

    function tick() {
      const video = videoRef.current
      if (video) {
        const t = video.currentTime
        const idx = segmentsRef.current.findIndex(w => t >= w.start && t <= w.end)
        if (idx !== last) {
          last = idx
          setActiveIdx(idx)
        }
      }
      rafId = requestAnimationFrame(tick)
    }

    rafId = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafId)
  }, [videoRef])

  return activeIdx
}
