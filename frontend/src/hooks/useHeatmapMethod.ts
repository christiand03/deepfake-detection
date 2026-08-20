/**
 * useHeatmapMethod — picks which explanation method renders the player overlay.
 *
 * Scope, and the reason this is a hook of its own rather than a flag inside
 * `useAnalysis` (docs/chefer_ablation.md §5): the method switch swaps the video overlay
 * and NOTHING else. Verdict, confidence and relevance timelines, region scores and
 * Phase 3/4 keep running on the bivariate AttnLRP result that `useAnalysis` holds. Keeping
 * the two states apart means a method change can never invalidate or reshape the analysis.
 *
 * `bivariate` is free — those frames already came with the analysis. The other two are
 * fetched lazily on first selection and then kept for the session, so flipping back and
 * forth costs one request per (clip, method) and nothing after that. The backend caches to
 * disk as well, so even a page reload is cheap.
 *
 * The selection carries the clip it was made for, and the effective method is derived
 * during render. Selecting Chefer on clip A and then switching to clip B therefore falls
 * back to the default view without an effect that resets state after the fact — which
 * would render once with A's frames under B's video before correcting itself.
 */

import { useCallback, useState } from 'react'
import { fetchHeatmap } from '../api/client'
import type { HeatmapMethod } from '../types/analysis'

interface UseHeatmapMethodArgs {
  clipId: string
  /** Frames from the analysis — the `bivariate` stage, and the fallback while loading. */
  bivariateFrames: string[] | null
  onError?: (message: string) => void
}

/** A method choice is only meaningful together with the clip it was made for. */
interface Selection {
  clipId: string
  method: HeatmapMethod
}

const cacheKey = (clipId: string, method: HeatmapMethod) => `${clipId}::${method}`

export function useHeatmapMethod({ clipId, bivariateFrames, onError }: UseHeatmapMethodArgs) {
  const [selection, setSelection] = useState<Selection>({ clipId, method: 'bivariate' })
  const [loading, setLoading] = useState<Selection | null>(null)
  const [fetched, setFetched] = useState<Record<string, string[]>>({})

  // Derived, not stored: a stale selection from another clip never reaches the player.
  const method: HeatmapMethod = selection.clipId === clipId ? selection.method : 'bivariate'
  const isLoading = loading !== null && loading.clipId === clipId

  const setMethod = useCallback(
    async (next: HeatmapMethod) => {
      const target: Selection = { clipId, method: next }
      setSelection(target)
      if (next === 'bivariate') return

      const key = cacheKey(clipId, next)
      if (fetched[key]) return

      setLoading(target)
      try {
        const result = await fetchHeatmap(clipId, next)
        setFetched(prev => ({ ...prev, [key]: result.heatmapFrames }))
      } catch (err) {
        // Fall back to the default view rather than leaving the player blank: the
        // analysis itself is unaffected by this failure.
        onError?.(err instanceof Error ? err.message : String(err))
        setSelection({ clipId, method: 'bivariate' })
      } finally {
        setLoading(null)
      }
    },
    [clipId, fetched, onError],
  )

  const frames =
    method === 'bivariate' ? bivariateFrames : (fetched[cacheKey(clipId, method)] ?? bivariateFrames)

  return { method, setMethod, frames, isLoading }
}
