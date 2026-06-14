/**
 * useAnalysis — state machine for clip analysis lifecycle.
 *
 * States: idle → scanning → done | error
 * Transitions: analyze() triggers scanning, resolves to done or error.
 * The current result is preserved between calls so the UI never goes blank.
 */

import { useCallback, useState } from 'react'
import { analyzeClip } from '../api/client'
import type { AnalysisResult, AnalysisState } from '../types/analysis'

export function useAnalysis() {
  const [state, setState] = useState<AnalysisState>({ status: 'idle' })

  const analyze = useCallback(
    async (
      clipId: string,
      opts?: { useMultimodal?: boolean; fusionMode?: 'cross_attention' | 'concat' },
    ) => {
      setState({ status: 'scanning' })
      try {
        const result: AnalysisResult = await analyzeClip(clipId, opts)
        setState({ status: 'done', result })
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        setState({ status: 'error', message })
      }
    },
    [],
  )

  const reset = useCallback(() => setState({ status: 'idle' }), [])

  return { state, analyze, reset }
}
