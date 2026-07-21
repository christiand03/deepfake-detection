/**
 * ExplanationContext — global controller for the single explanation dialog (F1).
 *
 * Any component calls `useExplanation().open('heatmap-overlay')` to open the
 * popup for a visual; the provider owns the one dialog instance. Mirrors the
 * `ErrorToastContext` pattern.
 */

import { createContext, useCallback, useContext, useMemo, useState } from 'react'
import { getExplanation } from '../explanations/registry'
import { ExplanationDialog } from '../explanations/ui/ExplanationDialog'
import type { VisualId } from '../explanations/types'

interface ExplanationContextValue {
  open: (id: VisualId) => void
  /** True if the given visual has authored content (button can render). */
  has: (id: VisualId) => boolean
}

const ExplanationContext = createContext<ExplanationContextValue>({
  open: () => undefined,
  has: () => false,
})

export function ExplanationProvider({ children }: { children: React.ReactNode }) {
  const [openId, setOpenId] = useState<VisualId | null>(null)

  const value = useMemo<ExplanationContextValue>(
    () => ({
      open: id => setOpenId(id),
      has: id => getExplanation(id) !== undefined,
    }),
    [],
  )

  const close = useCallback(() => setOpenId(null), [])
  const explanation = openId ? getExplanation(openId) ?? null : null

  return (
    <ExplanationContext.Provider value={value}>
      {children}
      <ExplanationDialog explanation={explanation} onClose={close} />
    </ExplanationContext.Provider>
  )
}

export function useExplanation() {
  return useContext(ExplanationContext)
}
