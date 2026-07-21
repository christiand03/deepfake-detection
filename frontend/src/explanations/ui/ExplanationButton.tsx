/**
 * ExplanationButton — the little "?" info button placed next to a visual.
 *
 * Renders nothing if the visual has no authored explanation yet, so it can be
 * dropped next to every visual now and light up as content lands.
 */

import { useExplanation } from '../../context/ExplanationContext'
import type { VisualId } from '../types'

export function ExplanationButton({
  id,
  label = 'Erklärung',
  size = 18,
}: {
  id: VisualId
  /** Accessible label / tooltip; the visible glyph is always "?". */
  label?: string
  size?: number
}) {
  const { open, has } = useExplanation()
  if (!has(id)) return null

  return (
    <button
      type="button"
      onClick={e => {
        e.stopPropagation()
        open(id)
      }}
      title={label}
      aria-label={label}
      style={{
        width: size,
        height: size,
        flexShrink: 0,
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: '50%',
        border: '1px solid #2a2f42',
        backgroundColor: 'rgba(13,15,20,0.85)',
        color: '#8b92a8',
        fontFamily: 'monospace',
        fontSize: Math.round(size * 0.62),
        fontWeight: 700,
        lineHeight: 1,
        cursor: 'pointer',
        padding: 0,
        transition: 'color 0.15s, border-color 0.15s',
      }}
      onMouseEnter={e => {
        e.currentTarget.style.color = '#00e5ff'
        e.currentTarget.style.borderColor = 'rgba(0,229,255,0.5)'
      }}
      onMouseLeave={e => {
        e.currentTarget.style.color = '#8b92a8'
        e.currentTarget.style.borderColor = '#2a2f42'
      }}
    >
      ?
    </button>
  )
}
