/**
 * XaiModeToggle — pill toggle for switching between xAI explanation methods.
 *
 * The active method is highlighted in accent-cyan. Shown in the right verdict
 * panel so analysts can change the explanation method before running analysis.
 */

import type { XaiMode } from '../../types/analysis'

interface XaiModeToggleProps {
  value: XaiMode
  onChange: (mode: XaiMode) => void
  disabled?: boolean
}

const MODES: { value: XaiMode; label: string; description: string }[] = [
  {
    value: 'rollout',
    label: 'Attention Rollout',
    description: 'Full transformer attention propagation',
  },
  {
    value: 'lrp',
    label: 'AttnLRP',
    description: 'Layer-wise Relevance Propagation',
  },
]

export function XaiModeToggle({ value, onChange, disabled }: XaiModeToggleProps) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {MODES.map(m => {
        const active = m.value === value
        return (
          <button
            key={m.value}
            onClick={() => !disabled && onChange(m.value)}
            disabled={disabled}
            style={{
              padding: '9px 12px',
              borderRadius: 6,
              textAlign: 'left',
              border: `1px solid ${active ? 'rgba(0,229,255,0.4)' : '#2a2f42'}`,
              backgroundColor: active ? 'rgba(0,229,255,0.08)' : 'transparent',
              cursor: disabled ? 'not-allowed' : 'pointer',
              transition: 'all 0.15s',
              opacity: disabled ? 0.5 : 1,
              outline: 'none',
              width: '100%',
            }}
          >
            <div
              style={{
                fontSize: 12,
                fontFamily: 'monospace',
                fontWeight: active ? 600 : 400,
                color: active ? '#00e5ff' : '#8b92a8',
                letterSpacing: '0.04em',
                marginBottom: 2,
              }}
            >
              {active && (
                <span style={{ marginRight: 6, color: '#00e5ff' }}>●</span>
              )}
              {m.label}
            </div>
            <div
              style={{
                fontSize: 10,
                fontFamily: 'monospace',
                color: '#4d5470',
                letterSpacing: '0.03em',
              }}
            >
              {m.description}
            </div>
          </button>
        )
      })}
    </div>
  )
}
