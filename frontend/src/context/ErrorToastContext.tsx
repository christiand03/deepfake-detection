/**
 * ErrorToastContext — lightweight global error notification system.
 *
 * Usage:
 *   const { showError } = useErrorToast()
 *   showError('POST /api/robustness → 503: Model not ready')
 */

import { createContext, useCallback, useContext, useRef, useState } from 'react'

interface ErrorToastContextValue {
  showError: (message: string) => void
}

const ErrorToastContext = createContext<ErrorToastContextValue>({
  showError: () => undefined,
})

interface Toast {
  id: number
  message: string
}

const DISMISS_MS = 5_000

export function ErrorToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])
  const counterRef = useRef(0)

  const showError = useCallback((message: string) => {
    const id = ++counterRef.current
    setToasts(prev => [...prev, { id, message }])
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id))
    }, DISMISS_MS)
  }, [])

  return (
    <ErrorToastContext.Provider value={{ showError }}>
      {children}
      <ErrorToastStack toasts={toasts} onDismiss={id => setToasts(prev => prev.filter(t => t.id !== id))} />
    </ErrorToastContext.Provider>
  )
}

export function useErrorToast() {
  return useContext(ErrorToastContext)
}

// ── Toast stack renderer ──────────────────────────────────────────────────────

function ErrorToastStack({
  toasts,
  onDismiss,
}: {
  toasts: Toast[]
  onDismiss: (id: number) => void
}) {
  if (toasts.length === 0) return null

  return (
    <div
      style={{
        position: 'fixed',
        bottom: 24,
        right: 24,
        zIndex: 9999,
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
        maxWidth: 420,
      }}
    >
      {toasts.map(t => (
        <div
          key={t.id}
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: 10,
            padding: '10px 14px',
            borderRadius: 8,
            backgroundColor: '#1b1f2e',
            border: '1px solid rgba(239,68,68,0.4)',
            boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
            animation: 'fadeInUp 0.2s ease',
          }}
        >
          {/* Error icon */}
          <span style={{ color: '#ef4444', fontSize: 14, lineHeight: 1.4, flexShrink: 0 }}>⚠</span>

          {/* Message */}
          <span
            style={{
              flex: 1,
              fontSize: 11,
              fontFamily: 'monospace',
              color: '#e8eaf0',
              lineHeight: 1.5,
              wordBreak: 'break-word',
            }}
          >
            {t.message}
          </span>

          {/* Dismiss button */}
          <button
            onClick={() => onDismiss(t.id)}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              color: '#4d5470',
              fontSize: 14,
              lineHeight: 1,
              padding: 0,
              flexShrink: 0,
            }}
            aria-label="Dismiss"
          >
            ×
          </button>
        </div>
      ))}
    </div>
  )
}
