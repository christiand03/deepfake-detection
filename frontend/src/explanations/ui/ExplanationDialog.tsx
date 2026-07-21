/**
 * ExplanationDialog — the single, reusable popup for every visual's explanation.
 *
 * The structure is coded here ONCE; the content is whatever `Explanation` the
 * caller passes (resolved from a `VisualId` by the registry). Renders into a
 * portal, closes on backdrop click or Escape, and orders sections canonically.
 */

import { useEffect } from 'react'
import { createPortal } from 'react-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { SECTION_ORDER, type Explanation } from '../types'
import { SectionBlock } from './SectionBlock'
import { CvRBadge } from './widgets'

export function ExplanationDialog({
  explanation,
  onClose,
}: {
  explanation: Explanation | null
  onClose: () => void
}) {
  // Close on Escape whenever a dialog is open.
  useEffect(() => {
    if (!explanation) return
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [explanation, onClose])

  const sections = explanation
    ? [...explanation.sections].sort(
        (a, b) => SECTION_ORDER.indexOf(a.kind) - SECTION_ORDER.indexOf(b.kind),
      )
    : []

  return createPortal(
    <AnimatePresence>
      {explanation && (
        <motion.div
          key="backdrop"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
          onClick={onClose}
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 10000,
            backgroundColor: 'rgba(4,6,10,0.72)',
            backdropFilter: 'blur(3px)',
            display: 'flex',
            alignItems: 'flex-start',
            justifyContent: 'center',
            padding: '48px 20px',
            overflowY: 'auto',
          }}
        >
          <motion.div
            key="panel"
            role="dialog"
            aria-modal="true"
            aria-label={explanation.title}
            initial={{ opacity: 0, y: 12, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 8, scale: 0.99 }}
            transition={{ duration: 0.2, ease: 'easeOut' }}
            onClick={e => e.stopPropagation()}
            style={{
              width: '100%',
              maxWidth: 620,
              backgroundColor: '#141720',
              border: '1px solid #2a2f42',
              borderRadius: 12,
              boxShadow: '0 24px 80px rgba(0,0,0,0.6)',
              overflow: 'hidden',
            }}
          >
            {/* ── Header ── */}
            <div
              style={{
                position: 'sticky',
                top: 0,
                zIndex: 1,
                padding: '16px 20px 14px',
                borderBottom: '1px solid #2a2f42',
                background: 'linear-gradient(180deg, #171b26 0%, #141720 100%)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12 }}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 10,
                      flexWrap: 'wrap',
                    }}
                  >
                    <h2
                      style={{
                        margin: 0,
                        fontSize: 16,
                        fontWeight: 700,
                        color: '#f2f4f8',
                        letterSpacing: '0.01em',
                      }}
                    >
                      {explanation.title}
                    </h2>
                    {explanation.cvr && <CvRBadge cvr={explanation.cvr} />}
                  </div>
                  {explanation.subtitle && (
                    <p
                      style={{
                        margin: '5px 0 0',
                        fontSize: 12,
                        lineHeight: 1.5,
                        color: '#8b92a8',
                      }}
                    >
                      {explanation.subtitle}
                    </p>
                  )}
                  {explanation.method && (
                    <div
                      style={{
                        marginTop: 7,
                        fontFamily: 'monospace',
                        fontSize: 9.5,
                        letterSpacing: '0.08em',
                        color: '#4d5470',
                      }}
                    >
                      {explanation.method}
                    </div>
                  )}
                </div>
                <button
                  onClick={onClose}
                  aria-label="Schließen"
                  style={{
                    flexShrink: 0,
                    width: 28,
                    height: 28,
                    borderRadius: 7,
                    border: '1px solid #2a2f42',
                    background: '#0d0f14',
                    color: '#8b92a8',
                    fontSize: 16,
                    lineHeight: 1,
                    cursor: 'pointer',
                  }}
                >
                  ×
                </button>
              </div>
            </div>

            {/* ── Body ── */}
            <div style={{ padding: '18px 20px 22px' }}>
              {sections.map((s, i) => (
                <SectionBlock key={`${s.kind}-${i}`} section={s} />
              ))}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>,
    document.body,
  )
}
