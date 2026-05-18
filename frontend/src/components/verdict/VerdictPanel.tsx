/**
 * VerdictPanel — right column of the main layout (Phase 3).
 *
 * Shows:
 *  • Idle:    XaiModeToggle + "run analysis" empty state
 *  • Scanning: skeleton shimmer blocks + XaiModeToggle (disabled)
 *  • Done:    VerdictGauge + AnomalyRegionBars + XaiModeToggle
 */

import { motion, AnimatePresence } from 'framer-motion'
import { VerdictGauge } from './VerdictGauge'
import type { AnalysisResult, ClipMeta } from '../../types/analysis'

interface VerdictPanelProps {
  result: AnalysisResult | null
  clip: ClipMeta | null
  isScanning: boolean
}

// ── Skeleton shimmer block ───────────────────────────────────────────────────

function SkeletonBlock({ height, style }: { height: number; style?: React.CSSProperties }) {
  return (
    <div
      className="shimmer"
      style={{
        height,
        borderRadius: 8,
        ...style,
      }}
    />
  )
}

// ── Section wrapper ──────────────────────────────────────────────────────────

function Section({
  title,
  children,
}: {
  title: string
  children: React.ReactNode
}) {
  return (
    <div
      className="rounded-lg p-4"
      style={{ backgroundColor: '#141720', border: '1px solid #2a2f42' }}
    >
      <div
        className="text-xs font-mono tracking-widest mb-3"
        style={{ color: '#4d5470' }}
      >
        {title}
      </div>
      {children}
    </div>
  )
}

// ── Main component ───────────────────────────────────────────────────────────

export function VerdictPanel({ result, clip: _clip, isScanning }: VerdictPanelProps) {
  return (
    <div className="flex flex-col gap-4">
      {/* ── VERDICT GAUGE ──────────────────────────────────────────────── */}
      <Section title="VERDICT">
        <AnimatePresence mode="wait">
          {isScanning ? (
            <motion.div
              key="scanning"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 10,
                padding: '8px 0',
              }}
            >
              <SkeletonBlock height={110} style={{ width: 200, borderRadius: 100 }} />
              <SkeletonBlock height={28} style={{ width: 100, borderRadius: 20 }} />
            </motion.div>
          ) : result ? (
            <motion.div
              key="result"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
            >
              {result.audio ? (
                /* ── Side-by-side: visual + audio ── */
                <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div
                      style={{
                        textAlign: 'center',
                        fontSize: 9,
                        fontFamily: 'monospace',
                        letterSpacing: '0.18em',
                        color: '#4d5470',
                        marginBottom: 4,
                      }}
                    >
                      VISUAL
                    </div>
                    <VerdictGauge
                      confidence={result.confidence}
                      verdict={result.verdict}
                      isScanning={false}
                    />
                  </div>
                  <div style={{ width: 1, backgroundColor: '#1e2233', alignSelf: 'stretch' }} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div
                      style={{
                        textAlign: 'center',
                        fontSize: 9,
                        fontFamily: 'monospace',
                        letterSpacing: '0.18em',
                        color: '#4d5470',
                        marginBottom: 4,
                      }}
                    >
                      AUDIO
                    </div>
                    <VerdictGauge
                      confidence={result.audio.confidence}
                      verdict={result.audio.verdict}
                      isScanning={false}
                    />
                  </div>
                </div>
              ) : (
                /* ── Single gauge (no audio) ── */
                <VerdictGauge
                  confidence={result.confidence}
                  verdict={result.verdict}
                  isScanning={false}
                />
              )}
            </motion.div>
          ) : (
            <motion.div
              key="idle"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 12,
                padding: '20px 0',
              }}
            >
              <svg width="160" height="90" viewBox="0 0 200 110">
                <path
                  d="M 28 92 A 72 72 0 0 1 172 92"
                  fill="none"
                  stroke="#1b1f2e"
                  strokeWidth="12"
                  strokeLinecap="round"
                />
                <text
                  x="100"
                  y="86"
                  textAnchor="middle"
                  fontSize="28"
                  fontWeight="700"
                  fill="#2a2f42"
                  fontFamily="monospace"
                >
                  —%
                </text>
              </svg>
              <span
                style={{ fontSize: 11, fontFamily: 'monospace', color: '#4d5470' }}
              >
                Run analysis to see verdict
              </span>
            </motion.div>
          )}
        </AnimatePresence>
      </Section>

      {/* TOP ANOMALY REGIONS removed — not supported by current inference pipeline */}

      {/* ── RESULT METADATA ────────────────────────────────────────────── */}
      {result && (
        <motion.div
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.4 }}
          className="rounded-lg px-4 py-3"
          style={{ backgroundColor: '#141720', border: '1px solid #2a2f42' }}
        >
          <div className="flex justify-between items-center">
            <span
              style={{ fontSize: 10, fontFamily: 'monospace', color: '#4d5470', letterSpacing: '0.1em' }}
            >
              FRAMES ANALYZED
            </span>
            <span
              style={{ fontSize: 10, fontFamily: 'monospace', color: '#8b92a8' }}
            >
              {result.heatmapFrames.length}
            </span>
          </div>
          <div className="flex justify-between items-center mt-1.5">
            <span
              style={{ fontSize: 10, fontFamily: 'monospace', color: '#4d5470', letterSpacing: '0.1em' }}
            >
              CLIP ID
            </span>
            <span
              style={{ fontSize: 10, fontFamily: 'monospace', color: '#8b92a8' }}
            >
              {result.clipId}
            </span>
          </div>
        </motion.div>
      )}
    </div>
  )
}
