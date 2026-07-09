/**
 * AdversarialPanel — Phase 4 UI: Adversarial Attack Lab.
 *
 * Visualises the impact of white-box attacks (FGSM / PGD) on the detector.
 * Shows clean vs. perturbed verdict, a difference map (magnified perturbation),
 * and an attention-shift table exposing xAI manipulation.
 */

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { runAdversarialAttack } from '../../api/client'
import { useErrorToast } from '../../context/ErrorToastContext'
import type { AnalysisResult, Phase4Result } from '../../types/analysis'
import { AttentionShiftTable } from '../shared/AttentionShiftTable'
import { RegionToggle } from '../shared/RegionToggle'
import { CropComparisonPlayer } from './CropComparisonPlayer'

interface AdversarialPanelProps {
  result: AnalysisResult | null
}

// ── Verdict flip badge ────────────────────────────────────────────────────────

function VerdictCompare({
  original,
  originalConf,
  perturbed,
  perturbedConf,
}: {
  original: 'FAKE' | 'REAL'
  originalConf: number
  perturbed: 'FAKE' | 'REAL'
  perturbedConf: number
}) {
  const flipped = original !== perturbed
  const origColor = original === 'FAKE' ? '#ef4444' : '#3b82f6'
  const pertColor = perturbed === 'FAKE' ? '#ef4444' : '#3b82f6'

  return (
    <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
      <div
        style={{
          flex: 1,
          backgroundColor: '#0d0f14',
          borderRadius: 8,
          padding: '10px 12px',
          border: '1px solid #2a2f42',
          textAlign: 'center',
        }}
      >
        <div style={{ fontSize: 8, fontFamily: 'monospace', color: '#4d5470', marginBottom: 4 }}>
          CLEAN
        </div>
        <div
          style={{ fontSize: 22, fontFamily: 'monospace', fontWeight: 700, color: origColor }}
        >
          {(originalConf * 100).toFixed(0)}%
        </div>
        <div
          style={{
            fontSize: 9,
            fontFamily: 'monospace',
            color: origColor,
            marginTop: 2,
          }}
        >
          {original}
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
        <span style={{ fontSize: 16, color: '#4d5470' }}>→</span>
        {flipped && (
          <span
            style={{
              fontSize: 8,
              fontFamily: 'monospace',
              color: '#f59e0b',
              fontWeight: 700,
              letterSpacing: '0.08em',
              backgroundColor: 'rgba(245,158,11,0.12)',
              border: '1px solid rgba(245,158,11,0.4)',
              borderRadius: 3,
              padding: '1px 4px',
              whiteSpace: 'nowrap',
            }}
          >
            FLIPPED!
          </span>
        )}
      </div>

      <div
        style={{
          flex: 1,
          backgroundColor: '#0d0f14',
          borderRadius: 8,
          padding: '10px 12px',
          border: `1px solid ${flipped ? '#f59e0b' : pertColor}`,
          boxShadow: flipped ? '0 0 12px rgba(245,158,11,0.2)' : 'none',
          textAlign: 'center',
        }}
      >
        <div style={{ fontSize: 8, fontFamily: 'monospace', color: '#4d5470', marginBottom: 4 }}>
          ATTACKED
        </div>
        <div
          style={{ fontSize: 22, fontFamily: 'monospace', fontWeight: 700, color: pertColor }}
        >
          {(perturbedConf * 100).toFixed(0)}%
        </div>
        <div
          style={{
            fontSize: 9,
            fontFamily: 'monospace',
            color: pertColor,
            marginTop: 2,
          }}
        >
          {perturbed}
        </div>
      </div>
    </div>
  )
}

// ── Main panel ───────────────────────────────────────────────────────────────

export function AdversarialPanel({ result }: AdversarialPanelProps) {
  const [method, setMethod] = useState<'FGSM' | 'PGD'>('FGSM')
  const [epsilon, setEpsilon] = useState(0.03)
  const [steps, setSteps] = useState(20)
  const [useMultimodal, setUseMultimodal] = useState(false)
  const [attackModalities, setAttackModalities] = useState<'video' | 'audio' | 'both'>('both')
  const [audioEpsilon, setAudioEpsilon] = useState(0.03)
  const [phase4, setPhase4] = useState<Phase4Result | null>(null)
  const [showRegions, setShowRegions] = useState(false)
  const [isRunning, setIsRunning] = useState(false)

  const { showError } = useErrorToast()

  useEffect(() => {
    setPhase4(null)
  }, [result?.clipId])

  function handleAttack() {
    if (!result) return
    setIsRunning(true)
    setPhase4(null)
    runAdversarialAttack(
      result.clipId,
      method,
      epsilon,
      steps,
      result,
      useMultimodal,
      attackModalities,
      audioEpsilon,
    )
      .then(p4 => setPhase4(p4))
      .catch(err => showError(`Adversarial attack failed: ${err instanceof Error ? err.message : String(err)}`))
      .finally(() => setIsRunning(false))
  }

  const hasResult = result !== null
  const showResults = phase4 !== null

  // Both sides come from the SAME model as the attack (I3): the clean baseline
  // (phase4.cleanVerdict/cleanConfidence) and the attacked verdict
  // (phase4.perturbedVerdict, reported directly — it cannot be re-derived from
  // the direction-less perturbedConfidence). Independent of the main panel toggle.
  const perturbedVerdict = phase4?.perturbedVerdict ?? null

  return (
    <div style={{ padding: '16px 20px' }}>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '260px 1fr',
          gap: 20,
          alignItems: 'start',
        }}
      >
        {/* ── Left: Attack config ── */}
        <div
          style={{
            backgroundColor: '#141720',
            borderRadius: 8,
            padding: '14px 16px',
            border: '1px solid #2a2f42',
          }}
        >
          <div
            style={{
              fontSize: 9,
              fontFamily: 'monospace',
              color: '#4d5470',
              letterSpacing: '0.15em',
              marginBottom: 14,
            }}
          >
            ATTACK CONFIGURATION
          </div>

          {/* Method toggle */}
          <div style={{ marginBottom: 16 }}>
            <div
              style={{
                fontSize: 11,
                fontFamily: 'monospace',
                color: '#e8eaf0',
                fontWeight: 600,
                marginBottom: 6,
              }}
            >
              Attack Method
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              {(['FGSM', 'PGD'] as const).map(m => (
                <button
                  key={m}
                  onClick={() => setMethod(m)}
                  disabled={isRunning}
                  style={{
                    flex: 1,
                    padding: '7px 0',
                    borderRadius: 5,
                    border: `1px solid ${method === m ? 'rgba(0,229,255,0.5)' : '#2a2f42'}`,
                    backgroundColor:
                      method === m ? 'rgba(0,229,255,0.12)' : '#0d0f14',
                    color: method === m ? '#00e5ff' : '#4d5470',
                    fontFamily: 'monospace',
                    fontSize: 11,
                    fontWeight: method === m ? 700 : 400,
                    cursor: isRunning ? 'not-allowed' : 'pointer',
                    letterSpacing: '0.06em',
                    transition: 'all 0.15s ease',
                  }}
                >
                  {method === m ? '●' : '○'} {m}
                </button>
              ))}
            </div>
            <div
              style={{
                marginTop: 5,
                fontSize: 9,
                fontFamily: 'monospace',
                color: '#4d5470',
              }}
            >
              {method === 'FGSM'
                ? 'Single-step gradient sign method'
                : 'Multi-step projected gradient descent'}
            </div>
          </div>

          {/* Epsilon */}
          <div style={{ marginBottom: 16 }}>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                marginBottom: 4,
                alignItems: 'baseline',
              }}
            >
              <span style={{ fontSize: 11, fontFamily: 'monospace', color: '#e8eaf0', fontWeight: 600 }}>
                ε (Epsilon)
              </span>
              <span style={{ fontSize: 10, fontFamily: 'monospace', color: '#00e5ff', fontWeight: 700 }}>
                {epsilon.toFixed(3)}
              </span>
            </div>
            <div style={{ fontSize: 9, fontFamily: 'monospace', color: '#4d5470', marginBottom: 5 }}>
              Max perturbation magnitude (L∞ norm)
            </div>
            <input
              type="range"
              min={0.001}
              max={0.1}
              step={0.001}
              value={epsilon}
              disabled={isRunning}
              onChange={e => setEpsilon(Number(e.target.value))}
              style={{ width: '100%', accentColor: '#00e5ff', cursor: isRunning ? 'not-allowed' : 'pointer' }}
            />
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                marginTop: 2,
                fontSize: 8,
                fontFamily: 'monospace',
                color: '#2a2f42',
              }}
            >
              <span>Imperceptible</span>
              <span>Visible</span>
            </div>
          </div>

          {/* Steps (PGD only) */}
          {method === 'PGD' && (
            <div style={{ marginBottom: 16 }}>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  marginBottom: 4,
                  alignItems: 'baseline',
                }}
              >
                <span
                  style={{
                    fontSize: 11,
                    fontFamily: 'monospace',
                    color: '#e8eaf0',
                    fontWeight: 600,
                  }}
                >
                  PGD Steps
                </span>
                <span
                  style={{
                    fontSize: 10,
                    fontFamily: 'monospace',
                    color: '#00e5ff',
                    fontWeight: 700,
                  }}
                >
                  {steps}
                </span>
              </div>
              <input
                type="range"
                min={5}
                max={40}
                step={5}
                value={steps}
                disabled={isRunning}
                onChange={e => setSteps(Number(e.target.value))}
                style={{
                  width: '100%',
                  accentColor: '#00e5ff',
                  cursor: isRunning ? 'not-allowed' : 'pointer',
                }}
              />
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  marginTop: 2,
                  fontSize: 8,
                  fontFamily: 'monospace',
                  color: '#2a2f42',
                }}
              >
                <span>Fast (5)</span>
                <span>Strong (40)</span>
              </div>
            </div>
          )}

          {/* Multimodal toggle — only when clip has audio */}
          {result?.audio != null && (
            <div
              style={{
                marginBottom: 16,
                borderTop: '1px solid #2a2f42',
                paddingTop: 14,
              }}
            >
              <label
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  cursor: isRunning ? 'not-allowed' : 'pointer',
                }}
              >
                <input
                  type="checkbox"
                  checked={useMultimodal}
                  disabled={isRunning}
                  onChange={e => setUseMultimodal(e.target.checked)}
                  style={{ accentColor: '#a855f7', cursor: 'inherit' }}
                />
                <span
                  style={{
                    fontSize: 11,
                    fontFamily: 'monospace',
                    color: useMultimodal ? '#a855f7' : '#e8eaf0',
                    fontWeight: 600,
                    letterSpacing: '0.04em',
                  }}
                >
                  MULTIMODAL MODEL
                </span>
              </label>
              <div
                style={{
                  marginTop: 4,
                  fontSize: 9,
                  fontFamily: 'monospace',
                  color: '#4d5470',
                }}
              >
                Attack via joint video+audio model
              </div>

              {useMultimodal && (
                <div style={{ marginTop: 12 }}>
                  {/* Modality selector */}
                  <div
                    style={{
                      fontSize: 11,
                      fontFamily: 'monospace',
                      color: '#e8eaf0',
                      fontWeight: 600,
                      marginBottom: 6,
                    }}
                  >
                    Target Modalities
                  </div>
                  <div style={{ display: 'flex', gap: 4, marginBottom: 12 }}>
                    {(['video', 'audio', 'both'] as const).map(m => (
                      <button
                        key={m}
                        onClick={() => setAttackModalities(m)}
                        disabled={isRunning}
                        style={{
                          flex: 1,
                          padding: '5px 0',
                          borderRadius: 5,
                          border: `1px solid ${
                            attackModalities === m
                              ? 'rgba(168,85,247,0.5)'
                              : '#2a2f42'
                          }`,
                          backgroundColor:
                            attackModalities === m
                              ? 'rgba(168,85,247,0.12)'
                              : '#0d0f14',
                          color:
                            attackModalities === m ? '#a855f7' : '#4d5470',
                          fontFamily: 'monospace',
                          fontSize: 9,
                          fontWeight: attackModalities === m ? 700 : 400,
                          cursor: isRunning ? 'not-allowed' : 'pointer',
                          letterSpacing: '0.04em',
                          textTransform: 'uppercase',
                          transition: 'all 0.15s ease',
                        }}
                      >
                        {m}
                      </button>
                    ))}
                  </div>

                  {/* Audio epsilon */}
                  <div>
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        marginBottom: 4,
                        alignItems: 'baseline',
                      }}
                    >
                      <span
                        style={{
                          fontSize: 11,
                          fontFamily: 'monospace',
                          color: '#e8eaf0',
                          fontWeight: 600,
                        }}
                      >
                        Audio ε
                      </span>
                      <span
                        style={{
                          fontSize: 10,
                          fontFamily: 'monospace',
                          color: '#a855f7',
                          fontWeight: 700,
                        }}
                      >
                        {audioEpsilon.toFixed(3)}
                      </span>
                    </div>
                    <div
                      style={{
                        fontSize: 9,
                        fontFamily: 'monospace',
                        color: '#4d5470',
                        marginBottom: 5,
                      }}
                    >
                      Audio L∞ perturbation budget
                    </div>
                    <input
                      type="range"
                      min={0.01}
                      max={0.5}
                      step={0.01}
                      value={audioEpsilon}
                      disabled={isRunning}
                      onChange={e => setAudioEpsilon(Number(e.target.value))}
                      style={{
                        width: '100%',
                        accentColor: '#a855f7',
                        cursor: isRunning ? 'not-allowed' : 'pointer',
                      }}
                    />
                  </div>
                </div>
              )}
            </div>
          )}

          <button
            onClick={handleAttack}
            disabled={!hasResult || isRunning}
            style={{
              width: '100%',
              marginTop: 4,
              padding: '9px 0',
              borderRadius: 6,
              border: `1px solid ${!hasResult || isRunning ? '#2a2f42' : 'rgba(239,68,68,0.4)'}`,
              backgroundColor:
                !hasResult || isRunning ? '#1b1f2e' : 'rgba(239,68,68,0.1)',
              color: !hasResult || isRunning ? '#4d5470' : '#ef4444',
              fontFamily: 'monospace',
              fontSize: 11,
              fontWeight: 700,
              letterSpacing: '0.1em',
              cursor: !hasResult || isRunning ? 'not-allowed' : 'pointer',
              transition: 'all 0.15s ease',
            }}
          >
            {isRunning ? '⏳ ATTACKING…' : `⚡ LAUNCH ${method}`}
          </button>

          {!hasResult && (
            <div
              style={{
                marginTop: 8,
                fontSize: 9,
                fontFamily: 'monospace',
                color: '#4d5470',
                textAlign: 'center',
              }}
            >
              Run video analysis first
            </div>
          )}
        </div>

        {/* ── Right: Results ── */}
        <div style={{ minHeight: 260 }}>
          <AnimatePresence mode="wait">
            {!showResults && !isRunning && (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                style={{
                  height: 260,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 8,
                  border: '1px dashed #2a2f42',
                  borderRadius: 8,
                }}
              >
                <div style={{ fontSize: 22, opacity: 0.3 }}>⚡</div>
                <div
                  style={{
                    fontSize: 10,
                    fontFamily: 'monospace',
                    color: '#4d5470',
                    textAlign: 'center',
                  }}
                >
                  {hasResult
                    ? 'Configure the attack and launch to see xAI impact'
                    : 'Run video analysis to unlock adversarial testing'}
                </div>
              </motion.div>
            )}

            {isRunning && (
              <motion.div
                key="loading"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                style={{
                  height: 260,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 10,
                  justifyContent: 'center',
                }}
              >
                <div className="shimmer" style={{ width: '100%', height: 70, borderRadius: 6 }} />
                <div className="shimmer" style={{ width: '100%', height: 80, borderRadius: 6 }} />
                <div className="shimmer" style={{ width: '100%', height: 70, borderRadius: 6 }} />
              </motion.div>
            )}

            {showResults && phase4 && result && perturbedVerdict && (
              <motion.div
                key="results"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.35 }}
                style={{ display: 'flex', flexDirection: 'column', gap: 12 }}
              >
                {/* Verdict comparison */}
                <VerdictCompare
                  original={phase4.cleanVerdict}
                  originalConf={phase4.cleanConfidence}
                  perturbed={perturbedVerdict}
                  perturbedConf={phase4.perturbedConfidence}
                />

                {/* Whole-clip crop player: clean → attacked (I2) */}
                <CropComparisonPlayer
                  title="HEATMAP — WHOLE CLIP (CLEAN → ATTACKED)"
                  left={{
                    label: 'CLEAN',
                    videoUrl: phase4.cleanVideoUrl,
                    heatmapFrames: phase4.cleanHeatmapFrames,
                    accent: '#2a2f42',
                    regionFrames: phase4.regionMaskFrames,
                  }}
                  right={{
                    label: 'ATTACKED',
                    videoUrl: phase4.attackedVideoUrl,
                    heatmapFrames: phase4.perturbedFrames,
                    accent: '#ef4444',
                  }}
                  showRegions={showRegions}
                />

                {/* Region overlay toggle (below the opacity slider, above the shift) */}
                <RegionToggle
                  checked={showRegions}
                  onChange={setShowRegions}
                  visible={!!phase4.regionMaskFrames && phase4.regionMaskFrames.length > 0}
                />

                {/* Attention shifts */}
                <AttentionShiftTable shifts={phase4.attentionShift} />

                {/* Audio frequency-band shift (multimodal attacks only) */}
                {phase4.audioAttentionShift && phase4.audioAttentionShift.length > 0 && (
                  <>
                    <div
                      style={{
                        fontSize: 9,
                        fontFamily: 'monospace',
                        color: '#a855f7',
                        letterSpacing: '0.15em',
                        marginTop: 4,
                      }}
                    >
                      AUDIO FREQUENCY SHIFT
                      {phase4.attackModalities && (
                        <span
                          style={{
                            marginLeft: 8,
                            color: '#4d5470',
                            textTransform: 'uppercase',
                          }}
                        >
                          [{phase4.attackModalities}]
                        </span>
                      )}
                    </div>
                    <AttentionShiftTable shifts={phase4.audioAttentionShift} />
                  </>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  )
}
