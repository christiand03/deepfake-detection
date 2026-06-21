/**
 * RobustnessPanel — Phase 3 UI: Social-Media Degradation Lab.
 *
 * Simulates real-world video quality degradation (H.264 re-encoding, frame-rate
 * drops, Gaussian noise) and shows how the detector's confidence degrades.
 * Uses the mock factory for offline development; will call the FastAPI
 * `/robustness` endpoint in production.
 */

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { runRobustnessTest } from '../../api/client'
import { useErrorToast } from '../../context/ErrorToastContext'
import type { AnalysisResult, Phase3Result } from '../../types/analysis'
import { AttentionShiftTable } from '../shared/AttentionShiftTable'
import { AudioFrequencyShift } from '../shared/AudioFrequencyShift'

interface RobustnessPanelProps {
  result: AnalysisResult | null
}

// ── Slider component ─────────────────────────────────────────────────────────

interface SliderRowProps {
  label: string
  sublabel: string
  value: number
  min: number
  max: number
  step: number
  leftLabel: string
  rightLabel: string
  onChange: (v: number) => void
  disabled?: boolean
}

function SliderRow({
  label,
  sublabel,
  value,
  min,
  max,
  step,
  leftLabel,
  rightLabel,
  onChange,
  disabled,
}: SliderRowProps) {
  return (
    <div style={{ marginBottom: 18 }}>
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
          {label}
        </span>
        <span
          style={{
            fontSize: 10,
            fontFamily: 'monospace',
            color: '#00e5ff',
            fontWeight: 700,
          }}
        >
          {value}
        </span>
      </div>
      <div style={{ fontSize: 9, fontFamily: 'monospace', color: '#4d5470', marginBottom: 5 }}>
        {sublabel}
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={e => onChange(Number(e.target.value))}
        style={{ width: '100%', accentColor: '#00e5ff', cursor: disabled ? 'not-allowed' : 'pointer' }}
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
        <span>{leftLabel}</span>
        <span>{rightLabel}</span>
      </div>
    </div>
  )
}

// ── Confidence delta display ──────────────────────────────────────────────────

function ConfidenceDelta({
  label,
  confidence,
  verdict,
  isBefore,
  flipped,
}: {
  label: string
  confidence: number
  verdict: 'FAKE' | 'REAL'
  isBefore: boolean
  flipped?: boolean
}) {
  const color = verdict === 'FAKE' ? '#ef4444' : '#3b82f6'
  const glow = verdict === 'FAKE' ? 'rgba(239,68,68,0.25)' : 'rgba(59,130,246,0.25)'
  // On a verdict flip the degraded box is highlighted amber, identical to the
  // adversarial lab's "ATTACKED" box.
  const flippedHighlight = flipped && !isBefore
  return (
    <div
      style={{
        flex: 1,
        backgroundColor: '#0d0f14',
        borderRadius: 8,
        padding: '10px 14px',
        border: `1px solid ${isBefore ? '#2a2f42' : flippedHighlight ? '#f59e0b' : color}`,
        boxShadow: isBefore
          ? 'none'
          : flippedHighlight
            ? '0 0 12px rgba(245,158,11,0.2)'
            : `0 0 12px ${glow}`,
        textAlign: 'center',
      }}
    >
      <div
        style={{ fontSize: 9, fontFamily: 'monospace', color: '#4d5470', marginBottom: 6 }}
      >
        {label}
      </div>
      <div
        style={{
          fontSize: 28,
          fontFamily: 'monospace',
          fontWeight: 700,
          color,
          lineHeight: 1,
          marginBottom: 4,
        }}
      >
        {(confidence * 100).toFixed(1)}%
      </div>
      <div
        style={{
          display: 'inline-block',
          fontSize: 10,
          fontFamily: 'monospace',
          fontWeight: 700,
          letterSpacing: '0.1em',
          color,
          backgroundColor: `${color}22`,
          border: `1px solid ${color}44`,
          borderRadius: 4,
          padding: '2px 8px',
        }}
      >
        {verdict}
      </div>
    </div>
  )
}

// ── Heatmap frame comparison ──────────────────────────────────────────────────

function HeatmapComparison({
  original,
  degraded,
}: {
  original: string
  degraded: string
}) {
  return (
    <div>
      <div
        style={{
          fontSize: 9,
          fontFamily: 'monospace',
          color: '#4d5470',
          letterSpacing: '0.12em',
          marginBottom: 6,
        }}
      >
        HEATMAP COMPARISON — FRAME #8
      </div>
      <div style={{ display: 'flex', gap: 8 }}>
        <div style={{ flex: 1, textAlign: 'center' }}>
          <img
            src={original}
            alt="Original heatmap"
            style={{
              width: '100%',
              height: 70,
              objectFit: 'cover',
              borderRadius: 4,
              border: '1px solid #2a2f42',
            }}
          />
          <div style={{ fontSize: 9, fontFamily: 'monospace', color: '#4d5470', marginTop: 3 }}>
            ORIGINAL
          </div>
        </div>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            color: '#4d5470',
            fontSize: 14,
          }}
        >
          →
        </div>
        <div style={{ flex: 1, textAlign: 'center' }}>
          <img
            src={degraded}
            alt="Degraded heatmap"
            style={{
              width: '100%',
              height: 70,
              objectFit: 'cover',
              borderRadius: 4,
              border: '1px solid #4d5470',
            }}
          />
          <div style={{ fontSize: 9, fontFamily: 'monospace', color: '#4d5470', marginTop: 3 }}>
            DEGRADED
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Breaking point indicator ──────────────────────────────────────────────────

function BreakingPoint({
  original,
  degraded,
  params,
}: {
  original: number
  degraded: number
  params: { crf: number; fps: number; noiseSigma: number }
}) {
  const drop = original - degraded
  const dropPct = (drop / original) * 100
  const severity = dropPct > 50 ? 'critical' : dropPct > 25 ? 'moderate' : 'low'
  const severityColor =
    severity === 'critical' ? '#ef4444' : severity === 'moderate' ? '#f59e0b' : '#22c55e'

  return (
    <div
      style={{
        backgroundColor: '#0d0f14',
        borderRadius: 8,
        padding: '10px 14px',
        border: `1px solid ${severityColor}44`,
      }}
    >
      <div
        style={{
          fontSize: 9,
          fontFamily: 'monospace',
          color: '#4d5470',
          letterSpacing: '0.12em',
          marginBottom: 8,
        }}
      >
        ROBUSTNESS ANALYSIS
      </div>
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
        <div>
          <span
            style={{
              fontSize: 9,
              fontFamily: 'monospace',
              color: '#4d5470',
            }}
          >
            Confidence drop:{' '}
          </span>
          <span
            style={{
              fontSize: 11,
              fontFamily: 'monospace',
              color: severityColor,
              fontWeight: 700,
            }}
          >
            −{dropPct.toFixed(1)}%
          </span>
        </div>
        <div>
          <span style={{ fontSize: 9, fontFamily: 'monospace', color: '#4d5470' }}>
            Severity:{' '}
          </span>
          <span
            style={{
              fontSize: 11,
              fontFamily: 'monospace',
              color: severityColor,
              fontWeight: 700,
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
            }}
          >
            {severity}
          </span>
        </div>
      </div>
      <div
        style={{ marginTop: 8, fontSize: 9, fontFamily: 'monospace', color: '#8b92a8' }}
      >
        CRF {params.crf} · {params.fps} fps · σ={params.noiseSigma}
        {severity === 'critical' &&
          ` — Breaking point reached. Classifier unreliable under these conditions.`}
        {severity === 'moderate' &&
          ` — Significant degradation. Detection weakened but functional.`}
        {severity === 'low' && ` — Minimal impact. Classifier remains robust.`}
      </div>
    </div>
  )
}

// ── Main panel ───────────────────────────────────────────────────────────────

export function RobustnessPanel({ result }: RobustnessPanelProps) {
  const [crf, setCrf] = useState(28)
  const [fps, setFps] = useState(25)
  const [noiseSigma, setNoiseSigma] = useState(0)
  const [audioEnabled, setAudioEnabled] = useState(false)
  const [audioBitrate, setAudioBitrate] = useState(64)
  const [useMultimodal, setUseMultimodal] = useState(false)
  const [phase3, setPhase3] = useState<Phase3Result | null>(null)
  const [isRunning, setIsRunning] = useState(false)

  const { showError } = useErrorToast()

  // Clear results when base result changes
  useEffect(() => {
    setPhase3(null)
  }, [result?.clipId])

  function handleRun() {
    if (!result) return
    setIsRunning(true)
    setPhase3(null)
    runRobustnessTest(
      result.clipId,
      {
        crf,
        fps,
        noiseSigma,
        audioBitrate: audioEnabled ? audioBitrate : undefined,
        useMultimodal,
        fusionMode: 'cross_attention',
      },
      result,
    )
      .then(p3 => setPhase3(p3))
      .catch(err => showError(`Robustness test failed: ${err instanceof Error ? err.message : String(err)}`))
      .finally(() => setIsRunning(false))
  }

  const hasResult = result !== null
  const showResults = phase3 !== null

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
        {/* ── Left: Parameters ── */}
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
              marginBottom: 16,
            }}
          >
            DEGRADATION PARAMETERS
          </div>

          <SliderRow
            label="Compression Quality"
            sublabel="H.264 Constant Rate Factor"
            value={crf}
            min={18}
            max={51}
            step={1}
            leftLabel="Lossless (18)"
            rightLabel="Heavy (51)"
            onChange={setCrf}
            disabled={isRunning}
          />

          <SliderRow
            label="Frame Rate"
            sublabel="Frames per second"
            value={fps}
            min={5}
            max={30}
            step={5}
            leftLabel="5 fps"
            rightLabel="30 fps"
            onChange={setFps}
            disabled={isRunning}
          />

          <SliderRow
            label="Gaussian Noise σ"
            sublabel="Pixel-level sensor noise"
            value={noiseSigma}
            min={0}
            max={50}
            step={2}
            leftLabel="None"
            rightLabel="Heavy"
            onChange={setNoiseSigma}
            disabled={isRunning}
          />

          {/* Audio compression toggle — standalone Wav2Vec audio test; only
              meaningful in unimodal mode (the fusion model grades audio jointly),
              so it is disabled while MULTIMODAL is active. */}
          <div
            style={{ marginTop: 14, paddingTop: 12, borderTop: '1px solid #2a2f42' }}
          >
            <label
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                cursor: isRunning || useMultimodal ? 'not-allowed' : 'pointer',
                opacity: isRunning || useMultimodal ? 0.4 : 1,
              }}
            >
              <input
                type="checkbox"
                checked={audioEnabled}
                onChange={e => setAudioEnabled(e.target.checked)}
                disabled={isRunning || useMultimodal}
                style={{ accentColor: '#00e5ff', cursor: 'inherit' }}
              />
              <span
                style={{
                  fontSize: 10,
                  fontFamily: 'monospace',
                  color: audioEnabled ? '#8b92a8' : '#4d5470',
                  letterSpacing: '0.08em',
                }}
              >
                TEST AUDIO COMPRESSION
              </span>
            </label>
            {useMultimodal ? (
              <div style={{ marginTop: 4, fontSize: 9, fontFamily: 'monospace', color: '#4d5470' }}>
                Disabled — the multimodal model already grades audio jointly.
              </div>
            ) : (
              audioEnabled && (
                <SliderRow
                  label="Audio Bitrate"
                  sublabel="AAC target bitrate (kbps)"
                  value={audioBitrate}
                  min={8}
                  max={320}
                  step={8}
                  leftLabel="8 kbps"
                  rightLabel="320 kbps"
                  onChange={setAudioBitrate}
                  disabled={isRunning}
                />
              )
            )}
          </div>

          {/* Multimodal toggle — only when the clip has an audio track */}
          {result?.audio != null && (
            <div style={{ marginTop: 14, paddingTop: 12, borderTop: '1px solid #2a2f42' }}>
              <label
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  cursor: isRunning ? 'not-allowed' : 'pointer',
                  opacity: isRunning ? 0.5 : 1,
                }}
              >
                <input
                  type="checkbox"
                  checked={useMultimodal}
                  onChange={e => {
                    setUseMultimodal(e.target.checked)
                    // Mutually exclusive with the standalone Wav2Vec audio test.
                    if (e.target.checked) setAudioEnabled(false)
                  }}
                  disabled={isRunning}
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
              <div style={{ marginTop: 4, fontSize: 9, fontFamily: 'monospace', color: '#4d5470' }}>
                Re-score the degraded clip with the joint video+audio fusion model.
              </div>
            </div>
          )}

          <button
            onClick={handleRun}
            disabled={!hasResult || isRunning}
            style={{
              width: '100%',
              marginTop: 4,
              padding: '9px 0',
              borderRadius: 6,
              border: `1px solid ${!hasResult || isRunning ? '#2a2f42' : 'rgba(0,229,255,0.3)'}`,
              backgroundColor:
                !hasResult || isRunning ? '#1b1f2e' : 'rgba(0,229,255,0.12)',
              color: !hasResult || isRunning ? '#4d5470' : '#00e5ff',
              fontFamily: 'monospace',
              fontSize: 11,
              fontWeight: 700,
              letterSpacing: '0.1em',
              cursor: !hasResult || isRunning ? 'not-allowed' : 'pointer',
              transition: 'all 0.15s ease',
            }}
          >
            {isRunning ? '⏳ RUNNING…' : '▶ RUN ROBUSTNESS TEST'}
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
        <div style={{ minHeight: 220 }}>
          <AnimatePresence mode="wait">
            {!showResults && !isRunning && (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                style={{
                  height: 220,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 8,
                  border: '1px dashed #2a2f42',
                  borderRadius: 8,
                }}
              >
                <div style={{ fontSize: 22, opacity: 0.3 }}>📡</div>
                <div
                  style={{
                    fontSize: 10,
                    fontFamily: 'monospace',
                    color: '#4d5470',
                    textAlign: 'center',
                  }}
                >
                  {hasResult
                    ? 'Configure parameters and run the robustness test'
                    : 'Run video analysis to unlock robustness testing'}
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
                  height: 220,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 10,
                }}
              >
                <div className="shimmer" style={{ width: '100%', height: 60, borderRadius: 6 }} />
                <div className="shimmer" style={{ width: '100%', height: 60, borderRadius: 6 }} />
                <div className="shimmer" style={{ width: '100%', height: 60, borderRadius: 6 }} />
              </motion.div>
            )}

            {showResults && phase3 && result && (
              <motion.div
                key="results"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.35 }}
                style={{ display: 'flex', flexDirection: 'column', gap: 12 }}
              >
                {/* Confidence before/after */}
                <div style={{ display: 'flex', gap: 10, alignItems: 'stretch' }}>
                  <ConfidenceDelta
                    label="CLEAN"
                    confidence={phase3.baselineConfidence}
                    verdict={phase3.baselineVerdict}
                    isBefore
                  />
                  <div
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      gap: 2,
                    }}
                  >
                    <span style={{ fontSize: 18, color: '#4d5470' }}>→</span>
                    {phase3.baselineVerdict !== phase3.degradedVerdict && (
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
                  <ConfidenceDelta
                    label="DEGRADED"
                    confidence={phase3.degradedConfidence}
                    verdict={phase3.degradedVerdict}
                    isBefore={false}
                    flipped={phase3.baselineVerdict !== phase3.degradedVerdict}
                  />
                </div>

                {/* Breaking point */}
                <BreakingPoint
                  original={phase3.baselineConfidence}
                  degraded={phase3.degradedConfidence}
                  params={phase3.params}
                />

                {/* Heatmap comparison */}
                <HeatmapComparison
                  original={result.heatmapFrames[8] ?? result.heatmapFrames[0]}
                  degraded={
                    phase3.degradedHeatmapFrames[8] ?? phase3.degradedHeatmapFrames[0]
                  }
                />

                {/* Attention shift */}
                {phase3.attentionShift.length > 0 && (
                  <AttentionShiftTable shifts={phase3.attentionShift} />
                )}

                {/* Audio frequency-band shift */}
                {phase3.audioRobustness && (
                  <AudioFrequencyShift audio={phase3.audioRobustness} />
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  )
}
