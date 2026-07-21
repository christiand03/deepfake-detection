/**
 * FrequencyHeatmap — Layer 3 as a band × time grid (replaces the 3-bar chart).
 *
 * Rows = the 3 frequency bands (High on top, Low on bottom, spectrogram-style),
 * columns = the model's 0.64 s decision windows — the SAME x-axis as the L1
 * decision timeline, so the two stack: L1 says *when it leans fake*, L3 says
 * *when AND which band*.
 *
 * Confidence view: fakeness-gated band-ablation grid — a cell lights up red where
 * removing that band would make that window look real (the band carries the fake);
 * real windows are dark (0 = nothing to attribute). Relevance view: the honest,
 * faint per-window gradient relevance (frequency does not localise gradient
 * relevance the way ablation does — kept for toggle consistency).
 */

import { useEffect, useRef } from 'react'

import { bivariateRgba } from '../../lib/seismicColormap'
import type { AudioAnalysis, AudioView } from '../../types/analysis'

const BANDS = [
  { key: 'high' as const, label: 'High', sub: '4–8k' },
  { key: 'mid' as const, label: 'Mid', sub: '0.5–4k' },
  { key: 'low' as const, label: 'Low', sub: '0–500' },
]
const ROW_H = 22

// Confidence cell: signed ablation fraction in [-1, 1]. Alpha from |v| (0 → fully
// transparent = real window / no fake to show), hue from sign with a gentle gain so
// the fake-carrying band reads as a clear red block.
function confCell(v: number): string {
  return bivariateRgba(Math.abs(v), v, {
    alphaGamma: 0.6,
    dirGamma: 1.0,
    dirGain: 1.4,
    dirCap: 0.85,
    maxAlpha: 0.92,
  })
}

// Relevance cell: honest faint bivariate gradient (same encoding language as L1).
function relCell(mag: number, dir: number): string {
  return bivariateRgba(mag, dir, { alphaGamma: 0.6, dirGamma: 1.6, dirGain: 4, dirCap: 0.85 })
}

interface FrequencyHeatmapProps {
  audio: AudioAnalysis
  /** Confidence (gated ablation grid) vs Relevance (faint gradient grid). */
  view: AudioView
  videoRef: React.RefObject<HTMLVideoElement | null>
  /** Clip duration (s) for the playhead + time axis. */
  duration: number
}

export function FrequencyHeatmap({ audio, view, videoRef, duration }: FrequencyHeatmapProps) {
  const isConf = view === 'confidence'
  const gc = audio.frequencyGridConfidence
  const gr = audio.frequencyGridRelevance
  const n = isConf ? (gc?.low.length ?? 0) : (gr?.low.magnitude.length ?? 0)
  // Confidence is a fakeness-gated ablation grid: real windows are 0 (nothing to
  // attribute), so an all-real clip yields an entirely transparent grid. Detect
  // that case to show an explicit empty-state instead of a silent dark block.
  const allRealConf =
    isConf &&
    gc != null &&
    (['low', 'mid', 'high'] as const).every(k => gc[k].every(v => v === 0))
  const playRef = useRef<HTMLDivElement>(null)

  // rAF playhead over the cell area — aligned to the same audio timeline as L1.
  useEffect(() => {
    let raf = 0
    const tick = () => {
      const el = playRef.current
      const vid = videoRef.current
      if (el && vid && duration > 0) {
        el.style.left = `${Math.min(100, Math.max(0, (vid.currentTime / duration) * 100))}%`
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [videoRef, duration])

  const cellColor = (bandKey: 'low' | 'mid' | 'high', w: number): string => {
    if (isConf) return confCell(gc![bandKey][w])
    const s = gr![bandKey]
    return relCell(s.magnitude[w], s.direction[w])
  }

  return (
    <div>
      <div
        style={{
          fontSize: 10,
          fontFamily: 'monospace',
          color: '#4d5470',
          letterSpacing: '0.12em',
          marginBottom: 8,
        }}
      >
        LAYER 3 — FREQUENCY × TIME {isConf ? 'CONFIDENCE' : 'RELEVANCE'}
      </div>

      {n === 0 ? (
        <div style={{ fontSize: 10, fontFamily: 'monospace', color: '#4d5470', padding: '6px 0' }}>
          Band × time grid unavailable
        </div>
      ) : (
        <>
          <div style={{ display: 'flex' }}>
            {/* Band labels */}
            <div style={{ display: 'flex', flexDirection: 'column', width: 44, flexShrink: 0 }}>
              {BANDS.map(b => (
                <div
                  key={b.key}
                  style={{
                    height: ROW_H,
                    marginBottom: 2,
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'center',
                    fontFamily: 'monospace',
                  }}
                >
                  <span style={{ fontSize: 10, fontWeight: 600, color: '#8b92a8', lineHeight: 1 }}>{b.label}</span>
                  <span style={{ fontSize: 7, color: '#4d5470', lineHeight: 1.2 }}>{b.sub}</span>
                </div>
              ))}
            </div>

            {/* Cell grid + playhead */}
            <div style={{ flex: 1, position: 'relative' }}>
              {BANDS.map(b => (
                <div key={b.key} style={{ height: ROW_H, marginBottom: 2, display: 'flex', borderRadius: 2, overflow: 'hidden' }}>
                  {Array.from({ length: n }, (_, w) => (
                    <div
                      key={w}
                      title={`${b.label} · win ${w + 1}`}
                      style={{
                        flex: 1,
                        backgroundColor: cellColor(b.key, w),
                        borderRight: w < n - 1 ? '1px solid rgba(13,15,20,0.6)' : undefined,
                      }}
                    />
                  ))}
                </div>
              ))}
              <div
                ref={playRef}
                style={{
                  position: 'absolute',
                  top: 0,
                  bottom: 2,
                  left: '0%',
                  width: 1.5,
                  backgroundColor: '#00e5ff',
                  boxShadow: '0 0 6px #00e5ff',
                  pointerEvents: 'none',
                }}
              />
              {allRealConf && (
                <div
                  style={{
                    position: 'absolute',
                    inset: 0,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    textAlign: 'center',
                    padding: '0 8px',
                    fontFamily: 'monospace',
                    fontSize: 9,
                    lineHeight: 1.5,
                    letterSpacing: '0.06em',
                    color: '#5e91ee',
                    pointerEvents: 'none',
                  }}
                >
                  All windows classified real — no fake evidence for any band to
                  carry.
                </div>
              )}
            </div>
          </div>

          {/* Time axis (aligned under the cell grid) */}
          <div style={{ display: 'flex' }}>
            <div style={{ width: 44, flexShrink: 0 }} />
            <div
              style={{
                flex: 1,
                display: 'flex',
                justifyContent: 'space-between',
                marginTop: 3,
                fontSize: 9,
                fontFamily: 'monospace',
                color: '#4d5470',
              }}
            >
              <span>0s</span>
              <span>{(duration / 2).toFixed(1)}s</span>
              <span>{duration.toFixed(1)}s</span>
            </div>
          </div>

          {/* Legend */}
          <div
            style={{
              display: 'flex',
              gap: 12,
              marginTop: 6,
              fontSize: 8,
              fontFamily: 'monospace',
            }}
          >
            <span style={{ color: '#ff7070' }}>■ {isConf ? 'band carries fake' : 'fake-leaning'}</span>
            <span style={{ color: '#5e91ee' }}>■ {isConf ? 'band pulls real' : 'real-leaning'}</span>
            <span style={{ color: '#4d5470' }}>■ {isConf ? 'real window (no fake)' : 'no signal'}</span>
          </div>
        </>
      )}
    </div>
  )
}
