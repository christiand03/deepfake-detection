/**
 * WaveformRelevanceLayer — Layer 1 of the audio xAI stack.
 *
 * Renders two tracks on an HTML5 Canvas:
 *   Top 40%  — seismic-coloured relevance band (per-chunk average relevance)
 *   Bottom 60% — grey waveform envelope (RMS amplitude per chunk)
 *
 * A vertical cyan playhead tracks the current video time.
 * The waveform + relevance are drawn once; the playhead is redrawn each frame.
 */

import { useEffect, useRef } from 'react'
import { relevanceToRgb } from '../../lib/seismicColormap'
import type { AudioAnalysis, AudioView } from '../../types/analysis'

interface WaveformRelevanceLayerProps {
  audio: AudioAnalysis
  videoRef: React.RefObject<HTMLVideoElement | null>
  /** Clip duration in seconds, used for the playhead position */
  duration: number
  /** Relevance (signed AttnLRP) vs. Confidence (per-window fake-prob) view (B4). */
  view: AudioView
}

const CANVAS_W = 900
const CANVAS_H = 90
const RELEVANCE_H = 30 // top strip height
const WAVEFORM_CY = RELEVANCE_H + (CANVAS_H - RELEVANCE_H) / 2 // vertical centre of waveform

/** Down-sample a Float32-like array to `nBuckets` values using a reducer */
function downsample(
  arr: number[],
  nBuckets: number,
  reduce: (chunk: number[]) => number,
): number[] {
  const chunkSize = Math.ceil(arr.length / nBuckets)
  return Array.from({ length: nBuckets }, (_, i) => {
    const start = i * chunkSize
    const chunk = arr.slice(start, start + chunkSize)
    return chunk.length === 0 ? 0 : reduce(chunk)
  })
}

function rms(chunk: number[]): number {
  return Math.sqrt(chunk.reduce((s, v) => s + v * v, 0) / chunk.length)
}

function mean(chunk: number[]): number {
  return chunk.reduce((s, v) => s + v, 0) / chunk.length
}

// Relevance L1 = the model's DECISION TIMELINE. The strip is bucketed at the
// model's native 0.64s window (one block per decision), so it lines up 1:1 with
// the per-window Confidence view and sits under the waveform as acoustic context.
// Bivariate per block: ALPHA from magnitude (engagement -> WHERE the model looked,
// opaque where it worked hard), HUE from the gamma-emphasised direction (WHICH WAY
// it leaned -> vivid red/blue for a coherent window, neutral white for an
// engaged-but-undecided one). So a fake decision = opaque + red, an undecided
// high-engagement window = opaque + white, a quiet window = faint. Tune like L2:
// higher L1_GAMMA / L1_COLOR_GAIN sharpen the hue; L1_ALPHA_GAMMA shapes opacity.
const L1_WINDOW_S = 0.64
const L1_GAMMA = 1.5 // direction suppression for the hue (de-noise)
const L1_COLOR_GAIN = 4.0 // lift a coherent lean into vivid red/blue
const L1_COLOR_CAP = 0.85 // stay below seismic's dark endpoint
const L1_ALPHA_GAMMA = 0.6 // magnitude -> alpha (engagement / intensity)
function l1BlockRgba(magnitude: number, direction: number, maxAlpha: number): string {
  const c =
    Math.sign(direction) *
    Math.min(L1_COLOR_CAP, Math.abs(direction) ** L1_GAMMA * L1_COLOR_GAIN)
  const [r, g, b] = relevanceToRgb(c) // hue + saturation from the gamma'd lean
  // Alpha = magnitude (engagement), independent of lean direction.
  const a = Math.min(maxAlpha, Math.max(0, magnitude) ** L1_ALPHA_GAMMA * maxAlpha)
  return `rgba(${r},${g},${b},${a.toFixed(3)})`
}

/**
 * Per-pixel fill colours for the relevance strip + waveform overlay.
 *
 * Relevance view: 0.64s decision blocks — each pixel takes its 640ms window's mean
 * direction through `l1BlockRgba` (colour + opacity from the gamma'd lean). The
 * fake decision becomes one clear block; real audio stays dark. Confidence view
 * (or older caches without the bivariate arrays): the legacy signed path — hue
 * from the signed value, alpha from |value|.
 */
function computeFills(
  audio: AudioAnalysis,
  view: AudioView,
  w: number,
): { strip: string[]; overlay: string[] } {
  const bivariate =
    view !== 'confidence' &&
    audio.waveformDirection.length > 0 &&
    audio.waveformMagnitude.length > 0

  if (bivariate) {
    const dir = audio.waveformDirection
    const mag = audio.waveformMagnitude
    const total = dir.length
    // Samples per 0.64s model-decision window (10 240 at 16 kHz).
    const spw = Math.max(1, Math.round(L1_WINDOW_S * audio.sampleRate))
    const nWin = Math.max(1, Math.ceil(total / spw))
    const winMean = (arr: number[], k: number) => {
      const seg = arr.slice(k * spw, (k + 1) * spw)
      return seg.length ? seg.reduce((s, v) => s + v, 0) / seg.length : 0
    }
    const winDir = Array.from({ length: nWin }, (_, k) => winMean(dir, k))
    const winMag = Array.from({ length: nWin }, (_, k) => winMean(mag, k))
    // Map each strip pixel (a time fraction of the clip) to its decision window.
    const winAt = (x: number) => Math.min(nWin - 1, Math.floor(((x / w) * total) / spw))
    return {
      strip: Array.from({ length: w }, (_, x) =>
        l1BlockRgba(winMag[winAt(x)], winDir[winAt(x)], 0.95),
      ),
      // Lower max-alpha on the waveform overlay so the grey envelope shows through.
      overlay: Array.from({ length: w }, (_, x) =>
        l1BlockRgba(winMag[winAt(x)], winDir[winAt(x)], 0.6),
      ),
    }
  }

  // Signed fallback: confidence (mapped 2*p - 1) or legacy waveformRelevance.
  const signal =
    view === 'confidence' && audio.waveformConfidence.length > 0
      ? audio.waveformConfidence.map(p => 2 * p - 1)
      : audio.waveformRelevance
  const relB = downsample(signal, w, mean)
  const signedRgba = (rel: number, k: number, mult: number, floor: number) => {
    const [r, g, b] = relevanceToRgb(rel)
    const alpha = k * Math.min(1, Math.abs(rel) * mult + floor)
    return `rgba(${r},${g},${b},${alpha.toFixed(3)})`
  }
  return {
    strip: relB.map(rel => signedRgba(rel, 0.75, 2, 0.15)),
    overlay: relB.map(rel => signedRgba(rel, 0.5, 1, 0.3)),
  }
}

function drawWaveform(
  ctx: CanvasRenderingContext2D,
  audio: AudioAnalysis,
  fills: { strip: string[]; overlay: string[] },
  w: number,
  h: number,
) {
  ctx.clearRect(0, 0, w, h)

  const ampBuckets = downsample(audio.waveformAmplitude, w, rms)

  // ── Relevance strip (top) ──────────────────────────────────────────────
  for (let x = 0; x < w; x++) {
    ctx.fillStyle = fills.strip[x]
    ctx.fillRect(x, 0, 1, RELEVANCE_H)
  }

  // Relevance strip border
  ctx.strokeStyle = '#2a2f42'
  ctx.lineWidth = 1
  ctx.strokeRect(0, 0, w, RELEVANCE_H)

  // ── Waveform (bottom portion) ──────────────────────────────────────────
  const maxAmp = Math.max(...ampBuckets, 0.001)
  const waveMaxH = (h - RELEVANCE_H - 4) / 2 // max half-height of waveform bar

  for (let x = 0; x < w; x++) {
    const amp = ampBuckets[x]
    const barH = Math.max(1, (amp / maxAmp) * waveMaxH)

    // Grey base bar
    ctx.fillStyle = `rgba(55,60,80,0.9)`
    ctx.fillRect(x, WAVEFORM_CY - barH, 1, barH * 2)

    // Relevance colour overlay on the bar
    ctx.fillStyle = fills.overlay[x]
    ctx.fillRect(x, WAVEFORM_CY - barH, 1, barH * 2)
  }

  // Centre line
  ctx.strokeStyle = '#2a2f42'
  ctx.lineWidth = 0.5
  ctx.beginPath()
  ctx.moveTo(0, WAVEFORM_CY)
  ctx.lineTo(w, WAVEFORM_CY)
  ctx.stroke()
}

function drawPlayhead(
  ctx: CanvasRenderingContext2D,
  currentTime: number,
  duration: number,
  w: number,
  h: number,
) {
  if (duration <= 0) return
  const x = Math.round((currentTime / duration) * w)
  ctx.save()
  ctx.strokeStyle = '#00e5ff'
  ctx.lineWidth = 1.5
  ctx.shadowColor = '#00e5ff'
  ctx.shadowBlur = 6
  ctx.beginPath()
  ctx.moveTo(x, 0)
  ctx.lineTo(x, h)
  ctx.stroke()
  ctx.restore()
}

export function WaveformRelevanceLayer({
  audio,
  videoRef,
  duration,
  view,
}: WaveformRelevanceLayerProps) {
  const baseCanvasRef = useRef<HTMLCanvasElement>(null)
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null)

  // Redraw on data OR view change. Confidence (fake-prob 0–1) is mapped onto the
  // seismic scale via 2*p - 1 (0.5 → neutral white, → 1 red/fake, → 0 blue/real);
  // relevance is already signed. Confidence falls back to relevance when the
  // per-sample array is missing (older cached results).
  useEffect(() => {
    const canvas = baseCanvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const fills = computeFills(audio, view, CANVAS_W)
    drawWaveform(ctx, audio, fills, CANVAS_W, CANVAS_H)
  }, [audio, view])

  // Playhead: drawn imperatively in a requestAnimationFrame loop straight from
  // the <video> element, so it stays smooth (~60 Hz) WITHOUT triggering any
  // React re-render (which would jitter the whole audio panel).
  useEffect(() => {
    const canvas = overlayCanvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    let rafId = 0
    function tick() {
      ctx!.clearRect(0, 0, CANVAS_W, CANVAS_H)
      const video = videoRef.current
      if (video) drawPlayhead(ctx!, video.currentTime, duration, CANVAS_W, CANVAS_H)
      rafId = requestAnimationFrame(tick)
    }
    rafId = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafId)
  }, [videoRef, duration])

  return (
    <div>
      <div
        style={{
          fontSize: 10,
          fontFamily: 'monospace',
          color: '#4d5470',
          letterSpacing: '0.12em',
          marginBottom: 4,
          display: 'flex',
          justifyContent: 'space-between',
        }}
      >
        <span>
          LAYER 1 — WAVEFORM {view === 'confidence' ? 'CONFIDENCE' : 'RELEVANCE'}
        </span>
        <span style={{ color: '#2a2f42' }}>
          <span style={{ color: '#3b4cc0', marginRight: 8 }}>■ real</span>
          <span style={{ color: '#b40426' }}>■ fake</span>
        </span>
      </div>

      <div
        style={{
          position: 'relative',
          width: '100%',
          height: CANVAS_H,
          borderRadius: 6,
          overflow: 'hidden',
          border: '1px solid #2a2f42',
          backgroundColor: '#0d0f14',
        }}
      >
        {/* Base canvas — waveform + relevance */}
        <canvas
          ref={baseCanvasRef}
          width={CANVAS_W}
          height={CANVAS_H}
          style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}
        />
        {/* Overlay canvas — playhead (separate so base isn't redrawn every frame) */}
        <canvas
          ref={overlayCanvasRef}
          width={CANVAS_W}
          height={CANVAS_H}
          style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}
        />
      </div>

      {/* Time axis */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          marginTop: 3,
          fontSize: 9,
          fontFamily: 'monospace',
          color: '#4d5470',
        }}
      >
        <span>0s</span>
        <span>{(duration / 4).toFixed(1)}s</span>
        <span>{(duration / 2).toFixed(1)}s</span>
        <span>{(duration * 0.75).toFixed(1)}s</span>
        <span>{duration.toFixed(1)}s</span>
      </div>
    </div>
  )
}
