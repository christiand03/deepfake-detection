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
import { seismicToRgb } from '../../lib/seismicColormap'
import type { AudioAnalysis } from '../../types/analysis'

interface WaveformRelevanceLayerProps {
  audio: AudioAnalysis
  currentTime: number
  /** Clip duration in seconds, used for the playhead position */
  duration: number
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

function drawWaveform(
  ctx: CanvasRenderingContext2D,
  audio: AudioAnalysis,
  w: number,
  h: number,
) {
  ctx.clearRect(0, 0, w, h)

  const ampBuckets = downsample(audio.waveformAmplitude, w, rms)
  const relBuckets = downsample(audio.waveformRelevance, w, mean)

  // ── Relevance strip (top) ──────────────────────────────────────────────
  for (let x = 0; x < w; x++) {
    const rel = relBuckets[x]
    const [r, g, b] = seismicToRgb(rel)
    const alpha = 0.75 * Math.min(1, Math.abs(rel) * 2 + 0.15)
    ctx.fillStyle = `rgba(${r},${g},${b},${alpha.toFixed(3)})`
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
    const rel = relBuckets[x]
    const [r, g, b] = seismicToRgb(rel)
    const alpha = 0.5 * Math.min(1, Math.abs(rel) + 0.3)

    // Grey base bar
    ctx.fillStyle = `rgba(55,60,80,0.9)`
    ctx.fillRect(x, WAVEFORM_CY - barH, 1, barH * 2)

    // Seismic colour overlay on the bar
    ctx.fillStyle = `rgba(${r},${g},${b},${alpha.toFixed(3)})`
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
  currentTime,
  duration,
}: WaveformRelevanceLayerProps) {
  const baseCanvasRef = useRef<HTMLCanvasElement>(null)
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null)

  // Draw waveform once on data change
  useEffect(() => {
    const canvas = baseCanvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    drawWaveform(ctx, audio, CANVAS_W, CANVAS_H)
  }, [audio])

  // Redraw playhead on time change
  useEffect(() => {
    const canvas = overlayCanvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H)
    drawPlayhead(ctx, currentTime, duration, CANVAS_W, CANVAS_H)
  }, [currentTime, duration])

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
        <span>LAYER 1 — WAVEFORM RELEVANCE</span>
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
