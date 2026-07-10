/**
 * API client — always calls the live FastAPI backend for analysis.
 *
 * fetchClips falls back to DEMO_CLIPS when VITE_USE_MOCK=true (offline dev).
 * analyzeClip always uses the real backend — no mock path.
 */

import type { AnalysisResult, ClipMeta, Phase3Result, Phase4Result } from '../types/analysis'
import { DEMO_CLIPS, makeMockPhase3Result, makeMockPhase4Result } from '../lib/mockData'

const USE_MOCK = import.meta.env.VITE_USE_MOCK === 'true'

// ── Simulate network latency in mock mode ────────────────────────────────────

function delay(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

// ── Public API functions ─────────────────────────────────────────────────────

export async function fetchClips(): Promise<ClipMeta[]> {
  // Always try the real backend first so the UI reflects the current
  // conf/clips.json without rebuilding.  When the backend is unreachable and
  // USE_MOCK is enabled (local dev without the server), fall back to DEMO_CLIPS.
  try {
    const res = await fetch('/api/clips')
    if (!res.ok) throw new Error(`GET /api/clips \u2192 ${res.status}`)
    return res.json() as Promise<ClipMeta[]>
  } catch (err) {
    if (USE_MOCK) {
      console.warn('[fetchClips] Backend unreachable, falling back to DEMO_CLIPS:', err)
      return DEMO_CLIPS
    }
    throw err
  }
}

export async function analyzeClip(
  clipId: string,
  opts?: { useMultimodal?: boolean; fusionMode?: 'cross_attention' | 'concat' },
): Promise<AnalysisResult> {
  // Mode is passed as query params (the endpoint reads them, not the body).
  const params = new URLSearchParams()
  if (opts?.useMultimodal) {
    params.set('use_multimodal', 'true')
    params.set('fusion_mode', opts.fusionMode ?? 'cross_attention')
  }
  const query = params.toString()
  const url = `/api/analyze/${clipId}${query ? `?${query}` : ''}`
  const res = await fetch(url, {
    method: 'POST',
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`POST ${url} → ${res.status}: ${text}`)
  }
  return res.json() as Promise<AnalysisResult>
}

export async function runRobustnessTest(
  clipId: string,
  params: {
    crf: number
    fps: number
    noiseSigma: number
    upscale?: boolean
    audioBitrate?: number
    useMultimodal?: boolean
    fusionMode?: 'cross_attention' | 'concat'
  },
  baseResult: AnalysisResult,
): Promise<Phase3Result> {
  if (USE_MOCK) {
    await delay(1400)
    return makeMockPhase3Result(params, baseResult)
  }
  const res = await fetch('/api/robustness', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      clip_id: clipId,
      crf: params.crf,
      fps: params.fps,
      noise_sigma: params.noiseSigma,
      upscale: params.upscale ?? false,
      audio_bitrate: params.audioBitrate ?? null,
      ...(params.useMultimodal !== undefined && { use_multimodal: params.useMultimodal }),
      ...(params.fusionMode !== undefined && { fusion_mode: params.fusionMode }),
    }),
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`POST /api/robustness → ${res.status}: ${text}`)
  }
  return res.json() as Promise<Phase3Result>
}

export async function runAdversarialAttack(
  clipId: string,
  method: 'FGSM' | 'PGD',
  epsilon: number,
  steps: number,
  baseResult: AnalysisResult,
  useMultimodal?: boolean,
  attackModalities?: 'video' | 'audio' | 'both',
  audioEpsilon?: number,
): Promise<Phase4Result> {
  if (USE_MOCK) {
    await delay(method === 'FGSM' ? 1200 : 1800)
    return makeMockPhase4Result(method, epsilon, baseResult, useMultimodal, attackModalities)
  }
  const res = await fetch('/api/adversarial', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      clip_id: clipId,
      method,
      epsilon,
      steps,
      ...(useMultimodal !== undefined && { use_multimodal: useMultimodal }),
      ...(attackModalities !== undefined && { attack_modalities: attackModalities }),
      ...(audioEpsilon !== undefined && { audio_epsilon: audioEpsilon }),
    }),
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`POST /api/adversarial → ${res.status}: ${text}`)
  }
  return res.json() as Promise<Phase4Result>
}
