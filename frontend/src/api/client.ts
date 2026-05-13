/**
 * API client — switches between live FastAPI backend and mock data
 * based on the VITE_USE_MOCK environment variable.
 *
 * Set VITE_USE_MOCK=false in .env.local to use the live backend.
 */

import type { AnalysisResult, ClipMeta, Phase3Result, Phase4Result, XaiMode } from '../types/analysis'
import { DEMO_CLIPS, makeMockPhase3Result, makeMockPhase4Result, makeMockResult } from '../lib/mockData'

const USE_MOCK = import.meta.env.VITE_USE_MOCK !== 'false'

// ── Simulate network latency in mock mode ────────────────────────────────────

function delay(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

// ── Public API functions ─────────────────────────────────────────────────────

export async function fetchClips(): Promise<ClipMeta[]> {
  if (USE_MOCK) {
    await delay(200)
    return DEMO_CLIPS
  }
  const res = await fetch('/api/clips')
  if (!res.ok) throw new Error(`GET /api/clips → ${res.status}`)
  return res.json() as Promise<ClipMeta[]>
}

export async function analyzeClip(
  clipId: string,
  xaiMode: XaiMode,
): Promise<AnalysisResult> {
  if (USE_MOCK) {
    // Simulate model inference time
    await delay(1800 + Math.random() * 600)
    const clip = DEMO_CLIPS.find(c => c.id === clipId)
    if (!clip) throw new Error(`Unknown clip id: ${clipId}`)
    return makeMockResult(clip)
  }
  const res = await fetch(`/api/analyze/${clipId}?xai_mode=${xaiMode}`, {
    method: 'POST',
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`POST /api/analyze/${clipId} → ${res.status}: ${text}`)
  }
  return res.json() as Promise<AnalysisResult>
}

export async function runRobustnessTest(
  clipId: string,
  params: { crf: number; fps: number; noiseSigma: number },
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
): Promise<Phase4Result> {
  if (USE_MOCK) {
    await delay(method === 'FGSM' ? 1200 : 1800)
    return makeMockPhase4Result(method, epsilon, baseResult)
  }
  const res = await fetch('/api/adversarial', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ clip_id: clipId, method, epsilon, steps }),
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`POST /api/adversarial → ${res.status}: ${text}`)
  }
  return res.json() as Promise<Phase4Result>
}
