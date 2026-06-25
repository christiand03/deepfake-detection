/**
 * mockData.ts — Realistic mock AnalysisResult for development.
 *
 * The heatmap frames are generated programmatically as 1×1 base64 PNGs with
 * varying seismic colors so that the canvas overlay has something real to render.
 * Per-frame scores simulate a plausible detector output for a FAKE clip.
 */

import type {
  AnalysisResult,
  ClipMeta,
  Phase3Result,
  Phase4Result,
} from '../types/analysis'

// ── Demo clip registry ──────────────────────────────────────────────────────

export const DEMO_CLIPS: ClipMeta[] = [
  {
    id: 'clip_01',
    label: 'FAKE',
    title: 'id00012 — fake video fake audio',
    videoSrc: '/clips/id00012__21Uxsk56VDQ__00001__fake_video_fake_audio.mp4',
    posterSrc: '',
    duration: 9.48,
    fps: 25,
    hasAudio: true,
  },
  {
    id: 'clip_02',
    label: 'FAKE',
    title: 'id00012 — fake video real audio',
    videoSrc: '/clips/id00012__21Uxsk56VDQ__00001__fake_video_real_audio.mp4',
    posterSrc: '',
    duration: 9.44,
    fps: 25,
    hasAudio: true,
  },
  {
    id: 'clip_03',
    label: 'REAL',
    title: 'id00012 — real',
    videoSrc: '/clips/id00012__21Uxsk56VDQ__00001__real.mp4',
    posterSrc: '',
    duration: 9.44,
    fps: 25,
    hasAudio: true,
  },
  {
    id: 'clip_04',
    label: 'FAKE',
    title: 'id00012 — real video fake audio',
    videoSrc: '/clips/id00012__21Uxsk56VDQ__00001__real_video_fake_audio.mp4',
    posterSrc: '',
    duration: 9.48,
    fps: 25,
    hasAudio: true,
  },
  {
    id: 'clip_05',
    label: 'FAKE',
    title: 'id00012 — fake video fake audio',
    videoSrc: '/clips/id00012__21Uxsk56VDQ__00002__fake_video_fake_audio.mp4',
    posterSrc: '',
    duration: 14.96,
    fps: 25,
    hasAudio: true,
  },
]

// ── Tiny seismic-tinted 4×4 PNG generator ───────────────────────────────────

/**
 * Returns a 1-pixel data URI where the pixel is seismic-mapped from `value`.
 * In production this is replaced by a proper per-frame PNG from the backend.
 */
function makeSeismicDataUri(value: number): string {
  // Map [-1,1] → blue→white→red (matching seismic stops)
  const t = (value + 1) / 2 // 0..1
  let r: number, g: number, b: number
  if (t < 0.5) {
    // blue → white
    const s = t / 0.5
    r = Math.round(lerp(0, 255, s))
    g = Math.round(lerp(0, 255, s))
    b = 255
  } else {
    // white → red
    const s = (t - 0.5) / 0.5
    r = 255
    g = Math.round(lerp(255, 0, s))
    b = Math.round(lerp(255, 0, s))
  }
  // Build a 1×1 BMP-style data via canvas — we just return a color data uri
  // Actually build a real tiny PNG via raw bytes would be complex; return SVG data URI instead
  // (canvas drawImage accepts SVG data URIs fine)
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="224" height="224"><rect width="224" height="224" fill="rgb(${r},${g},${b})" opacity="0.55"/></svg>`
  return `data:image/svg+xml;base64,${btoa(svg)}`
}

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t
}

// ── Per-frame score curve (16 frames, FAKE clip peaking in the middle) ──────

function makeFakeScoreCurve(n = 16): number[] {
  return Array.from({ length: n }, (_, i) => {
    const t = i / (n - 1)
    // Bell curve centred at 0.6 → high confidence in middle frames
    return 0.55 + 0.38 * Math.exp(-((t - 0.6) ** 2) / 0.06) + (Math.random() - 0.5) * 0.04
  })
}

function makeRealScoreCurve(n = 16): number[] {
  return Array.from({ length: n }, () => 0.12 + Math.random() * 0.18)
}

// ── Waveform + relevance mock data ──────────────────────────────────────────

function makeMockAudioData(isFake: boolean) {
  const sampleRate = 16000
  const durationSec = 6
  const T = sampleRate * durationSec

  // Amplitude: speech-like envelope
  const amplitude = Array.from({ length: T }, (_, i) => {
    const t = i / sampleRate
    return Math.sin(2 * Math.PI * 220 * t) * 0.3 * Math.exp(-0.1 * ((t - 3) ** 2))
  })

  // Relevance: fake has strong positive spikes around word transitions
  const relevance = Array.from({ length: T }, (_, i) => {
    const t = i / sampleRate
    if (!isFake) return (Math.random() - 0.5) * 0.2
    // Fake: peaks at ~1s, ~2.5s, ~4.5s
    const peaks = [1.0, 2.5, 4.5]
    const spike = peaks.reduce((acc, p) => acc + Math.exp(-((t - p) ** 2) / 0.05), 0)
    return Math.min(1, spike * 0.7 + (Math.random() - 0.5) * 0.1)
  })

  return { amplitude, relevance, sampleRate }
}

// ── Main mock result factory ─────────────────────────────────────────────────

export function makeMockResult(clip: ClipMeta): AnalysisResult {
  const isFake = clip.label === 'FAKE'
  const nFrames = Math.round(clip.duration * clip.fps)
  const scores = isFake ? makeFakeScoreCurve(nFrames) : makeRealScoreCurve(nFrames)
  const confidence = isFake ? 0.924 : 0.871

  const heatmapFrames = scores.map(s => makeSeismicDataUri(isFake ? s * 2 - 1 : -(s)))

  const audio = clip.hasAudio
    ? (() => {
        const { amplitude, relevance, sampleRate } = makeMockAudioData(isFake)
        // Mock per-sample confidence: map signed relevance onto a fake-prob 0–1
        // (0.5 neutral) so the Confidence view (B4) reads coherently.
        const waveformConfidence = relevance.map(r =>
          Math.min(1, Math.max(0, 0.5 + 0.5 * r)),
        )
        const rawWords = isFake
          ? [
              { word: 'We', start: 0.3, end: 0.5, relevance: 0.12 },
              { word: 'must', start: 0.55, end: 0.8, relevance: 0.81 },
              { word: 'act', start: 0.85, end: 1.05, relevance: 0.93 },
              { word: 'now', start: 1.1, end: 1.3, relevance: 0.77 },
              { word: 'to', start: 1.4, end: 1.55, relevance: 0.08 },
              { word: 'address', start: 1.6, end: 2.0, relevance: 0.62 },
              { word: 'this', start: 2.05, end: 2.2, relevance: 0.31 },
              { word: 'crisis', start: 2.25, end: 2.7, relevance: 0.89 },
            ]
          : [
              { word: 'Thank', start: 0.2, end: 0.5, relevance: -0.05 },
              { word: 'you', start: 0.55, end: 0.7, relevance: -0.12 },
              { word: 'all', start: 0.75, end: 0.9, relevance: -0.08 },
              { word: 'for', start: 0.95, end: 1.1, relevance: -0.03 },
              { word: 'being', start: 1.15, end: 1.5, relevance: -0.19 },
              { word: 'here', start: 1.55, end: 1.8, relevance: -0.07 },
            ]
        return {
          verdict: isFake ? ('FAKE' as const) : ('REAL' as const),
          confidence: isFake ? 0.924 : 0.871,
          waveformRelevance: relevance,
          waveformConfidence,
          waveformAmplitude: amplitude,
          sampleRate,
          wordSegments: rawWords.map(w => ({
            ...w,
            confidence: Math.min(1, Math.max(0, 0.5 + 0.5 * w.relevance)),
          })),
          frequencyBands: isFake
            ? { low: 0.21, mid: 0.54, high: 0.25 }
            : { low: -0.15, mid: -0.52, high: -0.33 },
          frequencyBandsRelevance: isFake
            ? { low: 0.18, mid: 0.6, high: 0.22 }
            : { low: -0.12, mid: -0.58, high: -0.3 },
        }
      })()
    : null

  // Per-chunk timelines (A1): one value per 16-frame chunk. For the FAKE mock the
  // manipulation sits in the middle chunks, so only those classify as FAKE.
  const nChunks = Math.max(1, Math.ceil(nFrames / 16))
  const perChunkConfidence = Array.from({ length: nChunks }, (_, c) => {
    const t = nChunks === 1 ? 0.5 : c / (nChunks - 1)
    if (!isFake) return 0.1 + Math.random() * 0.15
    return 0.2 + 0.75 * Math.exp(-((t - 0.55) ** 2) / 0.04) + (Math.random() - 0.5) * 0.04
  })
  // Small absolute values, like the real clip-global-normalised mean(|relevance|)
  // per chunk (the frontend applies a display gain).
  const perChunkRelevanceMagnitude = perChunkConfidence.map(
    p => Math.abs(p - 0.5) * 0.4 + Math.random() * 0.02,
  )
  const perChunkRelevanceSign = perChunkConfidence.map(p => (p > 0.5 ? 1 : -1))

  return {
    clipId: clip.id,
    verdict: clip.label,
    confidence,
    perFrameScores: scores,
    perChunkConfidence,
    perChunkRelevanceMagnitude,
    perChunkRelevanceSign,
    heatmapFrames,
    anomalyRegions: isFake
      ? [
          { region: 'Mouth', score: 0.84 },
          { region: 'Left Eye', score: 0.41 },
          { region: 'Jaw', score: 0.22 },
          { region: 'Forehead', score: 0.09 },
        ]
      : [
          { region: 'Mouth', score: 0.07 },
          { region: 'Left Eye', score: 0.04 },
          { region: 'Jaw', score: 0.02 },
          { region: 'Forehead', score: 0.01 },
        ],
    audio,
    cropBox: null,
    phase3: null,
    phase4: null,
  }
}

// ── Phase 3 / 4 mock factories ───────────────────────────────────────────────

function makeDegradedDataUri(value: number, degradation: number): string {
  const t = (value + 1) / 2
  let r: number, g: number, b: number
  if (t < 0.5) {
    const s = t / 0.5
    r = Math.round(lerp(0, 255, s))
    g = r
    b = 255
  } else {
    const s = (t - 0.5) / 0.5
    r = 255
    g = Math.round(lerp(255, 0, s))
    b = g
  }
  r = Math.round(lerp(r, 128, degradation))
  g = Math.round(lerp(g, 128, degradation))
  b = Math.round(lerp(b, 128, degradation))
  const opacity = (0.55 * (1 - degradation * 0.4)).toFixed(2)
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="224" height="224"><rect width="224" height="224" fill="rgb(${r},${g},${b})" opacity="${opacity}"/></svg>`
  return `data:image/svg+xml;base64,${btoa(svg)}`
}

function makePerturbedDataUri(value: number, epsilon: number): string {
  // Shift attention away from face centre — mix original heatmap toward neutral grey/blue
  const shift = Math.min(1, epsilon * 12)
  const t = (value + 1) / 2
  let r: number, g: number, b: number
  if (t < 0.5) {
    const s = t / 0.5
    r = Math.round(lerp(0, 255, s))
    g = r
    b = 255
  } else {
    const s = (t - 0.5) / 0.5
    r = 255
    g = Math.round(lerp(255, 0, s))
    b = g
  }
  // After attack: attention shifts to non-face periphery (neutral, slightly green-tinted)
  r = Math.round(lerp(r, 60, shift))
  g = Math.round(lerp(g, 80, shift))
  b = Math.round(lerp(b, 70, shift))
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="224" height="224"><rect width="224" height="224" fill="rgb(${r},${g},${b})" opacity="0.55"/></svg>`
  return `data:image/svg+xml;base64,${btoa(svg)}`
}

function makeDifferenceDataUri(epsilon: number): string {
  const intensity = Math.min(255, Math.round(epsilon * 2500))
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="224" height="224"><rect width="224" height="224" fill="#0d0f14"/><rect x="20" y="20" width="184" height="184" fill="rgb(${intensity},${Math.round(intensity * 0.3)},${Math.round(intensity * 0.2)})" opacity="0.65"/></svg>`
  return `data:image/svg+xml;base64,${btoa(svg)}`
}

/**
 * Generate a mock Phase3Result simulating social-media degradation.
 * Degradation score is a weighted combination of compression (CRF), frame-rate
 * reduction, and Gaussian noise.
 */
export function makeMockPhase3Result(
  params: { crf: number; fps: number; noiseSigma: number; audioBitrate?: number },
  baseResult: AnalysisResult,
): Phase3Result {
  const { crf, fps, noiseSigma, audioBitrate } = params
  const degradation = Math.min(
    1,
    ((crf - 18) / 33) * 0.5 + (1 - fps / 30) * 0.3 + (noiseSigma / 50) * 0.2,
  )
  const degradedConfidence = Math.max(0.05, baseResult.confidence * (1 - degradation * 0.85))
  const degradedHeatmapFrames = baseResult.perFrameScores.map(score =>
    makeDegradedDataUri(
      baseResult.verdict === 'FAKE' ? score * 2 - 1 : -score,
      degradation,
    ),
  )

  const audioRobustness =
    audioBitrate !== undefined
      ? (() => {
          const isFake = baseResult.verdict === 'FAKE'
          const baseConf = baseResult.audio?.confidence ?? (isFake ? 0.88 : 0.82)
          const compressionEffect = Math.max(0, (128 - audioBitrate) / 128)
          const baseBands = isFake
            ? { low: 0.21, mid: 0.54, high: 0.25 }
            : { low: 0.15, mid: 0.52, high: 0.33 }
          return {
            baseConfidence: baseConf,
            degradedConfidence: Math.max(0.05, baseConf * (1 - compressionEffect * 0.4)),
            baseFrequencyBands: baseBands,
            degradedFrequencyBands: {
              low: Math.max(0.01, baseBands.low * (1 - compressionEffect * 0.08)),
              mid: Math.max(0.01, baseBands.mid * (1 - compressionEffect * 0.45)),
              high: Math.max(0.01, baseBands.high * (1 - compressionEffect * 0.55)),
            },
            bitrate: audioBitrate,
          }
        })()
      : undefined

  // Mock degradedConfidence drops toward 0 (confidence in the ORIGINAL verdict);
  // a flip is simulated once it crosses 0.5. Report the degraded verdict + its
  // own-verdict confidence (≥ 0.5) like the backend does.
  const degradedFlipped = degradedConfidence < 0.5
  const degradedVerdict: 'FAKE' | 'REAL' = degradedFlipped
    ? baseResult.verdict === 'FAKE'
      ? 'REAL'
      : 'FAKE'
    : baseResult.verdict
  return {
    degradedHeatmapFrames,
    cleanHeatmapFrames: degradedHeatmapFrames,
    cleanVideoUrl: null,
    degradedVideoUrl: null,
    degradedVerdict,
    degradedConfidence: degradedFlipped ? 1 - degradedConfidence : degradedConfidence,
    baselineVerdict: baseResult.verdict,
    baselineConfidence: baseResult.confidence,
    params,
    attentionShift: [
      { region: 'Mouth',      before: 0.84, after: Math.max(0.05, 0.84 - degradation * 0.5) },
      { region: 'Left Eye',   before: 0.41, after: Math.max(0.03, 0.41 - degradation * 0.3) },
      { region: 'Right Eye',  before: 0.33, after: Math.max(0.02, 0.33 - degradation * 0.25) },
      { region: 'Jaw',        before: 0.22, after: Math.max(0.02, 0.22 - degradation * 0.2) },
      { region: 'Background', before: 0.04, after: Math.min(0.60, 0.04 + degradation * 0.3) },
    ],
    ...(audioRobustness !== undefined ? { audioRobustness } : {}),
  }
}

/**
 * Generate a mock Phase4Result for an FGSM or PGD adversarial attack.
 */
export function makeMockPhase4Result(
  attackMethod: 'FGSM' | 'PGD',
  epsilon: number,
  baseResult: AnalysisResult,
  useMultimodal?: boolean,
  attackModalities?: string,
): Phase4Result {
  const isFake = baseResult.verdict === 'FAKE'
  const attackStrength = attackMethod === 'PGD' ? epsilon * 2.5 : epsilon * 1.5
  const perturbedConfidence = Math.max(
    0.05,
    baseResult.confidence - attackStrength * (isFake ? 1.1 : 0.7),
  )

  const perturbedFrames = baseResult.perFrameScores.map(score =>
    makePerturbedDataUri(isFake ? score * 2 - 1 : -score, epsilon),
  )
  const differenceFrames = baseResult.perFrameScores.map(() => makeDifferenceDataUri(epsilon))

  // Mock perturbedConfidence drops toward 0 (confidence in the ORIGINAL class);
  // a flip is simulated once it crosses 0.5. Report the perturbed verdict + its
  // own-verdict confidence (≥ 0.5) like the backend does.
  const flipped = perturbedConfidence < 0.5
  const perturbedVerdict: 'FAKE' | 'REAL' = flipped
    ? baseResult.verdict === 'FAKE'
      ? 'REAL'
      : 'FAKE'
    : baseResult.verdict
  const base: Phase4Result = {
    perturbedFrames,
    cleanHeatmapFrames: perturbedFrames,
    cleanVideoUrl: null,
    attackedVideoUrl: null,
    perturbedVerdict,
    perturbedConfidence: flipped ? 1 - perturbedConfidence : perturbedConfidence,
    differenceFrames,
    attackMethod,
    epsilon,
    cleanVerdict: baseResult.verdict,
    cleanConfidence: baseResult.confidence,
    attentionShift: [
      { region: 'Mouth', before: 0.84, after: Math.max(0.04, 0.84 - attackStrength * 0.9) },
      { region: 'Left Eye', before: 0.41, after: Math.max(0.03, 0.41 - attackStrength * 0.6) },
      { region: 'Jaw', before: 0.22, after: Math.max(0.02, 0.22 - attackStrength * 0.4) },
      { region: 'Shoulder', before: 0.03, after: Math.min(0.94, 0.03 + attackStrength * 0.8) },
      {
        region: 'Background',
        before: 0.01,
        after: Math.min(0.82, 0.01 + attackStrength * 0.6),
      },
    ],
  }

  if (useMultimodal) {
    base.audioAttentionShift = [
      { region: 'Low 0\u2013500 Hz', before: 0.21, after: Math.max(0.02, 0.21 - attackStrength * 0.4) },
      { region: 'Mid 500\u20134 kHz', before: 0.54, after: Math.max(0.03, 0.54 - attackStrength * 0.7) },
      { region: 'High 4\u20138 kHz', before: 0.25, after: Math.min(0.88, 0.25 + attackStrength * 0.9) },
    ]
    base.attackModalities = attackModalities ?? 'both'
  }

  return base
}
