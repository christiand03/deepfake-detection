// ── Canonical data contracts shared by API client, hooks and components ──

export interface ClipMeta {
  id: string
  label: 'FAKE' | 'REAL'
  title: string
  /** Path relative to /public, e.g. "/clips/obama_fake.mp4" */
  videoSrc: string
  /** Path to poster image, e.g. "/clips/obama_fake.jpg" */
  posterSrc: string
  /** Duration in seconds */
  duration: number
  /** Nominal frame-rate used when mapping currentTime → frame index */
  fps: number
  hasAudio: boolean
}

export interface WordSegment {
  word: string
  start: number
  end: number
  relevance: number
}

export interface FrequencyBands {
  low: number
  mid: number
  high: number
}

export interface AudioAnalysis {
  verdict: 'FAKE' | 'REAL'
  /** Confidence in the audio verdict, 0–1 */
  confidence: number
  /** Per-sample AttnLRP relevance, normalised to [-1, 1]. Length = T_samples. */
  waveformRelevance: number[]
  /** Per-sample raw waveform amplitude for display. Length = T_samples. */
  waveformAmplitude: number[]
  sampleRate: number
  wordSegments: WordSegment[]
  frequencyBands: FrequencyBands
}

export interface AttentionShift {
  region: string
  before: number
  after: number
}

export interface AudioRobustness {
  baseConfidence: number
  degradedConfidence: number
  baseFrequencyBands: FrequencyBands
  degradedFrequencyBands: FrequencyBands
  bitrate: number
}

export interface Phase3Result {
  /** Crop-space (224) heatmaps overlaid on the face-crop before/after players (I2). */
  degradedHeatmapFrames: string[]
  cleanHeatmapFrames: string[]
  /** Face-crop video URLs (served at /media) behind the heatmaps. */
  cleanVideoUrl?: string | null
  degradedVideoUrl?: string | null
  /** Verdict after degradation (reported by the backend; never re-derive it). */
  degradedVerdict: 'FAKE' | 'REAL'
  degradedConfidence: number
  /**
   * Clean baseline from the SAME model as the degraded pass (I1/I3) — use these
   * for the "clean" side instead of the main analysis result, which may have
   * used a different model (unimodal vs. multimodal).
   */
  baselineVerdict: 'FAKE' | 'REAL'
  baselineConfidence: number
  params: {
    crf: number
    fps: number
    noiseSigma: number
  }
  attentionShift: AttentionShift[]
  audioRobustness?: AudioRobustness
}

export interface Phase4Result {
  /** Crop-space (224) heatmaps for the face-crop before/after players (I2). */
  perturbedFrames: string[]
  cleanHeatmapFrames: string[]
  /** Face-crop video URLs (served at /media) behind the heatmaps. */
  cleanVideoUrl?: string | null
  attackedVideoUrl?: string | null
  /** Verdict after the attack (reported by the backend; never re-derive it). */
  perturbedVerdict: 'FAKE' | 'REAL'
  /** Confidence in the attacked verdict (always ≥ 0.5). */
  perturbedConfidence: number
  differenceFrames: string[]
  attackMethod: 'FGSM' | 'PGD'
  epsilon: number
  attentionShift: AttentionShift[]
  audioAttentionShift?: AttentionShift[]
  attackModalities?: string
  /**
   * Clean baseline from the SAME model as the attack (I3) — use these for the
   * "clean" side of the comparison instead of the main analysis result, which
   * may have been produced by a different model (unimodal vs. multimodal).
   */
  cleanVerdict: 'FAKE' | 'REAL'
  cleanConfidence: number
}

export interface CropBox {
  /** Pixel coordinates of the face crop inside the original frame */
  x1: number
  y1: number
  x2: number
  y2: number
  origW: number
  origH: number
}

export interface AnalysisResult {
  clipId: string
  verdict: 'FAKE' | 'REAL'
  /** Overall confidence, 0–1 */
  confidence: number
  /** Per-frame confidence scores, length = number of video frames */
  perFrameScores: number[]
  /**
   * Per-frame AttnLRP heatmap images encoded as base64 data URIs.
   * Each string is a full "data:image/png;base64,..." URI ready for canvas drawImage.
   */
  heatmapFrames: string[]
  /** Top spatial anomaly regions with their LRP contribution scores */
  anomalyRegions: { region: string; score: number }[]
  audio: AudioAnalysis | null
  cropBox: CropBox | null
  phase3: Phase3Result | null
  phase4: Phase4Result | null
  /** Which model produced this result. Defaults to 'unimodal' when absent. */
  modelMode?: 'unimodal' | 'multimodal'
  /** Active fusion mode — only meaningful when modelMode === 'multimodal'. */
  fusionMode?: 'cross_attention' | 'concat' | null
}

export type FusionMode = 'cross_attention' | 'concat'
export type ModelMode = 'unimodal' | 'multimodal'

export type AnalysisState =
  | { status: 'idle' }
  | { status: 'scanning' }
  | { status: 'done'; result: AnalysisResult }
  | { status: 'error'; message: string }
