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

export interface Phase3Result {
  degradedHeatmapFrames: string[]
  degradedConfidence: number
  params: {
    crf: number
    fps: number
    noiseSigma: number
  }
}

export interface Phase4Result {
  perturbedFrames: string[]
  perturbedConfidence: number
  differenceFrames: string[]
  attackMethod: 'FGSM' | 'PGD'
  epsilon: number
  attentionShift: AttentionShift[]
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
  xaiMode: 'lrp' | 'rollout'
  /** Top spatial anomaly regions with their LRP contribution scores */
  anomalyRegions: { region: string; score: number }[]
  audio: AudioAnalysis | null
  cropBox: CropBox | null
  phase3: Phase3Result | null
  phase4: Phase4Result | null
}

export type AnalysisState =
  | { status: 'idle' }
  | { status: 'scanning' }
  | { status: 'done'; result: AnalysisResult }
  | { status: 'error'; message: string }

export type XaiMode = 'lrp' | 'rollout'
