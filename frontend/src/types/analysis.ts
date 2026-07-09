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
  /** Per-word fake-probability (0–1) for the Confidence view (B4). */
  confidence: number
}

export interface FrequencyBands {
  low: number
  mid: number
  high: number
}

/** Bivariate per-band relevance: magnitude (bar width) + direction (side/colour). */
export interface BandValue {
  magnitude: number
  direction: number
}

export interface FrequencyBandsRelevance {
  low: BandValue
  mid: BandValue
  high: BandValue
}

/**
 * L3 band×time heatmap — Confidence: per band, a fakeness-gated band-ablation
 * fraction per 0.64 s decision window (which band carries the fake, and when).
 * Signed: + = the band carried fake evidence in that window, − = pulled real,
 * 0 = real window (no fake to attribute). All three arrays have one value/window.
 */
export interface FrequencyGridConfidence {
  low: number[]
  mid: number[]
  high: number[]
}

/** Per-window magnitude + direction sequence for one band (relevance grid). */
export interface BandSeq {
  magnitude: number[]
  direction: number[]
}

/** L3 band×time heatmap — Relevance: honest faint per-window gradient relevance. */
export interface FrequencyGridRelevance {
  low: BandSeq
  mid: BandSeq
  high: BandSeq
}

export interface AudioAnalysis {
  verdict: 'FAKE' | 'REAL'
  /** Confidence in the audio verdict, 0–1 */
  confidence: number
  /** Per-sample AttnLRP relevance, normalised to [-1, 1]. Length = T_samples. */
  waveformRelevance: number[]
  /**
   * Bivariate Layer-1 Relevance channels (same length as waveformRelevance):
   * magnitude (|R_fake|+|R_real|, drives alpha) + direction (R_fake-R_real, drives
   * hue with |direction| gating). Empty for older cached results (L1 falls back to
   * waveformRelevance).
   */
  waveformMagnitude: number[]
  waveformDirection: number[]
  /**
   * Per-sample fake-probability (0–1) for the Confidence view (B4), same length
   * as waveformRelevance. Map to the seismic scale with `2*p - 1`. May be empty
   * for older cached results.
   */
  waveformConfidence: number[]
  /** Per-sample raw waveform amplitude for display. Length = T_samples. */
  waveformAmplitude: number[]
  sampleRate: number
  wordSegments: WordSegment[]
  /** Ablation-based band evidence — the Layer-3 Confidence view. */
  frequencyBands: FrequencyBands
  /**
   * Bivariate per-band relevance — the Layer-3 Relevance view (B4): magnitude
   * (bar width) + direction (side/colour). Null for older cached results (L3 then
   * falls back to the frequencyBands Confidence shape).
   */
  frequencyBandsRelevance: FrequencyBandsRelevance | null
  /**
   * L3 band×time heatmaps (replace the 3-bar chart). Confidence = gated ablation
   * grid (sharp; lights up where a band carries the fake). Relevance = faint
   * per-window gradient grid. Null for older cached results / paths that don't
   * compute them (e.g. multimodal) — L3 then falls back to the 3-bar bands.
   */
  frequencyGridConfidence: FrequencyGridConfidence | null
  frequencyGridRelevance: FrequencyGridRelevance | null
}

export interface AttentionShift {
  region: string
  // Bivariate LRP before/after (roadmap I4): magnitude = relevance/attention
  // share (|R_fake|+|R_real|); direction = signed verdict lean (R_fake−R_real,
  // + fake-supporting, − real-supporting).
  magnitudeBefore: number
  magnitudeAfter: number
  directionBefore: number
  directionAfter: number
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
  /**
   * Per-frame CLEAN-crop region-partition overlay (I4 debug view): one PNG data
   * URI per frame (tan-tinted facial regions + borders + labels), aligned to the
   * clean player's frames. Empty on the face-less fallback / pre-regen cache.
   */
  regionMaskFrames?: string[]
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
  /** Per-frame CLEAN-crop region-partition overlay PNGs (data URIs, I4 debug). */
  regionMaskFrames?: string[]
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
   * Per-16-frame-chunk timelines (A1). One value per chunk (window):
   * - `perChunkConfidence` — raw per-window fake probability (0–1, NOT max-pooled;
   *   the verdict still is). The confidence timeline classifies each chunk so a
   *   short manipulation shows as FAKE only where it occurs.
   * - `perChunkRelevanceMagnitude` — `mean(|relevance|)` per chunk (timeline height).
   * - `perChunkRelevanceSign` — sign of net relevance (+1 fake-supporting / −1 real).
   */
  perChunkConfidence: number[]
  perChunkRelevanceMagnitude: number[]
  perChunkRelevanceSign: number[]
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

/** Audio-panel view toggle (B4): signed relevance vs. per-window confidence. */
export type AudioView = 'relevance' | 'confidence'

export type AnalysisState =
  | { status: 'idle' }
  | { status: 'scanning' }
  | { status: 'done'; result: AnalysisResult }
  | { status: 'error'; message: string }
