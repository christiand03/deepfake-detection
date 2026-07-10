"""Pydantic v2 response and request schemas for the FastAPI backend.

These mirror the TypeScript types in ``frontend/src/types/analysis.ts`` exactly.
Field names use camelCase to match the JSON contract expected by the React frontend.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ClipMetaSchema(BaseModel):
    id: str
    label: Literal["FAKE", "REAL"]
    title: str
    videoSrc: str
    posterSrc: str
    duration: float
    fps: float
    hasAudio: bool
    # Hierarchy fields for the identity->scenario->segment->variant selector
    # (roadmap H1). Default "" keeps older clips.json entries loadable.
    identity: str = ""
    scenario: str = ""
    segment: str = ""
    variant: str = ""


class WordSegmentSchema(BaseModel):
    word: str
    start: float
    end: float
    relevance: float
    # Per-word fake-probability (0–1) for the Confidence view (B4); max over the
    # windows the word overlaps. Defaults to 0 for older cached results.
    confidence: float = 0.0


class FrequencyBandsSchema(BaseModel):
    low: float
    mid: float
    high: float


class BandValueSchema(BaseModel):
    """Bivariate per-band relevance: magnitude (bar width) + direction (side/colour)."""

    magnitude: float
    direction: float


class FrequencyBandsRelevanceSchema(BaseModel):
    """Layer-3 Relevance view: a magnitude/direction pair per frequency band."""

    low: BandValueSchema
    mid: BandValueSchema
    high: BandValueSchema


class FrequencyGridConfidenceSchema(BaseModel):
    """L3 band x time heatmap (Confidence): per band, a fakeness-gated band-ablation
    fraction per 0.64-s decision window (which band carries the fake, and when)."""

    low: list[float] = []
    mid: list[float] = []
    high: list[float] = []


class BandSeqSchema(BaseModel):
    """Per-window magnitude + direction sequence for one frequency band."""

    magnitude: list[float] = []
    direction: list[float] = []


class FrequencyGridRelevanceSchema(BaseModel):
    """L3 band x time heatmap (Relevance): bivariate gradient relevance per window."""

    low: BandSeqSchema
    mid: BandSeqSchema
    high: BandSeqSchema


class AudioAnalysisSchema(BaseModel):
    verdict: Literal["FAKE", "REAL"]
    confidence: float
    waveformRelevance: list[float]
    # Bivariate Layer-1 Relevance channels (same length as waveformRelevance):
    # magnitude (|R_fake|+|R_real|, drives alpha) + direction (R_fake-R_real, drives
    # hue, |direction|-gated). Empty for older cached results (L1 falls back to
    # waveformRelevance).
    waveformMagnitude: list[float] = []
    waveformDirection: list[float] = []
    # Per-sample fake-probability (0–1) for the Layer-1 Confidence view (B4),
    # same length as waveformRelevance. Empty for older cached results.
    waveformConfidence: list[float] = []
    waveformAmplitude: list[float]
    sampleRate: int
    wordSegments: list[WordSegmentSchema]
    frequencyBands: FrequencyBandsSchema
    # Bivariate per-band relevance for the Layer-3 Relevance view (B4): magnitude
    # (bar width) + direction (side/colour). frequencyBands stays the ablation-based
    # Confidence view. None when absent (older cached results fall back to it).
    frequencyBandsRelevance: FrequencyBandsRelevanceSchema | None = None
    # L3 band x time heatmaps (replace the 3-bar chart): confidence = fakeness-gated
    # band-ablation fraction per window; relevance = honest faint per-window gradient.
    # None for older cached results / paths that don't compute them (e.g. multimodal).
    frequencyGridConfidence: FrequencyGridConfidenceSchema | None = None
    frequencyGridRelevance: FrequencyGridRelevanceSchema | None = None


class AnomalyRegionSchema(BaseModel):
    region: str
    score: float


class RegionRelevanceSchema(BaseModel):
    """Whole-clip bivariate AttnLRP score per facial region (Phase 1/2 face map).

    Aggregates BOTH channels over the per-pixel region partition, averaged across
    every frame of the clip (not a before/after shift — that is Phase 3/4):

    * ``magnitude`` — relevance magnitude (``|R_fake|+|R_real|`` ≥ 0); how much
      AttnLRP relevance the region carries across the whole clip.
    * ``direction`` — signed contrastive lean (``R_fake − R_real``); + fake-
      supporting, − real-supporting. Drives the region's seismic fill hue.
    """

    region: str
    magnitude: float
    direction: float


class Phase3ParamsSchema(BaseModel):
    crf: int
    fps: int
    noiseSigma: int


class AudioRobustnessSchema(BaseModel):
    baseConfidence: float
    degradedConfidence: float
    baseFrequencyBands: FrequencyBandsSchema
    degradedFrequencyBands: FrequencyBandsSchema
    bitrate: int


class Phase3ResultSchema(BaseModel):
    # Crop-space (224) heatmaps overlaid on the face-crop before/after players (I2).
    degradedHeatmapFrames: list[str]
    cleanHeatmapFrames: list[str] = []
    # Face-crop video URLs (served at /media) behind the heatmaps. None on the
    # face-less fallback / older cached results.
    cleanVideoUrl: str | None = None
    degradedVideoUrl: str | None = None
    # Verdict AFTER degradation (reported directly; never re-derive it from the
    # direction-less ``degradedConfidence``, which is always ≥ 0.5).
    degradedVerdict: Literal["FAKE", "REAL"]
    degradedConfidence: float
    # Clean baseline from the SAME model as the degraded pass (I1/I3), so the
    # frontend's "clean vs. degraded" comparison is like-for-like regardless of
    # the main panel's model toggle.
    baselineVerdict: Literal["FAKE", "REAL"]
    baselineConfidence: float
    params: Phase3ParamsSchema
    attentionShift: list[AttentionShiftSchema]
    # True when MediaPipe could not detect a face in the DEGRADED clip and the
    # classifier was graded on the clean-baseline face crop instead (transparency
    # flag for the robustness panel — the pipeline's face detector broke, not the
    # classifier). False on normal runs and older cached results.
    degradedFaceLost: bool = False
    audioRobustness: AudioRobustnessSchema | None = None
    # Per-frame CLEAN-crop region-partition overlay (I4 debug view): one PNG data
    # URI per frame (tan-tinted facial regions + borders + labels), aligned to the
    # clean player's frames. Empty on the face-less fallback / pre-regen cache.
    regionMaskFrames: list[str] = []


class AttentionShiftSchema(BaseModel):
    """Bivariate per-region/-band attention shift for Phase 3/4 (roadmap I4).

    Carries BOTH AttnLRP channels before and after the perturbation so the
    frontend can encode magnitude change (bar length/side) and verdict/direction
    change (colour) in one mark:

    * ``magnitudeBefore`` / ``magnitudeAfter`` — relevance magnitude
      (``|R_fake|+|R_real|`` ≥ 0); the region's attention share.
    * ``directionBefore`` / ``directionAfter`` — signed contrastive direction
      (``R_fake − R_real``); the region's fake/real verdict lean.
    """

    region: str
    magnitudeBefore: float
    magnitudeAfter: float
    directionBefore: float
    directionAfter: float


class Phase4ResultSchema(BaseModel):
    # ``perturbedFrames``/``differenceFrames`` are crop-space (224) heatmaps
    # overlaid on the face-crop before/after players (I2). ``cleanHeatmapFrames``
    # is the clean-crop heatmap for the "before" player.
    cleanHeatmapFrames: list[str] = []
    # Face-crop video URLs (served at /media) behind the heatmaps. None on the
    # face-less fallback / older cached results.
    cleanVideoUrl: str | None = None
    attackedVideoUrl: str | None = None
    perturbedFrames: list[str]
    # Verdict AFTER the attack and the confidence in THAT verdict (always ≥ 0.5).
    # ``perturbedVerdict`` must be reported explicitly — it cannot be re-derived
    # from ``perturbedConfidence`` alone, which is direction-less (a flip is
    # otherwise invisible to the frontend).
    perturbedVerdict: Literal["FAKE", "REAL"]
    perturbedConfidence: float
    differenceFrames: list[str]
    attackMethod: Literal["FGSM", "PGD"]
    epsilon: float
    attentionShift: list[AttentionShiftSchema]
    audioAttentionShift: list[AttentionShiftSchema] | None = None
    attackModalities: str | None = None
    # Clean baseline from the SAME model as the attack (I3), so the frontend can
    # show a like-for-like "clean vs. attacked" comparison independent of the main
    # panel's model toggle.
    cleanVerdict: Literal["FAKE", "REAL"]
    cleanConfidence: float
    # Per-frame CLEAN-crop region-partition overlay (I4 debug view): one PNG data
    # URI per frame (tan-tinted facial regions + borders + labels), aligned to the
    # clean player's frames. Empty on the face-less fallback / pre-regen cache.
    regionMaskFrames: list[str] = []


class CropBoxSchema(BaseModel):
    """Face crop bounding box in the original (normalised) video frame."""

    x1: int
    y1: int
    x2: int
    y2: int
    origW: int
    origH: int


class AnalysisResultSchema(BaseModel):
    clipId: str
    verdict: Literal["FAKE", "REAL"]
    confidence: float
    perFrameScores: list[float]
    # Per-16-frame-chunk timelines (A1). Confidence = the raw per-window fake
    # probability (NOT max-pooled — the verdict still is); relevance hybrid =
    # magnitude (mean |relevance|) + sign (direction of net relevance). Empty on
    # older cached results.
    perChunkConfidence: list[float] = []
    perChunkRelevanceMagnitude: list[float] = []
    perChunkRelevanceSign: list[float] = []
    heatmapFrames: list[str]
    anomalyRegions: list[AnomalyRegionSchema]
    # Whole-clip bivariate per-region relevance (Phase 1/2 face map). Empty on
    # older cached results / the face-less fallback.
    regionRelevance: list[RegionRelevanceSchema] = []
    audio: AudioAnalysisSchema | None = None
    phase3: Phase3ResultSchema | None = None
    phase4: Phase4ResultSchema | None = None
    cropBox: CropBoxSchema | None = None
    # Which model produced this result. ``fusionMode`` is set only for multimodal.
    modelMode: Literal["unimodal", "multimodal"] = "unimodal"
    fusionMode: Literal["cross_attention", "concat"] | None = None


# ── Request bodies ────────────────────────────────────────────────────────────


class RobustnessRequest(BaseModel):
    clip_id: str
    crf: int = Field(28, ge=18, le=51, description="H.264 CRF (18=lossless, 51=worst quality)")
    fps: int = Field(25, ge=5, le=30, description="Output frame rate in fps")
    noise_sigma: int = Field(0, ge=0, le=50, description="Gaussian noise σ in pixel units (0=off)")
    audio_bitrate: int | None = Field(
        None, ge=8, le=320, description="AAC audio bitrate in kbps; None = skip audio compression test"
    )
    upscale: bool = Field(False, description="Simulate TikTok/WhatsApp downscale-upscale (640×360 → 1280×720)")
    use_multimodal: bool = False
    fusion_mode: Literal["cross_attention", "concat"] = "cross_attention"


class AdversarialRequest(BaseModel):
    clip_id: str
    method: Literal["FGSM", "PGD"] = "FGSM"
    epsilon: float = Field(0.03, gt=0.0, le=0.5, description="L∞ perturbation budget")
    steps: int = Field(20, ge=1, le=100, description="Gradient-descent iterations (PGD only)")
    use_multimodal: bool = False
    attack_modalities: Literal["video", "audio", "both"] = "both"
    audio_epsilon: float = Field(
        0.03, gt=0.0, le=0.5, description="Audio L\u221e perturbation budget (multimodal only)"
    )
