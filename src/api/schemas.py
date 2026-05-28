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


class WordSegmentSchema(BaseModel):
    word: str
    start: float
    end: float
    relevance: float


class FrequencyBandsSchema(BaseModel):
    low: float
    mid: float
    high: float


class AudioAnalysisSchema(BaseModel):
    verdict: Literal["FAKE", "REAL"]
    confidence: float
    waveformRelevance: list[float]
    waveformAmplitude: list[float]
    sampleRate: int
    wordSegments: list[WordSegmentSchema]
    frequencyBands: FrequencyBandsSchema


class AnomalyRegionSchema(BaseModel):
    region: str
    score: float


class Phase3ParamsSchema(BaseModel):
    crf: int
    fps: int
    noiseSigma: int


class Phase3ResultSchema(BaseModel):
    degradedHeatmapFrames: list[str]
    degradedConfidence: float
    params: Phase3ParamsSchema
    attentionShift: list[AttentionShiftSchema]


class AttentionShiftSchema(BaseModel):
    region: str
    before: float
    after: float


class Phase4ResultSchema(BaseModel):
    perturbedFrames: list[str]
    perturbedConfidence: float
    differenceFrames: list[str]
    attackMethod: Literal["FGSM", "PGD"]
    epsilon: float
    attentionShift: list[AttentionShiftSchema]


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
    heatmapFrames: list[str]
    anomalyRegions: list[AnomalyRegionSchema]
    audio: AudioAnalysisSchema | None = None
    phase3: Phase3ResultSchema | None = None
    phase4: Phase4ResultSchema | None = None
    cropBox: CropBoxSchema | None = None


# ── Request bodies ────────────────────────────────────────────────────────────


class RobustnessRequest(BaseModel):
    clip_id: str
    crf: int = Field(28, ge=18, le=51, description="H.264 CRF (18=lossless, 51=worst quality)")
    fps: int = Field(25, ge=5, le=30, description="Output frame rate in fps")
    noise_sigma: int = Field(0, ge=0, le=50, description="Gaussian noise σ in pixel units (0=off)")


class AdversarialRequest(BaseModel):
    clip_id: str
    method: Literal["FGSM", "PGD"] = "FGSM"
    epsilon: float = Field(0.03, gt=0.0, le=0.5, description="L∞ perturbation budget")
    steps: int = Field(20, ge=1, le=100, description="Gradient-descent iterations (PGD only)")
