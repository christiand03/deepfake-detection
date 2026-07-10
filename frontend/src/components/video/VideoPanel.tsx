/**
 * VideoPanel — left column of the main layout.
 *
 * Owns the DemoSelector, VideoAnalysisPlayer, FrameTimeline and AnalysisControls.
 * Also exposes the current analysis state + clip upward so VerdictPanel and
 * AudioLayers can consume it via props passed through App.tsx.
 */

import { useEffect, useState } from 'react'
import { ClipSelector } from './ClipSelector'
import { VideoAnalysisPlayer } from './VideoAnalysisPlayer'
import { ChunkTimelines } from './ChunkTimelines'
import { RegionFacePanel } from './RegionFacePanel'
import { AnalysisControls } from './AnalysisControls'
import { useAnalysis } from '../../hooks/useAnalysis'
import { useVideoSync } from '../../hooks/useVideoSync'
import { fetchClips } from '../../api/client'
import { useErrorToast } from '../../context/ErrorToastContext'
import type { AnalysisResult, ClipMeta, FusionMode, ModelMode } from '../../types/analysis'

interface VideoPanelProps {
  videoRef: React.RefObject<HTMLVideoElement | null>
  onResult?: (result: AnalysisResult | null) => void
  onClipChange?: (clip: ClipMeta) => void
  onScanningChange?: (scanning: boolean) => void
}

export function VideoPanel({
  videoRef,
  onResult,
  onClipChange,
  onScanningChange,
}: VideoPanelProps) {
  const [clips, setClips] = useState<ClipMeta[]>([])
  const [selectedId, setSelectedId] = useState<string>('')
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.85)
  const [modelMode, setModelMode] = useState<ModelMode>('unimodal')
  const [fusionMode, setFusionMode] = useState<FusionMode>('cross_attention')

  const { state, analyze, reset } = useAnalysis()
  const { showError } = useErrorToast()
  const isScanning = state.status === 'scanning'
  const isDone = state.status === 'done'
  const result = isDone ? state.result : null

  // Show a toast whenever analysis enters error state
  useEffect(() => {
    if (state.status === 'error') {
      showError(`Analysis failed: ${state.message}`)
    }
  }, [state, showError])

  // For FrameTimeline playhead sync (videoRef is passed in from App.tsx
  // and shared with AudioLayers for timeupdate subscriptions)
  const selectedClip = clips.find(c => c.id === selectedId)
  const totalFrames = result?.perFrameScores.length ?? 1
  const frameIndex = useVideoSync(videoRef, selectedClip?.fps ?? 25, totalFrames)

  // Load clip list on mount
  useEffect(() => {
    fetchClips()
      .then(list => {
        setClips(list)
        if (list.length > 0) setSelectedId(list[0].id)
      })
      .catch((err: unknown) => {
        showError(`Failed to load clips: ${err instanceof Error ? err.message : String(err)}`)
      })
  }, [showError])

  // Notify parent when scanning state changes
  useEffect(() => {
    onScanningChange?.(isScanning)
  }, [isScanning, onScanningChange])

  // Notify parent when result changes
  useEffect(() => {
    onResult?.(result)
  }, [result, onResult])

  // Notify parent when clip changes
  useEffect(() => {
    if (selectedClip) onClipChange?.(selectedClip)
  }, [selectedClip, onClipChange])

  // Multimodal requires an audio track; force unimodal for audio-less clips.
  const multimodalDisabled = !selectedClip?.hasAudio

  function handleSelect(id: string) {
    setSelectedId(id)
    reset()
    // A clip without audio can't run multimodal — fall back to unimodal.
    if (!clips.find(c => c.id === id)?.hasAudio) setModelMode('unimodal')
  }

  function handleModelModeChange(mode: ModelMode) {
    setModelMode(mode)
    reset() // stale result no longer matches the selected mode
  }

  function handleFusionModeChange(mode: FusionMode) {
    setFusionMode(mode)
    reset()
  }

  function handleAnalyze() {
    if (!selectedId) return
    analyze(selectedId, {
      useMultimodal: modelMode === 'multimodal' && !multimodalDisabled,
      fusionMode,
    })
  }

  if (!selectedClip) return null

  return (
    <div className="flex flex-col gap-4">
      <ClipSelector
        clips={clips}
        selectedId={selectedId}
        onSelect={handleSelect}
        disabled={isScanning}
      />

      <div style={{ position: 'relative' }}>
        <RegionFacePanel regions={result?.regionRelevance ?? []}>
          <VideoAnalysisPlayer
            ref={videoRef}
            clip={selectedClip}
            heatmapFrames={result?.heatmapFrames ?? null}
            frameIndex={frameIndex}
            isScanning={isScanning}
            heatmapOpacity={heatmapOpacity}
          />
        </RegionFacePanel>

        {/* Analysis controls overlay — top-right corner of the player. */}
        <div style={{ position: 'absolute', top: 10, right: 10, zIndex: 7 }}>
          <AnalysisControls
            onAnalyze={handleAnalyze}
            isScanning={isScanning}
            isDone={isDone}
            heatmapOpacity={heatmapOpacity}
            onOpacityChange={setHeatmapOpacity}
            modelMode={modelMode}
            onModelModeChange={handleModelModeChange}
            fusionMode={fusionMode}
            onFusionModeChange={handleFusionModeChange}
            multimodalDisabled={multimodalDisabled}
          />
        </div>
      </div>

      {result && (
        <ChunkTimelines
          confidence={result.perChunkConfidence}
          relevanceMagnitude={result.perChunkRelevanceMagnitude}
          relevanceSign={result.perChunkRelevanceSign}
          currentFrame={frameIndex}
          totalFrames={result.perFrameScores.length}
        />
      )}

      {state.status === 'error' && (
        <div
          className="text-xs font-mono px-3 py-2 rounded"
          style={{
            backgroundColor: 'rgba(239,68,68,0.08)',
            border: '1px solid rgba(239,68,68,0.2)',
            color: '#f87171',
          }}
        >
          Error: {state.message}
        </div>
      )}
    </div>
  )
}
