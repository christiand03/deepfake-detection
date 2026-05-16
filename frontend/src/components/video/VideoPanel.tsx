/**
 * VideoPanel — left column of the main layout.
 *
 * Owns the DemoSelector, VideoAnalysisPlayer, FrameTimeline and AnalysisControls.
 * Also exposes the current analysis state + clip upward so VerdictPanel and
 * AudioLayers can consume it via props passed through App.tsx.
 */

import { useEffect, useState } from 'react'
import { DemoSelector } from './DemoSelector'
import { VideoAnalysisPlayer } from './VideoAnalysisPlayer'
import { FrameTimeline } from './FrameTimeline'
import { AnalysisControls } from './AnalysisControls'
import { useAnalysis } from '../../hooks/useAnalysis'
import { useVideoSync } from '../../hooks/useVideoSync'
import { fetchClips } from '../../api/client'
import { useErrorToast } from '../../context/ErrorToastContext'
import type { AnalysisResult, ClipMeta } from '../../types/analysis'

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
  const xaiMode = 'lrp'
  const [clips, setClips] = useState<ClipMeta[]>([])
  const [selectedId, setSelectedId] = useState<string>('')
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.55)

  const { state, analyze } = useAnalysis()
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

  function handleSelect(id: string) {
    setSelectedId(id)
  }

  function handleAnalyze() {
    if (selectedId) analyze(selectedId, xaiMode)
  }

  if (!selectedClip) return null

  return (
    <div className="flex flex-col gap-4">
      <DemoSelector
        clips={clips}
        selectedId={selectedId}
        onSelect={handleSelect}
        disabled={isScanning}
      />

      <VideoAnalysisPlayer
        ref={videoRef}
        clip={selectedClip}
        heatmapFrames={result?.heatmapFrames ?? null}
        frameIndex={frameIndex}
        isScanning={isScanning}
        heatmapOpacity={heatmapOpacity}
      />

      {result && (
        <FrameTimeline
          scores={result.perFrameScores}
          currentFrame={frameIndex}
          verdict={result.verdict}
        />
      )}

      <AnalysisControls
        onAnalyze={handleAnalyze}
        isScanning={isScanning}
        isDone={isDone}
        heatmapOpacity={heatmapOpacity}
        onOpacityChange={setHeatmapOpacity}
      />

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
