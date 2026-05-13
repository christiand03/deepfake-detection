import type { XaiMode } from '../../types/analysis'

interface AnalysisControlsProps {
  onAnalyze: () => void
  isScanning: boolean
  isDone: boolean
  xaiMode: XaiMode
  onXaiModeChange: (mode: XaiMode) => void
  heatmapOpacity: number
  onOpacityChange: (v: number) => void
}

export function AnalysisControls({
  onAnalyze,
  isScanning,
  isDone,
  xaiMode,
  onXaiModeChange,
  heatmapOpacity,
  onOpacityChange,
}: AnalysisControlsProps) {
  const modes: { value: XaiMode; label: string }[] = [
    { value: 'rollout', label: 'Attention Rollout' },
    { value: 'lrp', label: 'AttnLRP' },
  ]

  return (
    <div className="flex flex-wrap items-center gap-3">
      {/* Analyze button */}
      <button
        onClick={onAnalyze}
        disabled={isScanning}
        style={{
          paddingInline: 20,
          paddingBlock: 9,
          borderRadius: 6,
          fontSize: 12,
          fontFamily: 'monospace',
          fontWeight: 700,
          letterSpacing: '0.12em',
          cursor: isScanning ? 'not-allowed' : 'pointer',
          border: '1px solid',
          transition: 'all 0.15s',
          ...(isScanning
            ? {
                backgroundColor: 'rgba(0,229,255,0.04)',
                borderColor: 'rgba(0,229,255,0.15)',
                color: 'rgba(0,229,255,0.4)',
              }
            : {
                backgroundColor: 'rgba(0,229,255,0.1)',
                borderColor: 'rgba(0,229,255,0.4)',
                color: '#00e5ff',
                boxShadow: '0 0 12px rgba(0,229,255,0.12)',
              }),
        }}
      >
        {isScanning ? '⏳ ANALYZING…' : isDone ? '↺ RE-ANALYZE' : '▶ ANALYZE'}
      </button>

      {/* xAI mode toggle */}
      <div
        className="flex rounded overflow-hidden"
        style={{ border: '1px solid #2a2f42' }}
      >
        {modes.map(m => {
          const active = m.value === xaiMode
          return (
            <button
              key={m.value}
              onClick={() => onXaiModeChange(m.value)}
              disabled={isScanning}
              style={{
                paddingInline: 12,
                paddingBlock: 7,
                fontSize: 11,
                fontFamily: 'monospace',
                cursor: isScanning ? 'not-allowed' : 'pointer',
                border: 'none',
                transition: 'all 0.15s',
                backgroundColor: active ? 'rgba(0,229,255,0.1)' : 'transparent',
                color: active ? '#00e5ff' : '#4d5470',
                fontWeight: active ? 600 : 400,
              }}
            >
              {m.label}
            </button>
          )
        })}
      </div>

      {/* Opacity slider — only shown when heatmap is visible */}
      {isDone && (
        <div className="flex items-center gap-2">
          <span style={{ fontSize: 11, fontFamily: 'monospace', color: '#4d5470' }}>
            OVERLAY
          </span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={heatmapOpacity}
            onChange={e => onOpacityChange(parseFloat(e.target.value))}
            style={{
              width: 80,
              accentColor: '#00e5ff',
              cursor: 'pointer',
            }}
          />
          <span
            style={{ fontSize: 11, fontFamily: 'monospace', color: '#8b92a8', width: 28 }}
          >
            {Math.round(heatmapOpacity * 100)}%
          </span>
        </div>
      )}
    </div>
  )
}
