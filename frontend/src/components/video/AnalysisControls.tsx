interface AnalysisControlsProps {
  onAnalyze: () => void
  isScanning: boolean
  isDone: boolean
  heatmapOpacity: number
  onOpacityChange: (v: number) => void
}

export function AnalysisControls({
  onAnalyze,
  isScanning,
  isDone,
  heatmapOpacity,
  onOpacityChange,
}: AnalysisControlsProps) {
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

      {/* xAI method fixed to AttnLRP — no toggle needed */}

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
