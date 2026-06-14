import type { FusionMode, ModelMode } from '../../types/analysis'

interface AnalysisControlsProps {
  onAnalyze: () => void
  isScanning: boolean
  isDone: boolean
  heatmapOpacity: number
  onOpacityChange: (v: number) => void
  modelMode: ModelMode
  onModelModeChange: (mode: ModelMode) => void
  fusionMode: FusionMode
  onFusionModeChange: (mode: FusionMode) => void
  /** True when the selected clip has no audio (multimodal needs both modalities). */
  multimodalDisabled: boolean
}

// Accent colours: cyan = unimodal/default, purple = multimodal.
const CYAN = '#00e5ff'
const PURPLE = '#a855f7'

// ── Generic segmented two-option toggle ──────────────────────────────────────

function SegmentedToggle<T extends string>({
  options,
  value,
  onChange,
  accent,
  disabled,
  disabledOptions,
  disabledTitle,
}: {
  options: { value: T; label: string }[]
  value: T
  onChange: (v: T) => void
  accent: string
  disabled?: boolean
  disabledOptions?: T[]
  disabledTitle?: string
}) {
  return (
    <div style={{ display: 'flex', gap: 4 }}>
      {options.map(opt => {
        const active = value === opt.value
        const optDisabled = disabled || disabledOptions?.includes(opt.value)
        return (
          <button
            key={opt.value}
            onClick={() => !optDisabled && onChange(opt.value)}
            disabled={optDisabled}
            title={optDisabled ? disabledTitle : undefined}
            style={{
              paddingInline: 12,
              paddingBlock: 7,
              borderRadius: 5,
              fontSize: 10,
              fontFamily: 'monospace',
              fontWeight: 700,
              letterSpacing: '0.1em',
              cursor: optDisabled ? 'not-allowed' : 'pointer',
              border: '1px solid',
              transition: 'all 0.15s',
              opacity: optDisabled ? 0.4 : 1,
              borderColor: active ? accent : '#2a2f42',
              backgroundColor: active ? `${accent}1f` : '#0d0f14',
              color: active ? accent : '#4d5470',
            }}
          >
            {active ? '● ' : '○ '}
            {opt.label}
          </button>
        )
      })}
    </div>
  )
}

export function AnalysisControls({
  onAnalyze,
  isScanning,
  isDone,
  heatmapOpacity,
  onOpacityChange,
  modelMode,
  onModelModeChange,
  fusionMode,
  onFusionModeChange,
  multimodalDisabled,
}: AnalysisControlsProps) {
  const isMultimodal = modelMode === 'multimodal'
  const accent = isMultimodal ? PURPLE : CYAN

  return (
    <div className="flex flex-col gap-3">
      {/* ── Model-mode + fusion selection ──────────────────────────────────── */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <div className="flex items-center gap-2">
          <span
            style={{ fontSize: 9, fontFamily: 'monospace', letterSpacing: '0.16em', color: '#4d5470' }}
          >
            MODEL
          </span>
          <SegmentedToggle<ModelMode>
            options={[
              { value: 'unimodal', label: 'UNIMODAL' },
              { value: 'multimodal', label: 'MULTIMODAL' },
            ]}
            value={modelMode}
            onChange={onModelModeChange}
            accent={accent}
            disabled={isScanning}
            disabledOptions={multimodalDisabled ? ['multimodal'] : undefined}
            disabledTitle="Requires an audio track"
          />
        </div>

        {isMultimodal && (
          <div className="flex items-center gap-2">
            <span
              style={{ fontSize: 9, fontFamily: 'monospace', letterSpacing: '0.16em', color: '#4d5470' }}
            >
              FUSION
            </span>
            <SegmentedToggle<FusionMode>
              options={[
                { value: 'cross_attention', label: 'CROSS-ATTN' },
                { value: 'concat', label: 'CONCAT' },
              ]}
              value={fusionMode}
              onChange={onFusionModeChange}
              accent={PURPLE}
              disabled={isScanning}
            />
          </div>
        )}
      </div>

      {/* ── Analyze button + overlay slider ────────────────────────────────── */}
      <div className="flex flex-wrap items-center gap-3">
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
                  backgroundColor: `${accent}0a`,
                  borderColor: `${accent}26`,
                  color: `${accent}66`,
                }
              : {
                  backgroundColor: `${accent}1a`,
                  borderColor: `${accent}66`,
                  color: accent,
                  boxShadow: `0 0 12px ${accent}1f`,
                }),
          }}
        >
          {isScanning ? '⏳ ANALYZING…' : isDone ? '↺ RE-ANALYZE' : '▶ ANALYZE'}
        </button>

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
                accentColor: accent,
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
    </div>
  )
}
