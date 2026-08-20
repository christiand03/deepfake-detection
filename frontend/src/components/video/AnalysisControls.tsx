import type { FusionMode, HeatmapMethod, ModelMode } from '../../types/analysis'

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
  /** Which method renders the PLAYER OVERLAY only — see heatmapMethod docs. */
  heatmapMethod: HeatmapMethod
  onHeatmapMethodChange: (m: HeatmapMethod) => void
  /** True while an alternative method's frames are being fetched. */
  heatmapLoading: boolean
}

// Accent colours: cyan = unimodal/default, purple = multimodal.
const CYAN = '#00e5ff'
const PURPLE = '#a855f7'
// Amber for the heatmap-method switch — deliberately neither of the model-mode colours,
// so it reads as a different axis of control and not as another model choice.
const AMBER = '#f59e0b'

// ── Generic segmented toggle ─────────────────────────────────────────────────

function SegmentedToggle<T extends string>({
  options,
  value,
  onChange,
  accent,
  disabled,
  disabledOptions,
  disabledTitle,
  vertical,
}: {
  options: { value: T; label: string }[]
  value: T
  onChange: (v: T) => void
  accent: string
  disabled?: boolean
  disabledOptions?: T[]
  disabledTitle?: string
  /** Stack the options instead of placing them in a row. The panel floats over the
   *  video, so a three-option row would widen it across the face; stacking keeps it
   *  narrow at the cost of a little height. */
  vertical?: boolean
}) {
  return (
    <div
      style={{
        display: 'flex',
        gap: 4,
        flexDirection: vertical ? 'column' : 'row',
        alignItems: vertical ? 'stretch' : 'center',
      }}
    >
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
              textAlign: vertical ? 'left' : 'center',
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
  heatmapMethod,
  onHeatmapMethodChange,
  heatmapLoading,
}: AnalysisControlsProps) {
  const isMultimodal = modelMode === 'multimodal'
  const accent = isMultimodal ? PURPLE : CYAN

  const labelStyle = {
    fontSize: 9,
    fontFamily: 'monospace',
    letterSpacing: '0.16em',
    color: '#8b92a8',
  } as const

  return (
    <div
      className="flex flex-col items-end gap-2.5"
      style={{
        padding: '10px 12px',
        borderRadius: 8,
        backgroundColor: 'rgba(13,15,20,0.82)',
        backdropFilter: 'blur(6px)',
        border: '1px solid #2a2f42',
        boxShadow: '0 8px 24px rgba(0,0,0,0.45)',
      }}
    >
      {/* ── Model-mode selection ───────────────────────────────────────────── */}
      <div className="flex items-center gap-2">
        <span style={labelStyle}>MODEL</span>
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

      {/* ── Fusion variant (multimodal only) ───────────────────────────────── */}
      {isMultimodal && (
        <div className="flex items-center gap-2">
          <span style={labelStyle}>VARIANT</span>
          <SegmentedToggle<FusionMode>
            options={[
              { value: 'cross_attention', label: 'FUSION' },
              { value: 'concat', label: 'CONCATENATION' },
            ]}
            value={fusionMode}
            onChange={onFusionModeChange}
            accent={PURPLE}
            disabled={isScanning}
          />
        </div>
      )}

      {/* ── Analyze button ─────────────────────────────────────────────────── */}
      <button
        onClick={onAnalyze}
        disabled={isScanning}
        style={{
          width: '100%',
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

      {/* ── Heatmap method — swaps the PLAYER OVERLAY ONLY ──────────────────
          Three stages, not two, on purpose (docs/chefer_ablation.md §5): going from
          stage 1 to 2 isolates the effect of dropping the direction axis, and 2 to 3
          the effect of the method. A two-way switch would change both at once and no
          screenshot could separate them. */}
      {isDone && (
        <div className="flex flex-col items-stretch gap-1" style={{ width: '100%' }}>
          <div className="flex items-center justify-end gap-2">
            <span style={labelStyle}>HEATMAP-METHODE</span>
            {heatmapLoading && (
              <span style={{ fontSize: 10, fontFamily: 'monospace', color: AMBER }}>⏳</span>
            )}
          </div>
          <SegmentedToggle<HeatmapMethod>
            options={[
              { value: 'bivariate', label: 'BIVARIATE LRP' },
              { value: 'lrp_magnitude', label: 'LRP MAGNITUDE' },
              { value: 'chefer', label: 'CHEFER ET AL.' },
            ]}
            value={heatmapMethod}
            onChange={onHeatmapMethodChange}
            accent={AMBER}
            disabled={isScanning || heatmapLoading}
            vertical
          />
          {/* Always visible, not a tooltip: a screenshot for the Beleg has to carry the
              scope of the switch with it. */}
          <span
            style={{
              fontSize: 9,
              fontFamily: 'monospace',
              color: '#6b7280',
              maxWidth: 168,
              textAlign: 'right',
              lineHeight: 1.4,
              alignSelf: 'flex-end',
            }}
          >
            Nur das Overlay im Player. Verdict, Regionen und Timelines bleiben
            Bivariate-LRP.
          </span>
        </div>
      )}

      {/* ── Overlay slider — only shown when heatmap is visible ─────────────── */}
      {isDone && (
        <div className="flex items-center gap-2">
          <span style={labelStyle}>OVERLAY</span>
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
  )
}
