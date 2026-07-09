/**
 * RegionToggle — checkbox that switches the Phase-3/4 crop player from the
 * AttnLRP heatmap to the landmark face-region overlay (I4 debug view).
 *
 * Rendered by the panels (below the video-opacity slider, above the attention-
 * shift visual) so it sits next to the thing it explains. Hidden when the clip
 * has no landmark region boxes (face-less fallback / pre-regen cache).
 */

export function RegionToggle({
  checked,
  onChange,
  visible,
}: {
  checked: boolean
  onChange: (v: boolean) => void
  visible: boolean
}) {
  if (!visible) return null
  return (
    <div style={{ display: 'flex', justifyContent: 'flex-end', margin: '2px 0 6px' }}>
      <label
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          fontSize: 9,
          fontFamily: 'monospace',
          color: checked ? '#d8bf94' : '#4d5470',
          letterSpacing: '0.08em',
          cursor: 'pointer',
          userSelect: 'none',
        }}
      >
        <input
          type="checkbox"
          checked={checked}
          onChange={e => onChange(e.target.checked)}
          style={{ accentColor: '#d8bf94', cursor: 'pointer' }}
        />
        SHOW FACE REGIONS
      </label>
    </div>
  )
}
