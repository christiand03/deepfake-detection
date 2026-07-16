/**
 * RotationWarning — inline caution for region-based visuals (face schematic,
 * attention shift) when the clip's face is near profile.
 *
 * MediaPipe FaceMesh regresses a frontal-template mesh and degrades badly at high
 * yaw — the self-occluded far side is hallucinated, so the per-region partition
 * these visuals are built from no longer lines up with the visible face. The
 * backend flags this (AnalysisResult / Phase3 / Phase4 `faceRotationWarning`);
 * this component surfaces it as a small yellow caution so the region attribution
 * isn't read literally.
 */

const AMBER = '#e8b23c'

export function RotationWarning({ compact = false }: { compact?: boolean }) {
  return (
    <div
      role="note"
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        fontFamily: 'monospace',
        fontSize: compact ? 9 : 10,
        lineHeight: 1.4,
        letterSpacing: '0.04em',
        color: AMBER,
        border: `1px solid ${AMBER}59`,
        backgroundColor: `${AMBER}14`,
        borderRadius: 4,
        padding: compact ? '3px 6px' : '5px 8px',
      }}
    >
      <span aria-hidden style={{ fontSize: compact ? 10 : 12, lineHeight: 1 }}>
        ⚠
      </span>
      <span>Head rotated — face regions may be unreliable</span>
    </div>
  )
}
