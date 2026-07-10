/**
 * FaceSchematic — whole-clip facial-region relevance map (Phase 1/2).
 *
 * A crisp front-facing vector face partitioned into the six AttnLRP regions
 * (Forehead, Left/Right Eye, Nose, Mouth, Jaw). Each region is FILLED with the
 * same bivariate seismic encoding used everywhere else in the app:
 *   • fill ALPHA   = whole-clip relevance magnitude, normalised to the strongest
 *     region (so the most-attended region reads at full strength).
 *   • fill HUE     = signed verdict lean (red = fake-supporting, blue = real).
 * The single most-attended region gets a bright outline so "where the model
 * looked most across the clip" is legible at a glance. Hovering a region
 * highlights it and shows the exact magnitude/direction, matching the other
 * hover popups (ChunkTimelines, AttentionShiftTable).
 *
 * This is a whole-clip AGGREGATE — not a before/after shift (that is Phase 3/4's
 * AttentionShiftTable). Fed by AnalysisResult.regionRelevance.
 */

import { useRef, useState } from 'react'

import { bivariateRgba } from '../../lib/seismicColormap'
import type { RegionRelevance } from '../../types/analysis'

// ── Region geometry (viewBox 0 0 200 250) ────────────────────────────────────
// Each region is an independently fillable + hoverable closed path. Shapes are
// anatomically placed on a symmetric front face; the label anchor is the point
// the hover tooltip is drawn from.
interface RegionShape {
  /** SVG path 'd' for the filled region. */
  d: string
  /** Centroid-ish anchor in viewBox units (for the top-region label). */
  anchor: { x: number; y: number }
}

const REGION_SHAPES: Record<string, RegionShape> = {
  Forehead: {
    d: 'M46,96 C48,50 70,26 100,26 C130,26 152,50 154,96 C120,84 80,84 46,96 Z',
    anchor: { x: 100, y: 58 },
  },
  'Left Eye': {
    d: 'M60,108 Q74,97 88,108 Q74,118 60,108 Z',
    anchor: { x: 74, y: 108 },
  },
  'Right Eye': {
    d: 'M112,108 Q126,97 140,108 Q126,118 112,108 Z',
    anchor: { x: 126, y: 108 },
  },
  Nose: {
    d: 'M97,112 C96,128 92,142 92,150 Q100,159 108,150 C108,142 104,128 103,112 Q100,108 97,112 Z',
    anchor: { x: 100, y: 134 },
  },
  Mouth: {
    d: 'M78,176 Q100,167 122,176 Q100,190 78,176 Z',
    anchor: { x: 100, y: 178 },
  },
  Jaw: {
    d: 'M44,132 C48,180 74,220 100,228 C126,220 152,180 156,132 C150,168 130,196 100,200 C70,196 50,168 44,132 Z',
    anchor: { x: 100, y: 210 },
  },
}

// Face silhouette drawn behind the regions for context.
const FACE_OUTLINE =
  'M100,22 C136,22 158,54 158,108 C158,172 132,222 100,230 ' +
  'C68,222 42,172 42,108 C42,54 64,22 100,22 Z'

// Bivariate fill tuning for the face: a handful of large regions (vs. per-pixel),
// so keep the hue linear (decisive leans stay coloured, only near-neutral fades)
// and lift low alphas a touch so faint regions still read as tinted, not black.
const FILL_OPTS = { maxAlpha: 0.92, alphaGamma: 0.6, dirGamma: 1.0, dirGain: 1.35, dirCap: 0.85 }

function fmt(v: number): string {
  return v.toFixed(3)
}

function signed(v: number): string {
  return `${v >= 0 ? '+' : ''}${v.toFixed(3)}`
}

export function FaceSchematic({ regions }: { regions: RegionRelevance[] }) {
  const rootRef = useRef<HTMLDivElement>(null)
  const [hover, setHover] = useState<{ region: string; x: number; y: number; w: number } | null>(
    null,
  )

  const byName = new Map(regions.map(r => [r.region, r]))
  const maxMag = Math.max(1e-6, ...regions.map(r => Math.abs(r.magnitude)))
  // Total attention across all regions — the tooltip reports each region's share
  // of this (sums to 100%), while the fill alpha stays peak-normalised so the
  // strongest region reads at full strength.
  const totalMag = Math.max(
    1e-6,
    regions.reduce((sum, r) => sum + Math.abs(r.magnitude), 0),
  )
  // Most-attended region across the whole clip (drives the highlight + caption).
  const top = regions.reduce<RegionRelevance | null>(
    (best, r) => (best === null || Math.abs(r.magnitude) > Math.abs(best.magnitude) ? r : best),
    null,
  )

  function onMove(e: React.MouseEvent, region: string) {
    const rect = rootRef.current?.getBoundingClientRect()
    if (!rect) return
    setHover({ region, x: e.clientX - rect.left, y: e.clientY - rect.top, w: rect.width })
  }

  const hovered = hover ? byName.get(hover.region) : undefined

  return (
    <div
      style={{
        display: 'flex',
        height: '100%',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 32,
        padding: '18px 24px',
        boxSizing: 'border-box',
      }}
    >
      {/* ── Face ───────────────────────────────────────────────────────────── */}
      <div ref={rootRef} style={{ position: 'relative', height: '100%', aspectRatio: '200 / 250' }}>
        <svg
          viewBox="0 0 200 250"
          shapeRendering="geometricPrecision"
          style={{ height: '100%', width: '100%', display: 'block', overflow: 'visible' }}
        >
          {/* Silhouette for context */}
          <path
            d={FACE_OUTLINE}
            fill="#0f1219"
            stroke="#2a2f42"
            strokeWidth={1.5}
            strokeLinejoin="round"
          />

          {Object.entries(REGION_SHAPES).map(([name, shape]) => {
            const r = byName.get(name)
            const magNorm = r ? Math.abs(r.magnitude) / maxMag : 0
            const fill = r ? bivariateRgba(magNorm, r.direction, FILL_OPTS) : 'transparent'
            const isHovered = hover?.region === name
            const isTop = top?.region === name
            const stroke = isHovered ? '#ffffff' : isTop ? 'rgba(232,234,240,0.85)' : 'rgba(180,190,214,0.35)'
            return (
              <path
                key={name}
                d={shape.d}
                fill={fill}
                stroke={stroke}
                strokeWidth={isHovered ? 2 : isTop ? 1.6 : 1}
                strokeLinejoin="round"
                style={{
                  cursor: 'pointer',
                  transition: 'stroke 0.12s, stroke-width 0.12s',
                  filter: isHovered
                    ? 'drop-shadow(0 0 5px rgba(255,255,255,0.55))'
                    : isTop
                      ? 'drop-shadow(0 0 4px rgba(200,210,235,0.35))'
                      : 'none',
                }}
                onMouseEnter={e => onMove(e, name)}
                onMouseMove={e => onMove(e, name)}
                onMouseLeave={() => setHover(cur => (cur?.region === name ? null : cur))}
              />
            )
          })}
        </svg>

        {/* Hover tooltip (mouse-following, matches the other popups) */}
        {hover && hovered && (
          <div
            style={{
              position: 'absolute',
              left: Math.min(hover.x + 10, hover.w - 6),
              top: hover.y,
              transform: 'translateY(-50%)',
              padding: '5px 8px',
              whiteSpace: 'nowrap',
              fontSize: 10,
              lineHeight: 1.5,
              fontFamily: 'monospace',
              color: '#e8eaf0',
              backgroundColor: '#1b1f2e',
              border: '1px solid #2a2f42',
              borderRadius: 5,
              pointerEvents: 'none',
              zIndex: 4,
            }}
          >
            <div style={{ color: '#a0a8c0', marginBottom: 2 }}>{hovered.region}</div>
            <div>
              <span style={{ color: '#8b92a8' }}>relevance </span>
              {fmt(Math.abs(hovered.magnitude))}{' '}
              <span style={{ color: '#8b92a8' }}>
                ({Math.round((Math.abs(hovered.magnitude) / totalMag) * 100)}% of total)
              </span>
            </div>
            <div>
              <span style={{ color: '#8b92a8' }}>lean </span>
              <span style={{ color: hovered.direction >= 0 ? '#ff7070' : '#5e91ee' }}>
                {hovered.direction >= 0 ? 'fake' : 'real'} {signed(hovered.direction)}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* ── Caption + legend ───────────────────────────────────────────────── */}
      <div
        style={{
          flexShrink: 0,
          maxWidth: 240,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          gap: 14,
        }}
      >
        <div
          style={{
            fontSize: 10,
            fontFamily: 'monospace',
            color: '#4d5470',
            letterSpacing: '0.14em',
          }}
        >
          REGION RELEVANCE · WHOLE CLIP
        </div>
        {top && (
          <div style={{ fontFamily: 'monospace' }}>
            <div style={{ fontSize: 9, color: '#4d5470', letterSpacing: '0.1em', marginBottom: 3 }}>
              MOST ATTENDED
            </div>
            <div
              style={{
                fontSize: 22,
                fontWeight: 600,
                color: top.direction >= 0 ? '#ff7070' : '#5e91ee',
              }}
            >
              {top.region}
            </div>
            <div style={{ fontSize: 10, color: '#8b92a8', marginTop: 2 }}>
              {Math.round((Math.abs(top.magnitude) / totalMag) * 100)}% of total attention
            </div>
          </div>
        )}
        <div style={{ fontSize: 9, fontFamily: 'monospace', color: '#4d5470', lineHeight: 1.9 }}>
          <div>fill = mean AttnLRP relevance</div>
          <div>
            <span style={{ color: '#5e91ee' }}>■ real-lean</span>{' '}
            <span style={{ color: '#ff7070', marginLeft: 8 }}>■ fake-lean</span>
          </div>
          <div style={{ color: '#3a4059' }}>hover over a region for values</div>
        </div>
      </div>
    </div>
  )
}
