/**
 * Reusable presentation widgets for explanation content.
 *
 * These are the "looks matter as much as the text" pieces: a seismic colour-scale
 * legend that matches the real colormap, callouts, boxed formulas, a chunk-strip
 * that visualises the 0.64 s temporal grid, key/value rows and small pills. They
 * are imported by the `content/*.tsx` files so every explanation reads as one
 * system.
 *
 * Palette matches the app (dark panels, monospace, cyan accent).
 */

import type { ReactNode } from 'react'
import { relevanceToRgb } from '../../lib/seismicColormap'

const MONO = 'monospace'

// ── ColorScaleLegend ──────────────────────────────────────────────────────────

/**
 * Horizontal seismic legend (fake ← neutral → real), sampled from the SAME
 * `relevanceToRgb` ramp the charts use, so the key is faithful. Optionally marks
 * the neutral Dead-Zone in the middle where near-zero values fade to transparent.
 */
export function ColorScaleLegend({
  leftLabel = 'fake-stützend',
  rightLabel = 'real-stützend',
  midLabel = 'neutral',
  deadZone = true,
  height = 14,
}: {
  leftLabel?: string
  rightLabel?: string
  midLabel?: string
  deadZone?: boolean
  height?: number
}) {
  // Sample the ramp at N stops. Direction axis runs +1 (fake, red) → -1 (real,
  // blue), so the left end is +1.
  const stops = 24
  const gradient = Array.from({ length: stops + 1 }, (_, i) => {
    const t = i / stops // 0..1 left→right
    const value = 1 - 2 * t // +1 → -1
    const [r, g, b] = relevanceToRgb(value)
    return `rgb(${r},${g},${b}) ${(t * 100).toFixed(1)}%`
  }).join(', ')

  return (
    <div style={{ margin: '2px 0' }}>
      <div style={{ position: 'relative' }}>
        <div
          style={{
            height,
            borderRadius: 4,
            background: `linear-gradient(90deg, ${gradient})`,
            border: '1px solid #2a2f42',
          }}
        />
        {deadZone && (
          // Dead-zone marker: the central band where |value| is small and colour
          // washes to neutral / alpha drops toward 0.
          <div
            style={{
              position: 'absolute',
              top: -2,
              bottom: -2,
              left: '42%',
              width: '16%',
              borderLeft: '1px dashed rgba(139,146,168,0.6)',
              borderRight: '1px dashed rgba(139,146,168,0.6)',
              pointerEvents: 'none',
            }}
          />
        )}
      </div>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          marginTop: 3,
          fontSize: 9,
          fontFamily: MONO,
          color: '#8b92a8',
        }}
      >
        <span style={{ color: '#ff7070' }}>◀ {leftLabel}</span>
        {deadZone && <span style={{ color: '#4d5470' }}>{midLabel} · Dead-Zone</span>}
        <span style={{ color: '#5e91ee' }}>{rightLabel} ▶</span>
      </div>
    </div>
  )
}

// ── Callout ───────────────────────────────────────────────────────────────────

type CalloutVariant = 'info' | 'warn' | 'tip'

const CALLOUT: Record<CalloutVariant, { color: string; bg: string; glyph: string }> = {
  info: { color: '#00e5ff', bg: 'rgba(0,229,255,0.07)', glyph: 'ℹ' },
  warn: { color: '#f59e0b', bg: 'rgba(245,158,11,0.08)', glyph: '⚠' },
  tip: { color: '#22c55e', bg: 'rgba(34,197,94,0.08)', glyph: '✓' },
}

/** Coloured callout box for a pitfall (warn), a note (info) or a tip. */
export function Callout({
  variant = 'info',
  title,
  children,
}: {
  variant?: CalloutVariant
  title?: string
  children: ReactNode
}) {
  const c = CALLOUT[variant]
  return (
    <div
      style={{
        display: 'flex',
        gap: 9,
        alignItems: 'flex-start',
        padding: '9px 12px',
        borderRadius: 7,
        backgroundColor: c.bg,
        border: `1px solid ${c.color}44`,
        borderLeft: `2px solid ${c.color}`,
        margin: '6px 0',
      }}
    >
      <span style={{ color: c.color, fontSize: 12, lineHeight: '18px', flexShrink: 0 }}>
        {c.glyph}
      </span>
      <div style={{ fontSize: 11.5, lineHeight: 1.6, color: '#c9cede' }}>
        {title && (
          <div style={{ color: c.color, fontWeight: 700, marginBottom: 2, fontFamily: MONO, fontSize: 10.5 }}>
            {title}
          </div>
        )}
        {children}
      </div>
    </div>
  )
}

// ── Formula ─────────────────────────────────────────────────────────────────

/** Monospace boxed formula / pseudo-code line(s). */
export function Formula({ children }: { children: ReactNode }) {
  return (
    <pre
      style={{
        margin: '6px 0',
        padding: '9px 12px',
        borderRadius: 7,
        backgroundColor: '#0b0d12',
        border: '1px solid #2a2f42',
        color: '#9fe8ff',
        fontFamily: MONO,
        fontSize: 11,
        lineHeight: 1.6,
        overflowX: 'auto',
        whiteSpace: 'pre',
      }}
    >
      {children}
    </pre>
  )
}

// ── Chip / inline highlight ─────────────────────────────────────────────────

/** Inline coloured pill for a domain term or a value. */
export function Chip({
  children,
  color = '#00e5ff',
}: {
  children: ReactNode
  color?: string
}) {
  return (
    <span
      style={{
        display: 'inline-block',
        padding: '1px 6px',
        borderRadius: 4,
        backgroundColor: `${color}1f`,
        border: `1px solid ${color}44`,
        color,
        fontFamily: MONO,
        fontSize: 10.5,
        fontWeight: 600,
        lineHeight: 1.5,
        whiteSpace: 'nowrap',
      }}
    >
      {children}
    </span>
  )
}

// ── CvRBadge (Confidence vs. Relevance) ──────────────────────────────────────

/** Header badge placing a visual on the confidence/relevance axis (F1 core). */
export function CvRBadge({ cvr }: { cvr: 'confidence' | 'relevance' | 'both' | 'neither' }) {
  const map = {
    confidence: { label: 'CONFIDENCE', sub: 'WAS / WIE-fake', color: '#00e5ff' },
    relevance: { label: 'RELEVANCE', sub: 'WO / WARUM', color: '#a855f7' },
    both: { label: 'CONFIDENCE + RELEVANCE', sub: 'umschaltbar', color: '#22c55e' },
    neither: { label: 'META', sub: 'Zustands-Hinweis', color: '#8b92a8' },
  }[cvr]
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'baseline',
        gap: 6,
        padding: '3px 9px',
        borderRadius: 5,
        backgroundColor: `${map.color}14`,
        border: `1px solid ${map.color}55`,
      }}
    >
      <span style={{ color: map.color, fontFamily: MONO, fontSize: 10, fontWeight: 700, letterSpacing: '0.08em' }}>
        {map.label}
      </span>
      <span style={{ color: '#8b92a8', fontFamily: MONO, fontSize: 8.5, letterSpacing: '0.06em' }}>
        {map.sub}
      </span>
    </span>
  )
}

// ── KeyValue rows ─────────────────────────────────────────────────────────────

/** Compact definition list — "what info you gain" style structured content. */
export function KeyValueList({ items }: { items: { k: ReactNode; v: ReactNode }[] }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 5, margin: '4px 0' }}>
      {items.map((it, i) => (
        <div key={i} style={{ display: 'flex', gap: 10, alignItems: 'baseline' }}>
          <span
            style={{
              flexShrink: 0,
              minWidth: 92,
              color: '#00e5ff',
              fontFamily: MONO,
              fontSize: 10.5,
              fontWeight: 600,
            }}
          >
            {it.k}
          </span>
          <span style={{ color: '#c9cede', fontSize: 11.5, lineHeight: 1.55 }}>{it.v}</span>
        </div>
      ))}
    </div>
  )
}

// ── ChunkStrip ───────────────────────────────────────────────────────────────

/**
 * Mini diagram of the temporal grid: the clip split into non-overlapping
 * 16-frame windows, each = 0.64 s @ 25 fps. Used to explain why short
 * manipulations can smear across a whole chunk.
 */
export function ChunkStrip({
  chunks = 6,
  highlight = 3,
  label = '16 Frames = 0,64 s @ 25 fps',
}: {
  chunks?: number
  highlight?: number
  label?: string
}) {
  const W = 320
  const H = 34
  const gap = 3
  const cw = (W - gap * (chunks - 1)) / chunks
  return (
    <div style={{ margin: '6px 0' }}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        {Array.from({ length: chunks }, (_, i) => {
          const x = i * (cw + gap)
          const active = i === highlight
          return (
            <g key={i}>
              <rect
                x={x}
                y={4}
                width={cw}
                height={H - 8}
                rx={3}
                fill={active ? 'rgba(255,112,112,0.22)' : '#12151d'}
                stroke={active ? '#ff7070' : '#2a2f42'}
                strokeWidth={active ? 1.4 : 1}
              />
              <text
                x={x + cw / 2}
                y={H / 2 + 3.5}
                textAnchor="middle"
                fontFamily={MONO}
                fontSize={8.5}
                fill={active ? '#ff7070' : '#4d5470'}
              >
                {`#${i}`}
              </text>
            </g>
          )
        })}
      </svg>
      <div style={{ fontSize: 9, fontFamily: MONO, color: '#4d5470', marginTop: 2 }}>
        {label}
      </div>
    </div>
  )
}

// ── Paragraph & list helpers ─────────────────────────────────────────────────

/** Standard body paragraph with the explanation's baseline typography. */
export function P({ children }: { children: ReactNode }) {
  return <p style={{ margin: '4px 0', fontSize: 11.5, lineHeight: 1.65, color: '#c9cede' }}>{children}</p>
}

/** Bulleted list with tuned spacing/colour. */
export function UL({ children }: { children: ReactNode }) {
  return (
    <ul
      style={{
        margin: '4px 0',
        paddingLeft: 16,
        fontSize: 11.5,
        lineHeight: 1.6,
        color: '#c9cede',
        display: 'flex',
        flexDirection: 'column',
        gap: 3,
      }}
    >
      {children}
    </ul>
  )
}

/** Emphasised inline term (kept English for domain words). */
export function Term({ children, color = '#e8eaf0' }: { children: ReactNode; color?: string }) {
  return <strong style={{ color, fontWeight: 700 }}>{children}</strong>
}

// ── BivariateLrpNote ──────────────────────────────────────────────────────────

/**
 * The recurring "why bivariate LRP" explainer, shared by every visual that rests
 * on our dual-seed relevance. Plain AttnLRP is single-target: one backward pass
 * explains a single logit and cannot separate "engaged but undecided" from
 * "leaning fake" — the fake-vs-real contrast is lost. The app therefore runs two
 * backwards (dual-seed) and splits the result into magnitude (engagement →
 * opacity) and direction (fake−real lean → colour). Rendered identically wherever
 * a bivariate map is the substrate, so the concept reads the same in every card.
 *
 * `compact` drops the boxed formula for cards whose real subject is a comparison
 * (Phase 3/4) rather than the derivation. `unit` names the aggregation
 * granularity of the hosting visual (e.g. "pro 16-Frame-Chunk", "pro Wort").
 */
export function BivariateLrpNote({
  compact = false,
  unit,
}: {
  compact?: boolean
  unit?: ReactNode
}) {
  const aggregation = unit ? <> Aggregation hier: {unit}.</> : null
  return (
    <Callout variant="info" title="Warum bivariat: AttnLRP allein ist single-target">
      <p style={{ margin: '2px 0' }}>
        Ein AttnLRP-Backward erklärt nur <Term>einen</Term> Ausgang — z. B. „welche
        Stellen haben den Fake-Logit angehoben". Eine solche Einzelziel-Karte trennt{' '}
        <em>„stark beteiligt, aber unentschieden"</em> nicht von{' '}
        <em>„klar Richtung Fake"</em>; der eigentliche Fake-vs-Real-Kontrast geht
        verloren.
      </p>
      {compact ? (
        <p style={{ margin: '4px 0 2px' }}>
          Deshalb <Term>Dual-Seed</Term>: zwei Backwards über denselben Forward
          liefern <Chip>Deckkraft = |R_fake| + |R_real|</Chip> (Engagement) und{' '}
          <Chip color="#ff7070">Farbe = R_fake − R_real</Chip> (Lean) — die bivariate
          Relevanz, die den Kontrast wiederherstellt.{aggregation}
        </p>
      ) : (
        <>
          <p style={{ margin: '4px 0 2px' }}>
            Deshalb läuft AttnLRP <Term>zweimal</Term> über denselben Forward-Pass
            („Dual-Seed") — je einmal auf den Fake- und den Real-Logit. Aus beiden
            Karten entstehen zwei entkoppelte Größen:
          </p>
          <Formula>{`magnitude = |R_fake| + |R_real|   → Engagement  (Deckkraft)
direction = R_fake − R_real       → Lean       (Farbe)`}</Formula>
          <p style={{ margin: '2px 0' }}>
            So bleibt die Logit-Differenz erhalten, die eine Einzelziel-Karte
            wegwirft.{aggregation}
          </p>
        </>
      )}
    </Callout>
  )
}

// ── DeadzoneNote ──────────────────────────────────────────────────────────────

/**
 * Shared "why a dead-zone around 0" explainer for the audio relevance layers.
 *
 * Carries the ARCHITECTURAL rationale identically everywhere: a fake-detection
 * transformer wobbles with near-0 relevance on genuine chunks (mere absence of a
 * manipulation), and can barely lean strongly toward "real" because it is trained
 * to flag fakes, not to certify real segments — so the real pole is weak/diffuse.
 * Each layer then passes its own MECHANISM as `children`.
 *
 * `present` flips the framing: L1/L2 apply a dead-zone (info); L3's magnitude view
 * deliberately does NOT (tip), to make the contrast explicit.
 */
export function DeadzoneNote({
  present,
  children,
}: {
  present: boolean
  children?: ReactNode
}) {
  return (
    <Callout
      variant={present ? 'info' : 'tip'}
      title={present ? 'Dead-Zone um 0 (bewusst)' : 'Bewusst KEINE Magnitude-Dead-Zone'}
    >
      <p style={{ margin: '2px 0' }}>
        Transformer schwanken auf echten Chunks mit einer Relevanz <em>nahe 0</em> —
        das ist nur die <Term>Abwesenheit einer Manipulation</Term>, kein
        Real-Beweis. Die Relevanz kann architektonisch kaum stark Richtung{' '}
        <Term color="#5e91ee">real</Term> zeigen: Das Modell ist trainiert, um{' '}
        <Term color="#ff7070">Fakes</Term> zu erkennen, nicht um echte Abschnitte zu
        bestätigen — der real-Pol bleibt daher schwach und diffus.
      </p>
      {children}
    </Callout>
  )
}

// ── RelevanceScaleNote ────────────────────────────────────────────────────────

/**
 * Shared note that explains, in plain terms, how to read a relevance value: it is
 * a relative score for how strongly a segment influenced the decision — not an
 * absolute value, not a "% of attention", and only comparable within the same
 * visual. Everything specific to THIS visual (what the value is here, why it is
 * small or reaches ~1, how the bars/opacity are scaled) lives in the per-visual
 * `frame`, so the generic lead never contradicts a specific card.
 */
export function RelevanceScaleNote({ frame }: { frame: ReactNode }) {
  return (
    <Callout variant="warn" title="Wie der Relevanz-Wert zu lesen ist">
      <p style={{ margin: '2px 0' }}>
        Relevanz zeigt, wie stark ein Abschnitt die Entscheidung des Modells
        beeinflusst hat. Sie ist relativ und kein Prozentwert und nur innerhalb
        dieses Visuals vergleichbar, nicht zwischen verschiedenen Clips.
      </p>
      <p style={{ margin: '4px 0 2px' }}>{frame}</p>
    </Callout>
  )
}
