/**
 * seismicColormap.ts
 *
 * Maps a value in [-1, 1] to an [R, G, B] triplet.
 *
 * Two ramps are provided:
 *   - `seismicToRgb`  — faithful matplotlib "seismic": blue (real) → white →
 *     red (fake), darkening to near-black navy / maroon at the poles. Kept for
 *     exact parity with the Python heatmap key-points.
 *   - `relevanceToRgb` — dark-background-tuned variant (F2): the poles are lifted
 *     to BRIGHT tints so the blue (real) end stays legible on the dark-grey panels
 *     instead of collapsing into the near-black #00004B navy. White centre; the
 *     call sites still scale alpha by |value|. Used by the audio charts.
 *
 * matplotlib seismic key-stops:
 *   0.00 → #00004B  (deep blue)   0.25 → #0000FF  (pure blue)
 *   0.50 → #FFFFFF  (white)       0.75 → #FF0000  (pure red)   1.00 → #800000
 */

type RGB = [number, number, number]

// Faithful matplotlib seismic — [position, [R, G, B]], channels in [0, 255].
const SEISMIC_STOPS: [number, RGB][] = [
  [0.0, [0, 0, 75]],
  [0.25, [0, 0, 255]],
  [0.5, [255, 255, 255]],
  [0.75, [255, 0, 0]],
  [1.0, [128, 0, 0]],
]

// Dark-background-tuned ramp (F2): same blue↔white↔red identity, but tuned for the
// dark-grey panels. The mid stops are clearly tinted (not near-white) so low values
// already read as blue/red instead of washing out, and the poles are deeper —
// legible, less neon. Symmetric so the scale stays balanced.
const VIVID_STOPS: [number, RGB][] = [
  [0.0, [46, 99, 214]], // #2E63D6 — strong real (deep blue, still legible)
  [0.25, [94, 145, 238]], // #5E91EE — clear mid blue (no longer near-white)
  [0.5, [255, 255, 255]], // #FFFFFF — neutral
  [0.75, [255, 112, 112]], // #FF7070 — clear mid red
  [1.0, [255, 59, 59]], // #FF3B3B — strong fake (bright red)
]

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

/** Interpolate a value ∈ [-1, 1] over the given key-stops (clamped). */
function interpolate(stops: [number, RGB][], value: number): RGB {
  // Map [-1, 1] → [0, 1]
  const t = Math.max(0, Math.min(1, (value + 1) / 2))

  let lo = stops[0]
  let hi = stops[stops.length - 1]
  for (let i = 0; i < stops.length - 1; i++) {
    if (t >= stops[i][0] && t <= stops[i + 1][0]) {
      lo = stops[i]
      hi = stops[i + 1]
      break
    }
  }

  const range = hi[0] - lo[0]
  const localT = range === 0 ? 0 : (t - lo[0]) / range
  return [
    Math.round(lerp(lo[1][0], hi[1][0], localT)),
    Math.round(lerp(lo[1][1], hi[1][1], localT)),
    Math.round(lerp(lo[1][2], hi[1][2], localT)),
  ]
}

/** Faithful matplotlib seismic mapping of value ∈ [-1, 1] → RGB. */
export function seismicToRgb(value: number): RGB {
  return interpolate(SEISMIC_STOPS, value)
}

/**
 * Dark-background-tuned relevance mapping of value ∈ [-1, 1] → RGB (F2).
 * Use this for visualisations rendered on the dark-grey panels (audio charts) so
 * the blue/real end stays legible; the centre stays white/neutral.
 */
export function relevanceToRgb(value: number): RGB {
  return interpolate(VIVID_STOPS, value)
}
