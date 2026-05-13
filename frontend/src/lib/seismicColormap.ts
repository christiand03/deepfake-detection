/**
 * seismicColormap.ts
 *
 * Maps a value in [-1, 1] to an [R, G, B] triplet using matplotlib's "seismic"
 * colormap — blue (negative/real evidence) → white (zero) → red (positive/fake evidence).
 *
 * The implementation matches the matplotlib seismic key-points exactly so that
 * frontend canvas overlays are visually consistent with Python heatmap PNGs.
 *
 * Key stops (from matplotlib source):
 *   0.00 → #00004B  (deep blue)
 *   0.25 → #0000FF  (pure blue)
 *   0.50 → #FFFFFF  (white / neutral)
 *   0.75 → #FF0000  (pure red)
 *   1.00 → #800000  (dark red)
 */

type RGB = [number, number, number]

// Normalised key-stops: [position, [R, G, B]] — all channels in [0, 255]
const STOPS: [number, RGB][] = [
  [0.00, [0,   0,   75]],
  [0.25, [0,   0,   255]],
  [0.50, [255, 255, 255]],
  [0.75, [255, 0,   0]],
  [1.00, [128, 0,   0]],
]

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

/**
 * Maps value ∈ [-1, 1] to an RGB triplet matching matplotlib seismic.
 * Values outside [-1, 1] are clamped.
 */
export function seismicToRgb(value: number): RGB {
  // Map [-1, 1] → [0, 1]
  const t = Math.max(0, Math.min(1, (value + 1) / 2))

  // Find surrounding stops
  let lo = STOPS[0]
  let hi = STOPS[STOPS.length - 1]

  for (let i = 0; i < STOPS.length - 1; i++) {
    if (t >= STOPS[i][0] && t <= STOPS[i + 1][0]) {
      lo = STOPS[i]
      hi = STOPS[i + 1]
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

/**
 * Converts a flat relevance array to a CSS rgba() string for use as a
 * bar/strip color. Alpha is scaled by the absolute magnitude so that
 * near-zero values are transparent.
 */
export function seismicToCss(value: number, maxAlpha = 0.85): string {
  const [r, g, b] = seismicToRgb(value)
  const alpha = maxAlpha * Math.abs(value)
  return `rgba(${r},${g},${b},${alpha.toFixed(3)})`
}
