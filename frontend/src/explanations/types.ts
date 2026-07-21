/**
 * Explanation content model (F1).
 *
 * Each visualisation in the UI has one explanation, authored as a `.tsx` file
 * under `content/`. The dialog structure is coded ONCE (`ui/ExplanationDialog`);
 * every explanation just supplies a title, a few meta fields and an ordered list
 * of `sections`. Only the sections a visual actually needs are provided — simple
 * visuals stay short, complex ones go deep ("more than needed rather than too
 * little", but never filler).
 *
 * Language rule (project-wide): prose is German, domain terms that map directly
 * to on-screen labels stay English (confidence, relevance, heatmap, chunk, …).
 */

import type { ReactNode } from 'react'

/**
 * Stable id per visual. Buttons reference a `VisualId`; the registry resolves it
 * to an `Explanation`. A button whose id is not yet in the registry renders
 * nothing, so content and buttons can be wired independently.
 */
export type VisualId =
  // Video panel
  | 'heatmap-overlay'
  | 'chunk-timelines'
  | 'region-face'
  | 'rotation-warning'
  // Verdict
  | 'verdict-gauges'
  // Audio panel
  | 'audio-toggle'
  | 'audio-l1-waveform'
  | 'audio-l2-words'
  | 'audio-l3-frequency'
  // Phase 3 — Robustness
  | 'robustness-confidence'
  | 'robustness-crop-compare'
  | 'attention-shift'
  | 'audio-frequency-shift'
  // Phase 4 — Adversarial
  | 'adversarial-heatmaps'
  | 'adversarial-confidence'

/**
 * The coverage checklist agreed for F1. Each kind is one block in the dialog,
 * with a default German header and an icon (see `SECTION_META`). A visual fills
 * only the kinds that add value for it.
 */
export type SectionKind =
  | 'what' // Was ist das
  | 'purpose' // Was tut es / welche Frage beantwortet es
  | 'method' // Wie wird gerechnet (zugrundeliegende xAI-Methode)
  | 'legend' // Farb-/Achsen-Legende
  | 'normalization' // Normalisierung & Dead-Zones
  | 'resolution' // Zeitliche/räumliche Auflösung & Granularität
  | 'interpret' // Wie liest man es
  | 'reading' // Konkretes Lese-Beispiel
  | 'gain' // Welche Information gewinnt man
  | 'pitfalls' // Häufige Fehlinterpretationen
  | 'trust' // Wann ist das Signal vertrauenswürdig
  | 'limitations' // Limitations
  | 'interaction' // Bedienung / Interaktion
  | 'links' // Verknüpfung mit anderen Visuals

/** Where a visual sits on the two complementary axes (F1 core message). */
export type ConfidenceRelevance = 'confidence' | 'relevance' | 'both' | 'neither'

export interface ExplanationSection {
  kind: SectionKind
  /** Overrides the default German header for this kind. */
  title?: string
  body: ReactNode
}

export interface Explanation {
  id: VisualId
  /** Human title shown in the dialog header (domain term, usually English). */
  title: string
  /** One-line what-it-is, shown under the title. */
  subtitle?: string
  /** Method/model badge, e.g. "AttnLRP · bivariate LRP". */
  method?: string
  /** Which axis this visual measures — drives the badge in the header. */
  cvr?: ConfidenceRelevance
  sections: ExplanationSection[]
}

/** Per-kind default header, short glyph and accent colour used by the dialog. */
export const SECTION_META: Record<
  SectionKind,
  { label: string; glyph: string; color: string }
> = {
  what: { label: 'Was das Visual zeigt', glyph: '◈', color: '#00e5ff' },
  purpose: { label: 'Zweck und Fragestellung', glyph: '◎', color: '#00e5ff' },
  method: { label: 'Berechnung und Methodik', glyph: '⚙', color: '#8b92a8' },
  legend: { label: 'Farbskala und Achsen', glyph: '▤', color: '#8b92a8' },
  normalization: { label: 'Normalisierung und Dead-Zones', glyph: '∿', color: '#8b92a8' },
  resolution: { label: 'Auflösung und Granularität', glyph: '⌗', color: '#8b92a8' },
  interpret: { label: 'Wie das Visual interpretiert werden sollte', glyph: '👁', color: '#00e5ff' },
  reading: { label: 'Beispielhafte Interpretation', glyph: '↳', color: '#22c55e' },
  gain: { label: 'Erkenntnisgewinn', glyph: '✦', color: '#00e5ff' },
  pitfalls: { label: 'Häufige Fehlinterpretationen', glyph: '⚠', color: '#f59e0b' },
  trust: { label: 'Verlässlichkeit des Signals', glyph: '◉', color: '#f59e0b' },
  limitations: { label: 'Grenzen und Einschränkungen', glyph: '▽', color: '#8b92a8' },
  interaction: { label: 'Bedienung und Interaktion', glyph: '⇲', color: '#8b92a8' },
  links: { label: 'Zusammenhang mit anderen Visualisierungen', glyph: '⇄', color: '#a855f7' },
}

/** Canonical order sections are rendered in, regardless of author order. */
export const SECTION_ORDER: SectionKind[] = [
  'what',
  'purpose',
  'method',
  'legend',
  'normalization',
  'resolution',
  'interpret',
  'reading',
  'gain',
  'pitfalls',
  'trust',
  'limitations',
  'interaction',
  'links',
]
