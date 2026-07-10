/**
 * Client-side grouping of the flat clip list into the four-level hierarchy the
 * ClipSelector navigates: Identity → Scenario → Segment → Variant (roadmap H1).
 *
 * The backend keeps analysis keyed on the flat `clip_id`; this module only
 * organises the same clips for presentation and picks a representative clip per
 * node for its thumbnail (`posterSrc`).
 */

import type { ClipMeta } from '../types/analysis'

/** Fixed display order for the 2×2 variant matrix (real first, then the fakes). */
const VARIANT_ORDER = [
  'real',
  'fake_video_fake_audio',
  'fake_video_real_audio',
  'real_video_fake_audio',
] as const

/** Human-readable label for a variant key. */
export function variantLabel(variant: string): string {
  switch (variant) {
    case 'real':
      return 'Real'
    case 'fake_video_fake_audio':
      return 'Fake video · Fake audio'
    case 'fake_video_real_audio':
      return 'Fake video · Real audio'
    case 'real_video_fake_audio':
      return 'Real video · Fake audio'
    default:
      return variant.replace(/_/g, ' ')
  }
}

export interface VariantNode {
  /** Real `clip_XX` id passed to the analysis pipeline. */
  clipId: string
  label: 'FAKE' | 'REAL'
  variant: string
  posterSrc: string
  clip: ClipMeta
}

export interface SegmentNode {
  key: string
  /** 1-based position for the "Segment N" label. */
  index: number
  variants: VariantNode[]
  /** Representative variant for this node's thumbnail (prefers `real`). */
  repr: VariantNode
}

export interface ScenarioNode {
  key: string
  index: number
  segments: SegmentNode[]
  repr: VariantNode
}

export interface IdentityNode {
  key: string
  scenarios: ScenarioNode[]
  repr: VariantNode
}

/** Prefer the `real` variant as a node's representative, else the first. */
function pickRepr(variants: VariantNode[]): VariantNode {
  return variants.find(v => v.variant === 'real') ?? variants[0]
}

function sortVariants(variants: VariantNode[]): VariantNode[] {
  const rank = (v: string) => {
    const i = VARIANT_ORDER.indexOf(v as (typeof VARIANT_ORDER)[number])
    return i === -1 ? VARIANT_ORDER.length : i
  }
  return [...variants].sort((a, b) => rank(a.variant) - rank(b.variant))
}

/** Build the Identity → Scenario → Segment → Variant tree from a flat clip list. */
export function buildClipTree(clips: ClipMeta[]): IdentityNode[] {
  // Nested insertion-ordered maps preserve the clips.json ordering.
  const ids = new Map<string, Map<string, Map<string, VariantNode[]>>>()

  for (const clip of clips) {
    const variantNode: VariantNode = {
      clipId: clip.id,
      label: clip.label,
      variant: clip.variant,
      posterSrc: clip.posterSrc,
      clip,
    }
    const scenarios = ids.get(clip.identity) ?? new Map()
    ids.set(clip.identity, scenarios)
    const segments = scenarios.get(clip.scenario) ?? new Map()
    scenarios.set(clip.scenario, segments)
    const variants = segments.get(clip.segment) ?? []
    variants.push(variantNode)
    segments.set(clip.segment, variants)
  }

  const tree: IdentityNode[] = []
  for (const [identityKey, scenarioMap] of ids) {
    const scenarios: ScenarioNode[] = []
    let sIndex = 1
    for (const [scenarioKey, segmentMap] of scenarioMap) {
      const segments: SegmentNode[] = []
      let segIndex = 1
      for (const [segmentKey, variants] of segmentMap) {
        const sorted = sortVariants(variants)
        segments.push({
          key: segmentKey,
          index: segIndex++,
          variants: sorted,
          repr: pickRepr(sorted),
        })
      }
      scenarios.push({
        key: scenarioKey,
        index: sIndex++,
        segments,
        repr: segments[0].repr,
      })
    }
    tree.push({ key: identityKey, scenarios, repr: scenarios[0].repr })
  }
  return tree
}

/** The identity/scenario/segment keys a `clipId` lives under, or null. */
export function locateClip(
  clips: ClipMeta[],
  clipId: string,
): { identity: string; scenario: string; segment: string } | null {
  const clip = clips.find(c => c.id === clipId)
  if (!clip) return null
  return { identity: clip.identity, scenario: clip.scenario, segment: clip.segment }
}
