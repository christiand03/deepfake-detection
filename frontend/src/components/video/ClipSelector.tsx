/**
 * ClipSelector — hierarchical clip picker (roadmap H1 + H2).
 *
 * Replaces the flat DemoSelector strip with a cascading Identity → Scenario →
 * Segment → Clip(variant) selector. Every level is a dropdown of
 * thumbnail-left / text-right rows; picking a level auto-opens the next
 * (auto-selecting single-option levels), and a Back button steps up one level.
 * A permanent breadcrumb shows the current path with thumbnails so the user can
 * recognise a clip they viewed earlier. Analysis stays keyed on the flat clip id.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import type { ClipMeta } from '../../types/analysis'
import {
  buildClipTree,
  locateClip,
  variantLabel,
  type IdentityNode,
  type ScenarioNode,
  type SegmentNode,
  type VariantNode,
} from '../../lib/clipTree'

interface ClipSelectorProps {
  clips: ClipMeta[]
  selectedId: string
  onSelect: (id: string) => void
  disabled?: boolean
}

/** Navigation path (keys chosen so far); `undefined` = not yet chosen. */
interface Path {
  identity?: string
  scenario?: string
  segment?: string
}

const LEVEL_TITLES = ['Identity', 'Scenario', 'Segment', 'Clip'] as const

const COLORS = {
  cardBg: '#141720',
  border: '#2a2f42',
  muted: '#4d5470',
  subtle: '#8b92a8',
  bright: '#e8eaf0',
  panelBg: '#0f1119',
  fake: '#ef4444',
  real: '#3b82f6',
}

// ── Small presentational pieces ────────────────────────────────────────────────

/** 16:9 black box with a centered 1:1 face-crop image (letterbox bars L/R). */
function ThumbBox({
  posterSrc,
  height,
  children,
}: {
  posterSrc: string
  height: number
  children?: React.ReactNode
}) {
  const width = Math.round((height * 16) / 9)
  return (
    <div
      className="relative flex-shrink-0 overflow-hidden rounded"
      style={{ width, height, backgroundColor: '#000' }}
    >
      {posterSrc && (
        <img
          src={posterSrc}
          alt=""
          className="absolute top-0 h-full object-cover"
          style={{ width: height, left: '50%', transform: 'translateX(-50%)' }}
          onError={e => {
            ;(e.currentTarget as HTMLImageElement).style.display = 'none'
          }}
        />
      )}
      {children}
    </div>
  )
}

function VariantBadge({ label }: { label: 'FAKE' | 'REAL' }) {
  const isFake = label === 'FAKE'
  return (
    <span
      className="px-1.5 py-0.5 rounded font-mono font-bold"
      style={{
        fontSize: 10,
        backgroundColor: isFake ? 'rgba(239,68,68,0.2)' : 'rgba(59,130,246,0.2)',
        color: isFake ? COLORS.fake : COLORS.real,
        border: `1px solid ${isFake ? 'rgba(239,68,68,0.4)' : 'rgba(59,130,246,0.4)'}`,
      }}
    >
      {label}
    </span>
  )
}

// ── Label helpers ──────────────────────────────────────────────────────────────

function optionLabel(level: number, node: OptionNode): string {
  switch (level) {
    case 0:
      return (node as IdentityNode).key
    case 1:
      return `Scenario ${(node as ScenarioNode).index}`
    case 2:
      return `Segment ${(node as SegmentNode).index}`
    default:
      return variantLabel((node as VariantNode).variant)
  }
}

type OptionNode = IdentityNode | ScenarioNode | SegmentNode | VariantNode

function nodeRepr(node: OptionNode): VariantNode {
  return 'repr' in node ? node.repr : node
}

function nodeKey(level: number, node: OptionNode): string {
  return level === 3 ? (node as VariantNode).clipId : (node as { key: string }).key
}

// ── Component ───────────────────────────────────────────────────────────────────

export function ClipSelector({ clips, selectedId, onSelect, disabled }: ClipSelectorProps) {
  const tree = useMemo(() => buildClipTree(clips), [clips])
  // `navPath` overrides the derived path during in-progress navigation; it resets
  // to null (fall back to the selected clip) whenever `selectedId` changes.
  const [navPath, setNavPath] = useState<Path | null>(null)
  const [prevSelected, setPrevSelected] = useState(selectedId)
  const [openLevel, setOpenLevel] = useState<number | null>(null)
  const rootRef = useRef<HTMLDivElement>(null)

  // Adjust state during render (React-endorsed) instead of a syncing effect:
  // when the externally-selected clip changes, drop any navigation override.
  if (selectedId !== prevSelected) {
    setPrevSelected(selectedId)
    setNavPath(null)
  }

  const derived = locateClip(clips, selectedId)
  const path: Path =
    navPath ??
    (derived
      ? { identity: derived.identity, scenario: derived.scenario, segment: derived.segment }
      : {})

  // Close the dropdown on outside-click / Escape.
  useEffect(() => {
    if (openLevel === null) return
    function onDown(e: MouseEvent) {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpenLevel(null)
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') setOpenLevel(null)
    }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [openLevel])

  // Resolve current nodes from the path.
  const identityNode = tree.find(n => n.key === path.identity)
  const scenarioNode = identityNode?.scenarios.find(s => s.key === path.scenario)
  const segmentNode = scenarioNode?.segments.find(s => s.key === path.segment)
  const selectedVariant = segmentNode?.variants.find(v => v.clipId === selectedId)

  // Options for a level given a candidate path (used by auto-advance too).
  function optionsFor(level: number, p: Path): OptionNode[] {
    if (level === 0) return tree
    const id = tree.find(n => n.key === p.identity)
    if (level === 1) return id?.scenarios ?? []
    const sc = id?.scenarios.find(s => s.key === p.scenario)
    if (level === 2) return sc?.segments ?? []
    const sg = sc?.segments.find(s => s.key === p.segment)
    return sg?.variants ?? []
  }

  function applySelect(level: number, p: Path, node: OptionNode): Path {
    if (level === 0) return { identity: (node as IdentityNode).key }
    if (level === 1) return { ...p, scenario: (node as ScenarioNode).key, segment: undefined }
    return { ...p, segment: (node as SegmentNode).key }
  }

  // From `start`, auto-select single-option levels (1..2) and return where to stop.
  function resolveOpen(start: number, base: Path): { path: Path; open: number } {
    let p = base
    let lvl = start
    while (lvl < 3) {
      const opts = optionsFor(lvl, p)
      if (opts.length === 1) {
        p = applySelect(lvl, p, opts[0])
        lvl++
      } else break
    }
    return { path: p, open: lvl }
  }

  function handleOptionClick(level: number, node: OptionNode) {
    if (disabled) return
    if (level === 3) {
      onSelect((node as VariantNode).clipId)
      setOpenLevel(null)
      return
    }
    const next = applySelect(level, path, node)
    const { path: resolved, open } = resolveOpen(level + 1, next)
    setNavPath(resolved)
    setOpenLevel(open)
  }

  function handleBack() {
    setOpenLevel(lvl => (lvl && lvl > 0 ? lvl - 1 : lvl))
  }

  const isFakeSel = (selectedVariant?.label ?? 'REAL') === 'FAKE'

  // ── Breadcrumb chips ──────────────────────────────────────────────────────────

  const chips: {
    level: number
    filled: boolean
    poster: string
    caption: string
    badge?: 'FAKE' | 'REAL'
  }[] = [
    {
      level: 0,
      filled: !!identityNode,
      poster: identityNode?.repr.posterSrc ?? '',
      caption: identityNode?.key ?? 'Select…',
    },
    {
      level: 1,
      filled: !!scenarioNode,
      poster: scenarioNode?.repr.posterSrc ?? '',
      caption: scenarioNode ? `Scenario ${scenarioNode.index}` : '—',
    },
    {
      level: 2,
      filled: !!segmentNode,
      poster: segmentNode?.repr.posterSrc ?? '',
      caption: segmentNode ? `Segment ${segmentNode.index}` : '—',
    },
    {
      level: 3,
      filled: !!selectedVariant,
      poster: selectedVariant?.posterSrc ?? '',
      caption: selectedVariant ? variantLabel(selectedVariant.variant) : '—',
      badge: selectedVariant?.label,
    },
  ]

  return (
    <div ref={rootRef} className="relative flex flex-col gap-2" style={{ zIndex: 20 }}>
      <span className="text-xs font-mono tracking-widest" style={{ color: COLORS.muted }}>
        CLIP SELECTION
      </span>

      {/* Permanent breadcrumb bar */}
      <div className="flex items-end gap-1 overflow-x-auto pb-1" style={{ scrollbarWidth: 'thin' }}>
        {chips.map((chip, i) => (
          <div key={chip.level} className="flex items-end gap-1">
            {i > 0 && (
              <span className="pb-3 text-sm" style={{ color: COLORS.muted }}>
                ›
              </span>
            )}
            <button
              onClick={() => !disabled && chip.filled && setOpenLevel(chip.level)}
              disabled={disabled}
              className="flex flex-col gap-1 items-start transition-opacity"
              style={{ cursor: disabled ? 'not-allowed' : 'pointer', opacity: disabled ? 0.5 : 1 }}
            >
              <span
                className="font-mono tracking-widest uppercase"
                style={{ fontSize: 9, color: openLevel === chip.level ? COLORS.bright : COLORS.muted }}
              >
                {LEVEL_TITLES[chip.level]}
              </span>
              <ThumbBox posterSrc={chip.poster} height={56}>
                <div
                  className="absolute inset-0"
                  style={{
                    border: `2px solid ${
                      openLevel === chip.level ? COLORS.subtle : chip.filled ? COLORS.border : '#1b1f2e'
                    }`,
                    borderRadius: 4,
                  }}
                />
                {chip.badge && (
                  <div className="absolute top-0.5 right-0.5">
                    <VariantBadge label={chip.badge} />
                  </div>
                )}
                <div
                  className="absolute bottom-0 left-0 right-0 px-1 py-0.5 text-left"
                  style={{
                    background: 'linear-gradient(to top, rgba(13,15,20,0.92) 0%, transparent 100%)',
                    fontSize: 10,
                    color: chip.filled ? COLORS.bright : COLORS.muted,
                    lineHeight: 1.2,
                  }}
                >
                  {chip.caption}
                </div>
              </ThumbBox>
            </button>
          </div>
        ))}
      </div>

      {/* Dropdown panel — overlays the video player below */}
      {openLevel !== null && (
        <div
          className="absolute left-0 right-0 rounded-lg shadow-xl"
          style={{
            top: '100%',
            marginTop: 4,
            zIndex: 40,
            backgroundColor: COLORS.panelBg,
            border: `1px solid ${COLORS.border}`,
          }}
        >
          {/* Header: level title + Back */}
          <div
            className="flex items-center justify-between px-3 py-2"
            style={{ borderBottom: `1px solid ${COLORS.border}` }}
          >
            <span className="text-xs font-mono tracking-widest" style={{ color: COLORS.subtle }}>
              {LEVEL_TITLES[openLevel].toUpperCase()}
            </span>
            <button
              onClick={handleBack}
              disabled={openLevel === 0}
              className="text-xs font-mono px-2 py-1 rounded transition-colors"
              style={{
                color: openLevel === 0 ? COLORS.muted : COLORS.subtle,
                cursor: openLevel === 0 ? 'default' : 'pointer',
                opacity: openLevel === 0 ? 0.4 : 1,
                border: `1px solid ${COLORS.border}`,
              }}
            >
              ← Back
            </button>
          </div>

          {/* Rows */}
          <div className="overflow-y-auto py-1" style={{ maxHeight: 340, scrollbarWidth: 'thin' }}>
            {optionsFor(openLevel, path).map(node => {
              const key = nodeKey(openLevel, node)
              const repr = nodeRepr(node)
              const isVariant = openLevel === 3
              const label = optionLabel(openLevel, node)
              const currentKey =
                openLevel === 0
                  ? path.identity
                  : openLevel === 1
                    ? path.scenario
                    : openLevel === 2
                      ? path.segment
                      : selectedId
              const isSel = key === currentKey
              const variantLabelColor = isVariant
                ? (node as VariantNode).label === 'FAKE'
                  ? COLORS.fake
                  : COLORS.real
                : COLORS.bright
              return (
                <button
                  key={key}
                  onClick={() => handleOptionClick(openLevel, node)}
                  className="flex w-full items-center gap-3 px-3 py-2 text-left transition-colors"
                  style={{ backgroundColor: isSel ? 'rgba(59,130,246,0.08)' : 'transparent' }}
                  onMouseEnter={e => {
                    if (!isSel) e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.03)'
                  }}
                  onMouseLeave={e => {
                    if (!isSel) e.currentTarget.style.backgroundColor = 'transparent'
                  }}
                >
                  <ThumbBox posterSrc={repr.posterSrc} height={80}>
                    <div
                      className="absolute inset-0 rounded"
                      style={{ border: isSel ? `2px solid ${COLORS.subtle}` : 'none' }}
                    />
                  </ThumbBox>
                  <div className="flex flex-col items-start gap-1">
                    <span style={{ fontSize: 13, color: variantLabelColor, fontWeight: isSel ? 600 : 400 }}>
                      {label}
                    </span>
                    {isVariant && <VariantBadge label={(node as VariantNode).label} />}
                  </div>
                </button>
              )
            })}
          </div>
        </div>
      )}

      {/* Reserve a hairline so the selected clip's fake/real accent is visible */}
      <div style={{ height: 1, backgroundColor: isFakeSel ? 'rgba(239,68,68,0.25)' : 'rgba(59,130,246,0.25)' }} />
    </div>
  )
}
