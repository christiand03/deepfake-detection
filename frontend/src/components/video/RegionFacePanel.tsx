/**
 * RegionFacePanel — wraps the video player and adds a left-edge toggle tab that
 * slides the whole-clip FaceSchematic in FROM THE LEFT, over the player.
 *
 * The face map answers "WHERE did the model look across the whole clip" (spatial
 * aggregate); it takes over the player's larger area so the schematic reads at a
 * good size, while the timelines ("WHEN did it react") stay visible below. The
 * tab flips the player between video and face map. Hidden entirely when the clip
 * carries no per-region data (face-less fallback / older cached result).
 */

import { AnimatePresence, motion } from 'framer-motion'

import type { RegionRelevance } from '../../types/analysis'
import { FaceSchematic } from './FaceSchematic'

export function RegionFacePanel({
  regions,
  rotated = false,
  open,
  onOpenChange,
  children,
}: {
  regions: RegionRelevance[]
  /** Clip's face is near profile → the schematic shows an unreliability caution. */
  rotated?: boolean
  /** Controlled open state (lifted to VideoPanel so the explain button can
   *  switch between the heatmap and the region-face explanation). */
  open: boolean
  onOpenChange: (open: boolean) => void
  children: React.ReactNode
}) {
  const hasData = regions.length > 0

  return (
    <div style={{ position: 'relative' }}>
      {children}

      <AnimatePresence>
        {open && hasData && (
          <motion.div
            key="face-map"
            initial={{ x: '-102%' }}
            animate={{ x: 0 }}
            exit={{ x: '-102%' }}
            transition={{ type: 'spring', stiffness: 280, damping: 32 }}
            style={{
              position: 'absolute',
              inset: 0,
              zIndex: 5,
              borderRadius: 8,
              overflow: 'hidden',
              backgroundColor: '#141720',
              border: '1px solid #2a2f42',
              boxShadow: '0 10px 34px rgba(0,0,0,0.55)',
            }}
          >
            <FaceSchematic regions={regions} rotated={rotated} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Left-edge toggle tab — stays above the sliding panel so it can close it. */}
      {hasData && (
        <button
          type="button"
          onClick={() => onOpenChange(!open)}
          title={open ? 'Hide face map' : 'Show whole-clip face relevance map'}
          style={{
            position: 'absolute',
            left: 0,
            top: '50%',
            transform: 'translateY(-50%)',
            zIndex: 6,
            width: 20,
            height: 92,
            padding: 0,
            border: '1px solid #2a2f42',
            borderLeft: 'none',
            borderRadius: '0 6px 6px 0',
            backgroundColor: open ? '#1f2740' : '#161a26',
            color: open ? '#cdd6f0' : '#6b7390',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <span
            style={{
              writingMode: 'vertical-rl',
              transform: 'rotate(180deg)',
              fontSize: 8.5,
              fontFamily: 'monospace',
              letterSpacing: '0.16em',
              userSelect: 'none',
            }}
          >
            {open ? 'CLOSE' : 'FACE MAP'}
          </span>
        </button>
      )}
    </div>
  )
}
