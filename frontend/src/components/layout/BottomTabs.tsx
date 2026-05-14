/**
 * BottomTabs — full-width tabbed panel below the main two-column layout.
 *
 * Tabs:
 *   AUDIO xAI      — WaveformRelevanceLayer + WordTokenChart + FrequencyBandChart
 *   ROBUSTNESS LAB — Social-media degradation simulation (project Phase 3)
 *   ADVERSARIAL LAB — FGSM / PGD attack visualisation (project Phase 4)
 */

import { motion, AnimatePresence } from 'framer-motion'
import { RobustnessPanel } from '../phases/RobustnessPanel'
import { AdversarialPanel } from '../phases/AdversarialPanel'
import type { AnalysisResult } from '../../types/analysis'

type LabTab = 'robustness' | 'adversarial'

interface TabDef {
  id: LabTab
  label: string
  icon: string
  badge: string
}

const TABS: TabDef[] = [
  { id: 'robustness', label: 'Robustness Lab', icon: '📡', badge: 'Phase 3' },
  { id: 'adversarial', label: 'Adversarial Lab', icon: '⚡', badge: 'Phase 4' },
]

interface BottomTabsProps {
  result: AnalysisResult | null
  activeTab: LabTab | null
  onTabChange: (tab: LabTab | null) => void
}

export function BottomTabs({ result, activeTab, onTabChange }: BottomTabsProps) {

  return (
    <div>
      {/* Tab bar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'stretch',
          borderBottom: '1px solid #2a2f42',
          backgroundColor: '#141720',
          paddingLeft: 20,
          gap: 2,
        }}
      >
        {TABS.map(tab => {
          const isActive = activeTab === tab.id
          return (
            <button
              key={tab.id}
              onClick={() => onTabChange(activeTab === tab.id ? null : tab.id)}
              style={{
                position: 'relative',
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                padding: '8px 16px',
                border: 'none',
                borderBottom: `2px solid ${isActive ? '#00e5ff' : 'transparent'}`,
                backgroundColor: 'transparent',
                color: isActive ? '#e8eaf0' : '#4d5470',
                fontFamily: 'monospace',
                fontSize: 11,
                fontWeight: isActive ? 600 : 400,
                letterSpacing: '0.06em',
                cursor: 'pointer',
                transition: 'color 0.15s ease, border-color 0.15s ease',
                whiteSpace: 'nowrap',
              }}
            >
              <span style={{ fontSize: 12 }}>{tab.icon}</span>
              <span>{tab.label.toUpperCase()}</span>
              {tab.badge && (
                <span
                  style={{
                    fontSize: 8,
                    fontFamily: 'monospace',
                    letterSpacing: '0.06em',
                    color: isActive ? '#00e5ff' : '#2a2f42',
                    backgroundColor: isActive ? 'rgba(0,229,255,0.08)' : 'transparent',
                    border: `1px solid ${isActive ? 'rgba(0,229,255,0.25)' : '#2a2f42'}`,
                    borderRadius: 3,
                    padding: '1px 5px',
                    transition: 'all 0.15s ease',
                  }}
                >
                  {tab.badge}
                </span>
              )}
            </button>
          )
        })}

        {/* Spacer + hint */}
        <div style={{ flex: 1 }} />
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            paddingRight: 20,
            fontSize: 9,
            fontFamily: 'monospace',
            color: '#2a2f42',
            letterSpacing: '0.1em',
          }}
        >
          Robustness & Adversarial Labs
        </div>
      </div>

      {/* Tab content — only rendered when a lab is active */}
      <AnimatePresence mode="wait">
        {activeTab !== null && (
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25, ease: 'easeOut' }}
          >
            {activeTab === 'robustness' && <RobustnessPanel result={result} />}
            {activeTab === 'adversarial' && <AdversarialPanel result={result} />}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
