import { Scan } from 'lucide-react'
import { useBackendHealth } from '../../hooks/useBackendHealth'
import type { BackendStatus } from '../../hooks/useBackendHealth'

const STATUS_CONFIG: Record<
  BackendStatus,
  { dot: string; label: string; bg: string; border: string; text: string }
> = {
  mock: {
    dot: '#f59e0b',
    label: 'MOCK',
    bg: 'rgba(245,158,11,0.08)',
    border: 'rgba(245,158,11,0.25)',
    text: '#f59e0b',
  },
  online: {
    dot: '#22c55e',
    label: 'LIVE',
    bg: 'rgba(34,197,94,0.08)',
    border: 'rgba(34,197,94,0.25)',
    text: '#22c55e',
  },
  offline: {
    dot: '#ef4444',
    label: 'OFFLINE',
    bg: 'rgba(239,68,68,0.08)',
    border: 'rgba(239,68,68,0.25)',
    text: '#ef4444',
  },
  pending: {
    dot: '#4d5470',
    label: '…',
    bg: 'rgba(77,84,112,0.08)',
    border: 'rgba(77,84,112,0.25)',
    text: '#4d5470',
  },
}

export function Header() {
  const status = useBackendHealth()
  const cfg = STATUS_CONFIG[status]

  return (
    <header
      className="flex items-center justify-between px-6 py-3 border-b"
      style={{
        backgroundColor: '#141720',
        borderColor: '#2a2f42',
      }}
    >
      {/* Left: Logo + Title */}
      <div className="flex items-center gap-3">
        <div
          className="flex items-center justify-center w-8 h-8 rounded"
          style={{ backgroundColor: 'rgba(0,229,255,0.12)', border: '1px solid rgba(0,229,255,0.3)' }}
        >
          <Scan size={16} style={{ color: '#00e5ff' }} />
        </div>
        <div>
          <div
            className="text-sm font-semibold tracking-wide"
            style={{ color: '#e8eaf0', letterSpacing: '0.04em' }}
          >
            MULTIMODAL DEEPFAKE xAI
          </div>
          <div className="text-xs" style={{ color: '#4d5470' }}>
            Cross-Modal Attention · AttnLRP
          </div>
        </div>
      </div>

      {/* Center: Phase badge */}
      <div
        className="px-3 py-1 rounded-full text-xs font-mono font-medium tracking-wider"
        style={{
          backgroundColor: 'rgba(0,229,255,0.08)',
          border: '1px solid rgba(0,229,255,0.2)',
          color: '#00e5ff',
        }}
      >
        VideoMAE · Wav2Vec 2.0
      </div>

      {/* Right: backend status + GitHub */}
      <div className="flex items-center gap-4">
        {/* Backend status pill */}
        <div
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono font-semibold"
          style={{
            backgroundColor: cfg.bg,
            border: `1px solid ${cfg.border}`,
            color: cfg.text,
          }}
          title={
            status === 'mock'
              ? 'Running in mock mode — set VITE_USE_MOCK=false to connect to the API'
              : status === 'online'
                ? 'FastAPI backend is reachable'
                : status === 'offline'
                  ? 'FastAPI backend is not reachable'
                  : 'Checking backend…'
          }
        >
          <span
            style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              backgroundColor: cfg.dot,
              display: 'inline-block',
              flexShrink: 0,
              boxShadow: status === 'online' ? `0 0 6px ${cfg.dot}` : 'none',
            }}
          />
          {cfg.label}
        </div>

        {/* GitHub link */}
        <a
          href="https://github.com/christiand03/deepfake-detection"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 text-xs transition-colors"
          style={{ color: '#8b92a8' }}
          onMouseEnter={e => ((e.currentTarget as HTMLAnchorElement).style.color = '#e8eaf0')}
          onMouseLeave={e => ((e.currentTarget as HTMLAnchorElement).style.color = '#8b92a8')}
        >
          <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
            <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844a9.59 9.59 0 012.504.337c1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.02 10.02 0 0022 12.017C22 6.484 17.522 2 12 2z" />
          </svg>
          <span>christiand03/deepfake-detection</span>
        </a>
      </div>
    </header>
  )
}
