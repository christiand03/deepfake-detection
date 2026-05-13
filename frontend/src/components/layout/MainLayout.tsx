import type { ReactNode } from 'react'

interface MainLayoutProps {
  left: ReactNode
  right: ReactNode
  bottom?: ReactNode
}

/**
 * Two-column (60/40) main layout with optional bottom panel.
 * Left: video analysis pipeline. Right: verdict + xAI panels.
 */
export function MainLayout({ left, right, bottom }: MainLayoutProps) {
  return (
    <div className="flex flex-col flex-1" style={{ minWidth: 1280 }}>
      {/* Two-column content area */}
      <div
        className="grid flex-1"
        style={{
          gridTemplateColumns: '1fr 0.67fr',
          gap: '1px',
          backgroundColor: '#2a2f42',
        }}
      >
        {/* Left column — 60% */}
        <div
          className="flex flex-col gap-4 p-5"
          style={{ backgroundColor: '#0d0f14' }}
        >
          {left}
        </div>

        {/* Right column — 40% */}
        <div
          className="flex flex-col gap-4 p-5"
          style={{ backgroundColor: '#0d0f14' }}
        >
          {right}
        </div>
      </div>

      {/* Optional bottom panel (Phase tabs) */}
      {bottom && (
        <div
          style={{
            borderTop: '1px solid #2a2f42',
            backgroundColor: '#0d0f14',
          }}
        >
          {bottom}
        </div>
      )}
    </div>
  )
}
