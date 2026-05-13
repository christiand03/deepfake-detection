import type { ClipMeta } from '../../types/analysis'

interface DemoSelectorProps {
  clips: ClipMeta[]
  selectedId: string
  onSelect: (id: string) => void
  disabled?: boolean
}

export function DemoSelector({ clips, selectedId, onSelect, disabled }: DemoSelectorProps) {
  return (
    <div className="flex flex-col gap-2">
      <span
        className="text-xs font-mono tracking-widest"
        style={{ color: '#4d5470' }}
      >
        DEMO CLIP
      </span>
      <div className="flex gap-2 overflow-x-auto pb-1" style={{ scrollbarWidth: 'thin' }}>
        {clips.map(clip => {
          const isSelected = clip.id === selectedId
          const isFake = clip.label === 'FAKE'

          return (
            <button
              key={clip.id}
              onClick={() => !disabled && onSelect(clip.id)}
              disabled={disabled}
              className="flex-shrink-0 relative rounded-lg overflow-hidden transition-all"
              style={{
                width: 140,
                height: 80,
                border: isSelected
                  ? `2px solid ${isFake ? '#ef4444' : '#3b82f6'}`
                  : '2px solid #2a2f42',
                backgroundColor: '#141720',
                cursor: disabled ? 'not-allowed' : 'pointer',
                opacity: disabled ? 0.5 : 1,
                outline: 'none',
              }}
            >
              {/* Poster image or placeholder */}
              <div
                className="absolute inset-0"
                style={{ backgroundColor: '#1b1f2e' }}
              >
                <img
                  src={clip.posterSrc}
                  alt=""
                  className="w-full h-full object-cover"
                  style={{ opacity: 0.7 }}
                  onError={e => { (e.currentTarget as HTMLImageElement).style.display = 'none' }}
                />
              </div>

              {/* Gradient overlay */}
              <div
                className="absolute inset-0"
                style={{
                  background: 'linear-gradient(to top, rgba(13,15,20,0.9) 0%, transparent 60%)',
                }}
              />

              {/* Label chip */}
              <div
                className="absolute top-1.5 right-1.5 px-1.5 py-0.5 rounded text-xs font-mono font-bold"
                style={{
                  backgroundColor: isFake ? 'rgba(239,68,68,0.2)' : 'rgba(59,130,246,0.2)',
                  color: isFake ? '#ef4444' : '#3b82f6',
                  border: `1px solid ${isFake ? 'rgba(239,68,68,0.4)' : 'rgba(59,130,246,0.4)'}`,
                  fontSize: 10,
                }}
              >
                {clip.label}
              </div>

              {/* Title */}
              <div
                className="absolute bottom-0 left-0 right-0 px-2 py-1.5 text-left"
                style={{
                  fontSize: 11,
                  color: isSelected ? '#e8eaf0' : '#8b92a8',
                  lineHeight: 1.3,
                  fontWeight: isSelected ? 600 : 400,
                }}
              >
                {clip.title}
              </div>

              {/* Selected indicator */}
              {isSelected && (
                <div
                  className="absolute inset-0 rounded-lg"
                  style={{
                    boxShadow: `inset 0 0 0 2px ${isFake ? '#ef4444' : '#3b82f6'}`,
                    pointerEvents: 'none',
                  }}
                />
              )}
            </button>
          )
        })}
      </div>
    </div>
  )
}
