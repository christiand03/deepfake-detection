/**
 * SectionBlock — renders one explanation section with a consistent iconed
 * header. Every explanation reuses this, so all popups read the same way.
 */

import { SECTION_META, type ExplanationSection } from '../types'

export function SectionBlock({ section }: { section: ExplanationSection }) {
  const meta = SECTION_META[section.kind]
  const title = section.title ?? meta.label
  return (
    <section style={{ marginBottom: 16 }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          marginBottom: 6,
        }}
      >
        <span
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: 18,
            height: 18,
            borderRadius: 5,
            backgroundColor: `${meta.color}18`,
            border: `1px solid ${meta.color}44`,
            color: meta.color,
            fontSize: 11,
            lineHeight: 1,
            flexShrink: 0,
          }}
        >
          {meta.glyph}
        </span>
        <h3
          style={{
            margin: 0,
            fontFamily: 'monospace',
            fontSize: 11,
            fontWeight: 700,
            letterSpacing: '0.1em',
            textTransform: 'uppercase',
            color: '#e8eaf0',
          }}
        >
          {title}
        </h3>
        <div style={{ flex: 1, height: 1, backgroundColor: '#1e2233' }} />
      </div>
      <div style={{ paddingLeft: 26 }}>{section.body}</div>
    </section>
  )
}
