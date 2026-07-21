/**
 * Explanation content — Chunk-Timelines (Confidence + Relevance, unter dem Player).
 * Quellen: docs/xai_pipeline_reference.md §3.3, §6.2; ChunkTimelines.tsx.
 */

import type { Explanation } from '../types'
import {
  BivariateLrpNote,
  Callout,
  Chip,
  ColorScaleLegend,
  KeyValueList,
  P,
  RelevanceScaleNote,
  Term,
  UL,
} from '../ui/widgets'

export const chunkTimelines: Explanation = {
  id: 'chunk-timelines',
  title: 'Chunk-Timelines',
  subtitle:
    'Zwei gestapelte Zeitachsen über den ganzen Clip: oben die Confidence pro Chunk, unten die Relevance pro Chunk.',
  method: 'Per-Window Fake-Prob · bivariate LRP · gemeinsamer Playhead',
  cvr: 'both',
  sections: [
    {
      kind: 'what',
      body: (
        <>
          <P>
            Zwei übereinanderliegende Spuren, beide über die volle Cliplänge, mit
            einem gemeinsamen cyan Playhead:
          </P>
          <KeyValueList
            items={[
              {
                k: 'CONFIDENCE',
                v: (
                  <>
                    Die Klassifikation <Term>jedes einzelnen Chunks</Term> (REAL ↔
                    FAKE).
                  </>
                ),
              },
              {
                k: 'RELEVANCE',
                v: (
                  <>
                    Der Einfluss auf das Ergebnis (magnitude),
                    Balken-<Term>Farbe</Term> = Richtung (rot = fake-, blau =
                    real-stützend).
                  </>
                ),
              },
            ]}
          />
        </>
      ),
    },
    {
      kind: 'purpose',
      body: (
        <P>
          Beantwortet <Term>„WANN im Clip reagiert das Modell?"</Term>. Die
          Confidence-Spur lokalisiert den Zeitpunkt einer Manipulation (ein kurzer
          Fake-Abschnitt erscheint nur dort als FAKE); die Relevance-Spur zeigt,
          welche Chunks die Gesamtentscheidung am stärksten getragen haben und in
          welche Richtung.
        </P>
      ),
    },
    {
      kind: 'method',
      body: (
        <>
          <P>
            Pro 16-Frame-Chunk liefert das Modell eine{' '}
            <Chip>Fake-Wahrscheinlichkeit</Chip> (Confidence-Spur). Die
            Relevance-Timeline kommt aus derselben bivariaten Dual-Seed-LRP wie die
            Heatmap. Wichtig: <Chip>Höhe = mean magnitude</Chip> pro Chunk{' '}
          </P>
          <BivariateLrpNote compact unit="pro 16-Frame-Chunk (Höhe = magnitude, Farbe = direction)" />
        </>
      ),
    },
    {
      kind: 'legend',
      body: (
        <>
          <ColorScaleLegend />
          <KeyValueList
            items={[
              { k: 'Mittellinie', v: '0,5 in der Confidence-Spur — darüber FAKE (rot), darunter REAL (blau).' },
              { k: 'Balkenhöhe', v: 'Relevance-Stärke des Chunks (magnitude).' },
              { k: 'Balkenfarbe', v: 'Richtung der Netto-Relevanz (fake vs. real).' },
            ]}
          />
        </>
      ),
    },
    {
      kind: 'normalization',
      body: (
        <P>
          Die Relevanzwerte pro Chunk sind absolut klein, weil sie über das ganze
          Bild gemittelt werden. Damit die Unterschiede sichtbar werden, wird die
          Balkenhöhe gleichmäßig mit dem Faktor 4 gestreckt. Dieser Faktor ist für
          alle Balken gleich, die Verhältnisse untereinander bleiben also erhalten.
          Weil die Streckung fest ist, sind die Balkenhöhen nur innerhalb eines Clips
          vergleichbar, nicht zwischen verschiedenen Clips.
        </P>
      ),
    },
    {
      kind: 'resolution',
      body: (
        <>
          <P>
            Ein Punkt/Balken = ein 16-Frame-Chunk ≈ 0,64 s. Confidence- und
            Relevance-Timeline beachten alle Frames. Wenn die letzten Frames eines
            Clips nicht mehr zu einem vollständigen Chunk kombiniert werden können wird
            der letzte Frame wiederholt um die benötigten 16 Frames zu füllen. Dies ist ein
            Kompromiss, da VideoMAE immer 16 Frame Chunks benötigt und echte (die letzten Frames)
            nicht verworfen werden sollen. Die Alternative wäre für den letzten Chunk die letzten
            16 Frames des Clips als separaten Chunk zu verarbeiten und doppelte Frames zu droppen.
            Das verlagert aber nur das Problem trotz eines komplexeren Ansatzes.
          </P>
        </>
      ),
    },
    {
      kind: 'interpret',
      body: (
        <>
          <RelevanceScaleNote
            frame={
              <>
                Hier steht ein Wert je 16-Frame-Chunk: wie stark dieser Chunk die
                Entscheidung beeinflusst hat. Weil er über das ganze Bild gemittelt
                wird, bleibt er klein — auch die wichtigsten Chunks liegen deutlich
                unter 1. Es zählt daher nicht der absolute Wert, sondern der Vergleich
                der Balken: Ein höherer Balken hatte mehr Einfluss, seine Farbe zeigt
                die Richtung (rot = fake, blau = real). Die Höhe ist nur zur besseren
                Sichtbarkeit gestreckt; die Prozentzahl im Tooltip sagt, wie voll der
                Balken ist, nicht wie viel Aufmerksamkeit er bekommt. Die
                Confidence-Spur darüber ist etwas anderes: eine echte
                Wahrscheinlichkeit von 0 bis 100 %.
              </>
            }
          />
          <UL>
            <li>
              Ein kurzer <Term color="#ff7070">roter Ausschlag</Term> über der
              Mittellinie → ein lokal manipulierter Abschnitt; der Rest bleibt real.
            </li>
            <li>
              Ein hoher Relevance-Balken → dieser Chunk war einflussreich; die Farbe
              sagt, ob fake- oder real-stützend.
            </li>
            <li>Beide Spuren am Playhead ablesen, um Zeitpunkt und Einfluss zu koppeln.</li>
          </UL>
        </>
      ),
    },
    {
      kind: 'reading',
      body: (
        <Callout variant="tip" title="Beispiel">
          Die Confidence bleibt blau (REAL), springt bei Sekunde 3,3 kurz über die
          Mittellinie ins Rot und fällt danach zurück → genau dort sitzt die
          Manipulation. Der Relevance-Balken an derselben Stelle ist hoch und rot →
          dieser Chunk trägt das Fake-Urteil.
        </Callout>
      ),
    },
    {
      kind: 'gain',
      body: (
        <KeyValueList
          items={[
            { k: 'Zeitpunkt', v: 'Wann eine Manipulation auftritt (Confidence-Spur).' },
            { k: 'Einfluss', v: 'Welche Chunks die Entscheidung tragen (Relevance-Höhe).' },
            { k: 'Richtung', v: 'Ob ein Chunk fake- oder real-stützend wirkt (Relevance-Farbe).' },
          ]}
        />
      ),
    },
    {
      kind: 'pitfalls',
      body: (
        <>
          <Callout variant="warn" title="Per-Chunk-Confidence ≠ Gesamt-Verdict">
            Das Panel zeigt jeden Chunk einzeln. Der finale Verdict (Gauge) ist ein{' '}
            <em>Max-Pool</em> über alle Chunks — ein einzelner roter Ausschlag reicht
            dort für FAKE.
          </Callout>
          <Callout variant="warn" title="Balkenhöhe ist skaliert">
            Die Balkenhöhe ist zur Sichtbarkeit gestreckt und kann voll ausschlagen,
            obwohl der Relevanzwert dahinter klein ist. Die Balken daher nur
            untereinander vergleichen, nicht als absolute Größe lesen.
          </Callout>
        </>
      ),
    },
    {
      kind: 'interaction',
      body: (
        <UL>
          <li>Gemeinsamer Playhead läuft mit dem Video mit.</li>
          <li>
            <Chip>Hover</Chip> über einen Chunk zeigt Fake-Prob bzw. Relevance-Stärke
            und Richtung exakt.
          </li>
        </UL>
      ),
    },
    {
      kind: 'links',
      body: (
        <UL>
          <li>
            <Term color="#a855f7">Heatmap-Overlay</Term>: dieselben 16-Frame-Chunks
            — die Relevance-Spur ist deren Aggregation.
          </li>
          <li>
            <Term color="#a855f7">Verdict-Gauges</Term>: der Max-Pool der
            Confidence-Spur.
          </li>
          <li>
            <Term color="#a855f7">Region-Relevance</Term>: die räumliche Aggregation
            derselben Relevanz.
          </li>
        </UL>
      ),
    },
  ],
}
