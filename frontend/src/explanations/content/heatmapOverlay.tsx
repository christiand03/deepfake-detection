/**
 * Explanation content — Relevance-Heatmap-Overlay (video player).
 *
 * EDIT THIS FILE to change the text. Prose is German; domain terms stay English
 * (heatmap, relevance, confidence, chunk, magnitude, direction, …). Technical
 * numbers follow docs/xai_pipeline_reference.md §2.1, §3, §4.1, §4.4, §6.1.
 */

import type { Explanation } from '../types'
import {
  BivariateLrpNote,
  Callout,
  ChunkStrip,
  Chip,
  ColorScaleLegend,
  KeyValueList,
  P,
  Term,
  UL,
} from '../ui/widgets'

export const heatmapOverlay: Explanation = {
  id: 'heatmap-overlay',
  title: 'Relevance-Heatmap-Overlay',
  subtitle:
    'Die rot/blauen Patches über dem Gesicht zeigen, wo das Modell Evidenz gefunden hat — und wohin diese Evidenz lehnt (fake vs. real).',
  method: 'AttnLRP (dual-seed) · bivariate LRP · VideoMAE',
  cvr: 'relevance',
  sections: [
    {
      kind: 'what',
      body: (
        <>
          <P>
            Ein halbtransparentes Overlay, das deckungsgleich über dem laufenden
            Video liegt — eine pro Pixel berechnete{' '}
            <Term color="#00e5ff">Relevance-Heatmap</Term>, wie stark und in welche
            Richtung ein Bildbereich die Entscheidung des Modells beeinflusst hat.
          </P>
          <P>
            Die Heatmap ist <Term>bivariat</Term> — sie trägt zwei entkoppelte
            Kanäle gleichzeitig:
          </P>
          <KeyValueList
            items={[
              {
                k: 'magnitude',
                v: (
                  <>
                    <Term>Engagement</Term> — wie stark das Modell hier überhaupt
                    gearbeitet hat. Steuert die <Chip>Deckkraft</Chip> (0 =
                    unsichtbar/transparent).
                  </>
                ),
              },
              {
                k: 'direction',
                v: (
                  <>
                    <Term>Lean</Term> — wohin die Evidenz zeigt. Steuert die{' '}
                    <Chip color="#ff7070">Farbe</Chip> (rot = fake, blau = real).
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
          Beantwortet die Frage <Term>„WO im Gesicht schaut das Modell hin — und
          spricht das dort Gesehene für Fake oder für Real?"</Term>. Das ist der
          Kern des „Depth-over-Breadth"-Ansatzes: nicht nur <em>ob</em> Fake,
          sondern <em>warum</em>.
        </P>
      ),
    },
    {
      kind: 'method',
      body: (
        <>
          <P>
            Grundlage ist <Term color="#00e5ff">AttnLRP</Term> (Attention-Aware
            Layer-wise Relevance Propagation, Achtibat et al., ICML 2024) auf dem
            VideoMAE-Backbone, in der <Chip>Input × Gradient</Chip>-Formulierung.
          </P>
          <BivariateLrpNote unit="pro 16-Frame-Chunk (0,64 s), clip-global normiert" />
          <P>
            Die 224×224-Heatmap wird anschließend per{' '}
            <Term>Upprojektion</Term> pro 16-Frame-Chunk mit der{' '}
            <Term>jeweils mitwandernden</Term> Face-Box ins Vollbild gesetzt — sie
            folgt also dem bewegten Gesicht. Pixel außerhalb des Crops sind exakt 0
            → voll transparent, daher keine sichtbare Rechteck-Kante.
          </P>
        </>
      ),
    },
    {
      kind: 'legend',
      body: (
        <>
          <ColorScaleLegend leftLabel="fake-stützend (rot)" rightLabel="real-stützend (blau)" />
          <KeyValueList
            items={[
              { k: 'Rot', v: 'Bildbereich liefert Evidenz FÜR Fake.' },
              { k: 'Blau', v: 'Bildbereich liefert Evidenz FÜR Real.' },
              {
                k: 'Deckkraft',
                v: 'Wie stark das Modell hier engagiert war (magnitude).',
              },
              {
                k: 'Transparent',
                v: 'Kein nennenswertes Engagement — das Modell hat diesen Bereich ignoriert.',
              },
            ]}
          />
        </>
      ),
    },
    {
      kind: 'normalization',
      body: (
        <>
          <P>
            Beide Kanäle werden <Term>clip-global</Term> mit einer{' '}
            <Chip>Perzentil-Normierung (99 %)</Chip> skaliert — nicht mit dem
            globalen Abs-Maximum. Vorteil: Ein einzelner Ausreißer-Pixel drückt
            nicht mehr den ganzen Rest auf ~0; seltene Extremwerte werden bei ±1
            gekappt, der Bulk bleibt sichtbar. „Clip-global" heißt: alle Chunks
            teilen dieselbe Skala, ein schwacher Chunk wird also nicht künstlich
            angehoben.
          </P>
          <Callout variant="info" title="Skalen-Reichweite: hier IST volle Deckkraft erreichbar">
            Wichtig, weil es dieses Visual von den Timelines/Region-Aggregaten
            unterscheidet: Die Norm läuft <Term>pro Pixel</Term>, nicht als
            Mittelwert. Die stärksten Pixel (oberstes Perzentil) treffen daher das
            Skalen-Maximum → <Term>volle Deckkraft ist erreichbar</Term>. Bei der
            Chunk-Timeline und dem Region-Schema wird dagegen über ganze
            Frames/Regionen <em>gemittelt</em>, sodass die Werte klein bleiben und
            ~1 nie erreichen. Absolut bleibt die Deckkraft trotzdem{' '}
            <em>clip-relativ</em>: „der stärkste Patch dieses Clips", kein
            Cross-Clip-Wert. Und ein Zahlenwert wird nicht ausgegeben — das Signal
            lebt allein in Deckkraft (magnitude) und Farbe (direction).
          </Callout>
          <Callout variant="info" title="Dead-Zone (bewusst)">
            Schwach <em>gerichtete</em> Pixel (die von Frame zu Frame zwischen rot
            und blau kippen würden) werden über eine <Chip>direction-Gamma</Chip>{' '}
            von 1,6 auf near-neutral/weiß gedämpft, statt zu flackern. Die
            Deckkraft nutzt <Chip>alpha-Gamma 0,5</Chip> (hebt schwache, aber
            nicht-null Bereiche an) <Term>ohne Alpha-Floor</Term> — unbeachtete
            Frames bleiben transparent, statt eine Sichtbarkeit zu erhalten.
          </Callout>
        </>
      ),
    },
    {
      kind: 'resolution',
      body: (
        <>
          <P>
            Räumlich: intern 224×224, per Bilinear-Upsample ins Vollbild — feine
            Strukturen wirken also leicht weichgezeichnet. Zeitlich wird die
            Heatmap pro nicht-überlappendem <Term>16-Frame-Chunk</Term>{' '}
            berechnet:
          </P>
          <ChunkStrip chunks={6} highlight={2} />
          <P>
            Ein Chunk ≈ 0,64 s. Eine sehr kurze Manipulation (z. B. 0,16 s) liegt
            damit vollständig in <em>einem</em> Chunk und teilt sich dessen
            gemittelte Heatmap — sie kann also über den ganzen Chunk „verschmieren".
          </P>
        </>
      ),
    },
    {
      kind: 'interpret',
      body: (
        <UL>
          <li>
            <Term color="#ff7070">Kräftig roter Patch</Term> auf einer Region →
            starkes, fake-stützendes Signal genau dort.
          </li>
          <li>
            <Term color="#5e91ee">Blauer Patch</Term> → die Region drückt aktiv
            Richtung real (z. B. konsistente, unmanipulierte Bildmerkmale).
          </li>
          <li>
            <Term>Transparent / farblos</Term> → das Modell hat hier kaum
            gearbeitet. Das ist eine Aussage über Aufmerksamkeit, nicht über
            Echtheit.
          </li>
          <li>
            Deckkraft lesen als „wie sicher engagiert", Farbe als „in welche
            Richtung" — beides getrennt betrachten.
          </li>
        </UL>
      ),
    },
    {
      kind: 'reading',
      body: (
        <Callout variant="tip" title="Beispiel">
          Ein heller roter Fleck sitzt bei jedem gesprochenen Wort auf der
          Mundpartie, der Rest des Gesichts bleibt blass. → Das Modell stützt sein
          Fake-Urteil auf die Mund-/Lippen-Region — ein typisches Muster für
          Lip-Sync-/Reenactment-Artefakte. Wandert der Fleck stattdessen unruhig an
          den Bildrand oder in den Hintergrund, ist das Signal wenig fundiert.
        </Callout>
      ),
    },
    {
      kind: 'gain',
      body: (
        <KeyValueList
          items={[
            { k: 'Lokalisierung', v: 'Welche Gesichtsregion die Entscheidung trägt.' },
            { k: 'Richtung', v: 'Ob eine Region fake- oder real-stützend wirkt.' },
            { k: 'Plausibilität', v: 'Ob das Modell auf sinnvolle Stellen (Mund/Augen) statt auf Hintergrund schaut.' },
            { k: 'Zeitbezug', v: 'Zusammen mit dem Playhead: wann sich das Muster ändert.' },
          ]}
        />
      ),
    },
    {
      kind: 'pitfalls',
      body: (
        <>
          <Callout variant="warn" title="Transparent ≠ real">
            Ein durchsichtiger Bereich bedeutet <em>„hier kein Engagement"</em> —
            Abwesenheit von Evidenz, kein Real-Beweis.
          </Callout>
          <Callout variant="warn" title="Blau ≠ echter Inhalt">
            Blau bedeutet <em>„drückt Richtung real"</em> im Sinne des
            Klassifikations-Gradienten — nicht, dass der Bildinhalt garantiert
            unmanipuliert ist.
          </Callout>
          <Callout variant="warn" title="Helligkeit ≠ Korrektheit">
            Deckkraft = Engagement, nicht Trefferwahrscheinlichkeit. Ein starkes
            Signal an falscher Stelle ist trotzdem hell. Und weil die Skala relativ
            (perzentil, clip-global) ist, heißt „der hellste Patch im Clip" nicht
            automatisch „absolut stark".
          </Callout>
        </>
      ),
    },
    {
      kind: 'trust',
      body: (
        <UL>
          <li>
            <Term color="#22c55e">Verlässlich</Term>, wenn die Manipulation räumlich
            konzentriert ist und die Heatmap stabil auf derselben Region sitzt.
          </li>
          <li>
            <Term color="#f59e0b">Vorsichtig</Term> bei sehr kurzen/kleinen
            Manipulationen: sie verschmieren über den 0,64-s-Chunk und über die
            Upprojektion.
          </li>
          <li>
            Setzt voraus, dass die Face-Box dem Gesicht folgt. Bei starker
            Kopf-Rotation kann die Zuordnung leiden (siehe Rotation-Warning).
          </li>
        </UL>
      ),
    },
    {
      kind: 'limitations',
      body: (
        <UL>
          <li>
            Zeigt <Term>Relevance</Term>, nicht <Term>Confidence</Term>: sagt „wo &
            wohin", aber nicht „wie fake ist der Clip insgesamt" — dafür sind
            Verdict-Gauges und die Confidence-Timeline da.
          </li>
          <li>
            Perzentil-Normierung ist relativ zum Clip — Heatmaps verschiedener
            Clips sind in der absoluten Intensität nicht 1:1 vergleichbar.
          </li>
          <li>Bilinear-Upsampling glättet feine Kanten.</li>
        </UL>
      ),
    },
    {
      kind: 'interaction',
      body: (
        <UL>
          <li>
            <Chip>Opacity-Slider</Chip> (oben rechts am Player) blendet das Overlay
            stufenlos ein/aus — zum Vergleich mit dem rohen Bild.
          </li>
          <li>Beim Abspielen/Scrubben wechselt die Heatmap frameweise mit.</li>
          <li>
            <Chip color="#f59e0b">Heatmap-Methode</Chip> schaltet zwischen drei
            Darstellungen um — <strong>ausschließlich für dieses Overlay</strong>.
            Verdict, Confidence- und Relevance-Timeline, Region-Scores und Phase 3/4
            laufen unverändert auf Bivariate-LRP weiter, auch wenn hier umgestellt ist.
          </li>
        </UL>
      ),
    },
    {
      kind: 'method',
      title: 'Die drei Heatmap-Methoden (Ablation)',
      body: (
        <>
          <KeyValueList
            items={[
              {
                k: 'Bivariate LRP',
                v: (
                  <>
                    Der Default. <Term>Magnitude</Term> (Deckkraft) und{' '}
                    <Term color="#ff7070">Direction</Term> (rot/blau) zugleich — zwei
                    Achsen, wie oben beschrieben.
                  </>
                ),
              },
              {
                k: 'LRP — nur Magnitude',
                v: (
                  <>
                    Dieselbe AttnLRP-Rechnung, aber die Richtungsachse wird verworfen.
                    Eine sequenzielle Colormap statt rot/blau, weil eine reine
                    Magnitude-Karte kein Vorzeichen hat.
                  </>
                ),
              },
              {
                k: 'Chefer et al.',
                v: (
                  <>
                    <Term color="#f59e0b">Andere Methode</Term>, nicht nur andere
                    Darstellung: gradienten-gewichtetes Attention-Rollout (ICCV 2021),
                    das keine Zeile Code mit unserem LRP-Pfad teilt. Bauartbedingt
                    nicht-negativ und breiter gestreut.
                  </>
                ),
              },
            ]}
          />
          <Callout>
            Warum drei Stufen und nicht zwei: Der Schritt 1 → 2 zeigt, was das Weglassen
            der Richtungsachse ausmacht, der Schritt 2 → 3 den Unterschied der Methode.
            Ein Zweifach-Schalter würde beides gleichzeitig ändern — man könnte dann
            nicht sagen, woher ein sichtbarer Unterschied kommt.
          </Callout>
          <P>
            Gemessen gegen die Ground-Truth-Manipulationsmasken zeigen beide Methoden
            dieselbe Verbesserung der Lokalisierung nach dem
            Relevance-Regularization-Training. Da Chefer nicht die Größe ist, auf die
            trainiert wurde, ist das der Teil der Verbesserung, der über die optimierte
            Metrik hinaus generalisiert.
          </P>
        </>
      ),
    },
    {
      kind: 'links',
      body: (
        <UL>
          <li>
            <Term color="#a855f7">Chunk-Timelines</Term>: nutzen exakt dieselben
            16-Frame-Chunks — die Relevance-Spur ist die zeitliche Verdichtung
            dieser Heatmap.
          </li>
          <li>
            <Term color="#a855f7">Region-Relevance (Gesichts-Schema)</Term>:
            aggregiert dieselbe Heatmap auf anatomische Regionen (Mund, Augen …).
          </li>
          <li>
            <Term color="#a855f7">Verdict-Gauges</Term>: die komplementäre
            Confidence-Sicht (WAS/WIE-fake) zu diesem WO/WARUM.
          </li>
          <li>
            <Term color="#a855f7">Attention-Shift (Phase 3/4)</Term>: vergleicht
            diese Region-Scores vor/nach Degradation bzw. Angriff.
          </li>
        </UL>
      ),
    },
  ],
}
