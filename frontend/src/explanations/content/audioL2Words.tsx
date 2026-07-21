/**
 * Explanation content — L2 Word-Tokens (Audio).
 * Quellen: docs/xai_pipeline_reference.md §7.2; WordTokenChart.tsx; xai.md §3.
 */

import type { Explanation } from '../types'
import {
  BivariateLrpNote,
  Callout,
  Chip,
  ColorScaleLegend,
  DeadzoneNote,
  Formula,
  P,
  RelevanceScaleNote,
  Term,
  UL,
} from '../ui/widgets'

export const audioL2Words: Explanation = {
  id: 'audio-l2-words',
  title: 'L2 — Word-Tokens',
  subtitle:
    'Ein Balken pro gesprochenem Wort, rot/blau nach Relevanz — beantwortet, bei welchem Wort das Modell Manipulation vermutet.',
  method: 'WhisperX-Alignment · mean bivariate direction pro Wort',
  cvr: 'both',
  sections: [
    {
      kind: 'what',
      body: (
        <P>
          Ein Diverging-Bar-Chart: für jedes Wort ein Balken, dessen Länge und Farbe
          die Relevanz (bzw. Confidence) dieses Wortes kodieren. Das aktuell
          gesprochene Wort ist cyan umrandet.
        </P>
      ),
    },
    {
      kind: 'purpose',
      body: (
        <P>
          Beantwortet <Term>„Bei welchem Wort vermutet das Modell Manipulation?"</Term>.
        </P>
      ),
    },
    {
      kind: 'method',
      body: (
        <>
          <BivariateLrpNote compact unit="je gesprochenem Wort (mittlere direction)" />
          <P>
            Die Wort-Zeitstempel liefert <Term>WhisperX</Term> (Forced Aligner,
            offline gecacht). Pro Wort wird die mittlere bivariate Richtung über seine
            Samples genommen. Für die Anzeige (nur Relevance-Sicht) wird betont:
          </P>
          <Formula>{`emphasize(v) = sign(v) · min(1, |v|^2.5 · 1.8)`}</Formula>
        </>
      ),
    },
    {
      kind: 'legend',
      body: <ColorScaleLegend />,
    },
    {
      kind: 'normalization',
      body: (
        <DeadzoneNote present>
          <P>
            Die Betonung (Formel oben) wirkt <Term>multiplikativ auf den Wert</Term>.
            Das <Chip>Gamma 2,5</Chip> staucht kleine Werte überproportional: Das
            Rauschband schwacher Wörter (etwa 0,20–0,25) fällt auf ~0,03 und wird
            unsichtbar, während ein konzentriert manipuliertes Wort (~0,78) mit ~0,97
            groß bleibt. Der <Chip>Gain 1,8</Chip> hebt die Überlebenden an,{' '}
            <Chip>min(1, …)</Chip> kappt bei 1. Bewusst multiplikativ statt subtraktiv:
            Eine subtraktive Zone ließ dasselbe Band als schwache Balken stehen.
          </P>
          <P>
            Dadurch sind die Balkenwerte bewusst <em>nicht mehr die exakten
            Modellwerte</em> — L2 soll die auffälligen Wörter hervorheben, nicht deren
            exakte Höhe wiedergeben. Die Confidence-Sicht bleibt unberührt.
          </P>
        </DeadzoneNote>
      ),
    },
    {
      kind: 'resolution',
      body: (
        <P>
          Auflösung = ein Balken pro Wort. Der Wert eines Wortes ist der{' '}
          <Term>Mittelwert der Relevanz über seine Wav2Vec2-Feature-Frames</Term> —
          ein Frame entspricht dem 320-Sample-Stride des Encoders, also 20 ms
          (50 Frames/s). Ein Wort deckt je nach Länge 1 bis mehrere Dutzend Frames ab.
        </P>
      ),
    },
    {
      kind: 'trust',
      body: (
        <>
          <P>
            Der Balkenwert eines Wortes ist der <Term>Mittelwert der Relevanz über
            seine Feature-Frames</Term>. AttnLRP zerlegt die Fenster-Entscheidung in
            vorzeichenbehaftete Beiträge pro Frame; über <em>viele</em> Frames
            gleichen sich gegenläufige Beiträge aus, und es bleibt ein robustes
            Netto-Signal. Genau dieses Ausmitteln macht den Wert vertrauenswürdig.
          </P>
          <Callout variant="warn" title="Sehr kurze Wörter: Wert nicht belastbar">
            <P>
              Ein sehr kurzes Wort wie <Term>„a" (20 ms ≈ 1 Frame)</Term> hat kaum
              etwas zum Ausmitteln: sein Wert ist ein <em>einzelner, nicht
              ausgeglichener</em> Frame-Beitrag. Ein einzelner Frame kann zufällig
              stark ausschlagen, ohne dass sich benachbarte Beiträge dagegen
              aufheben — daher kann ein <em>hoher Balken auf einem 1–2-Frame-Wort
              reines Rauschen sein</em> und nicht echte Manipulations-Evidenz.
            </P>
            <P>
              Faustregel: Balken auf den kürzesten Funktionswörtern („a", „of", „an")
              nicht als Signal lesen. Erst ab mehreren Frames (~normale Wortlänge)
              trägt der Mittelwert genug Frames, um belastbar zu sein.
            </P>
          </Callout>
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
                Ein Balken pro Wort zeigt, wie stark das Modell das Wort für fake
                (rot) oder real (blau) hält. Höhe und Zahl sind bewusst nachgeschärft:
                schwache, verrauschte Wörter werden gedämpft, klar manipulierte treten
                hervor. Die Zahl ist deshalb eine bereinigte Stärke, nicht der exakte
                Messwert des Modells.
              </>
            }
          />
          <UL>
            <li>Ein hoher <Term color="#ff7070">roter</Term> Balken → das Modell hält dieses Wort für fake-verdächtig.</li>
            <li>Ein <Term color="#5e91ee">blauer</Term> Balken → real-stützend.</li>
            <li>Kurze Balken → unter dem Rauschband, kein klares Signal.</li>
          </UL>
        </>
      ),
    },
    {
      kind: 'pitfalls',
      body: (
        <Callout variant="warn" title="Kurzer Balken ≠ sauber">
          Ein Balken nahe null bedeutet „unter der Dead-Zone", nicht „garantiert
          echt". Und die Wort-Aggregation glättet Chunks, die in zwei Wörter fallen.
        </Callout>
      ),
    },
    {
      kind: 'limitations',
      body: (
        <P>
          Setzt ein Wort-Alignment voraus. Fehlt es (z. B. unklare Aussprache), zeigt
          der Layer „Word-level alignment unavailable".
        </P>
      ),
    },
    {
      kind: 'interaction',
      body: (
        <UL>
          <li>Der Chart ist horizontal scrollbar bei vielen Wörtern.</li>
          <li>Das aktive Wort wird beim Abspielen flüssig hervorgehoben (rAF).</li>
          <li><Chip>Hover</Chip> zeigt Wort und Wert.</li>
        </UL>
      ),
    },
    {
      kind: 'links',
      body: (
        <UL>
          <li>
            <Term color="#a855f7">L1 Waveform</Term>: dieselbe Relevanz, kontinuierlich
            über die Zeit.
          </li>
          <li>
            <Term color="#a855f7">L3 Frequency</Term>: welcher Artefakttyp (Band) statt
            welches Wort.
          </li>
        </UL>
      ),
    },
  ],
}
