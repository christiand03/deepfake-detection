"""Synchronisations- und Sensitivitäts-Tests für CrossAttentionFusion.

Prüft drei Ebenen:
  1. Daten-Ebene   — Audio- und Video-Chunk decken exakt dieselbe Zeitspanne ab
  2. Modell-Ebene  — Output reagiert auf Audio-Änderungen (kein "taubes" Modell)
  3. Attention-Ebene — Attention-Weights sind nicht uniform (das Modell "schaut" irgendwo hin)

Ausführen:
    python tests/test_synchronization.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from src.models.multimodal_module import CrossAttentionFusion, MultimodalDeepfakeModule

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Preprocessing-Konstanten (müssen zur preprocess.yaml passen) ──────────────
NUM_FRAMES              = 16
TARGET_FPS              = 25
AUDIO_SAMPLES_PER_CHUNK = 10_240
SAMPLE_RATE             = 16_000

# ── Modell-Konstanten ─────────────────────────────────────────────────────────
VIDEO_DIM  = 768
AUDIO_DIM  = 768
FUSION_DIM = 512
NUM_HEADS  = 8
BATCH      = 2

print(f"\n{'='*60}")
print(f"  Device: {DEVICE}")
print(f"{'='*60}\n")


# ── Test 1: Daten-Ebene — Zeitspannen-Alignment ───────────────────────────────

def test_chunk_duration_alignment():
    """Video-Chunk-Dauer == Audio-Chunk-Dauer (Preprocessing-Garantie).

    Wenn das nicht stimmt, attendieren die Transformer Tokens aus
    unterschiedlichen Zeitfenstern — egal wie gut die Attention ist.
    """
    print("Test 1: Daten-Ebene — Zeitspannen-Alignment")

    video_duration_s = NUM_FRAMES / TARGET_FPS
    audio_duration_s = AUDIO_SAMPLES_PER_CHUNK / SAMPLE_RATE

    print(f"  Video-Chunk: {NUM_FRAMES} Frames @ {TARGET_FPS} fps = {video_duration_s:.4f}s")
    print(f"  Audio-Chunk: {AUDIO_SAMPLES_PER_CHUNK} Samples @ {SAMPLE_RATE} Hz = {audio_duration_s:.4f}s")

    assert abs(video_duration_s - audio_duration_s) < 1e-6, (
        f"Zeitspannen stimmen NICHT überein: Video={video_duration_s:.4f}s, Audio={audio_duration_s:.4f}s\n"
        f"  → Preprocessing-Konstanten prüfen: NUM_FRAMES, TARGET_FPS, AUDIO_SAMPLES_PER_CHUNK, SAMPLE_RATE"
    )
    print(f"  ✓ Beide Chunks decken exakt {video_duration_s:.4f}s ab")


# ── Test 2: Modell-Ebene — Sensitivität gegenüber Audio-Änderungen ────────────

def test_audio_sensitivity():
    """Output muss sich ändern, wenn Audio ausgetauscht wird (bei gleichem Video).

    Wenn sich nichts ändert → Cross-Attention ignoriert Audio komplett.
    Das wäre ein stilles Versagen: Das Modell würde trainieren, aber nur
    Video-Features nutzen.
    """
    print("\nTest 2: Modell-Ebene — Sensitivität gegenüber Audio")

    fusion = CrossAttentionFusion(
        video_dim=VIDEO_DIM, audio_dim=AUDIO_DIM,
        fusion_dim=FUSION_DIM, num_heads=NUM_HEADS,
    ).to(DEVICE).eval()

    video_h  = torch.randn(BATCH, 1569, VIDEO_DIM, device=DEVICE)
    audio_a  = torch.randn(BATCH, 32, AUDIO_DIM, device=DEVICE)
    audio_b  = torch.randn(BATCH, 32, AUDIO_DIM, device=DEVICE)  # komplett anderes Audio

    with torch.no_grad():
        logits_a = fusion(video_h, audio_a)
        logits_b = fusion(video_h, audio_b)

    diff = (logits_a - logits_b).abs().mean().item()
    assert diff > 1e-6, (
        f"Output hat sich bei anderem Audio NICHT geändert (diff={diff:.2e})!\n"
        "  → Cross-Attention ignoriert Audio — Implementierung prüfen."
    )
    print(f"  ✓ Output ändert sich bei anderem Audio  (mean |Δlogits| = {diff:.4f})")


# ── Test 3: Modell-Ebene — Sensitivität gegenüber Video-Änderungen ────────────

def test_video_sensitivity():
    """Analog zu Test 2 für Video — stellt sicher dass auch Video-Branch aktiv ist."""
    print("\nTest 3: Modell-Ebene — Sensitivität gegenüber Video")

    fusion = CrossAttentionFusion(
        video_dim=VIDEO_DIM, audio_dim=AUDIO_DIM,
        fusion_dim=FUSION_DIM, num_heads=NUM_HEADS,
    ).to(DEVICE).eval()

    audio_h  = torch.randn(BATCH, 32, AUDIO_DIM, device=DEVICE)
    video_a  = torch.randn(BATCH, 1569, VIDEO_DIM, device=DEVICE)
    video_b  = torch.randn(BATCH, 1569, VIDEO_DIM, device=DEVICE)

    with torch.no_grad():
        logits_a = fusion(video_a, audio_h)
        logits_b = fusion(video_b, audio_h)

    diff = (logits_a - logits_b).abs().mean().item()
    assert diff > 1e-6, (
        f"Output hat sich bei anderem Video NICHT geändert (diff={diff:.2e})!\n"
        "  → Cross-Attention ignoriert Video — Implementierung prüfen."
    )
    print(f"  ✓ Output ändert sich bei anderem Video  (mean |Δlogits| = {diff:.4f})")


# ── Test 4: Attention-Ebene — Weights nicht uniform ──────────────────────────

def test_attention_weights_not_uniform():
    """Attention-Weights dürfen nicht gleichmäßig über alle Tokens verteilt sein.

    Uniform = 1/T für alle Tokens → Attention verhält sich wie Mean-Pooling
    → kein Vorteil gegenüber einfacher Konkatenation.

    Direkt nach der Initialisierung mit Zufallsgewichten sind die Weights
    leicht nicht-uniform durch die Projektion — nach dem Training sollten
    sie deutlich spitzer werden.
    """
    print("\nTest 4: Attention-Ebene — Attention-Weights nicht uniform")

    # Wir patchen MultiheadAttention, um die Weights abzugreifen
    class FusionWithWeights(CrossAttentionFusion):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.last_v2a_weights = None
            self.last_a2v_weights = None

        def forward(self, video_hidden, audio_hidden):
            v = self.video_proj(video_hidden)
            a = self.audio_proj(audio_hidden)

            v_cross, v2a_w = self.v_to_a_attn(query=v, key=a, value=a,
                                               need_weights=True, average_attn_weights=True)
            self.last_v2a_weights = v2a_w.detach()   # (B, T_v, T_a)
            v = self.v_to_a_norm(v + v_cross)

            a_cross, a2v_w = self.a_to_v_attn(query=a, key=v, value=v,
                                               need_weights=True, average_attn_weights=True)
            self.last_a2v_weights = a2v_w.detach()   # (B, T_a, T_v)
            a = self.a_to_v_norm(a + a_cross)

            v_pool = v.mean(dim=1)
            a_pool = a.mean(dim=1)
            return self.classifier(torch.cat([v_pool, a_pool], dim=1))

    fusion = FusionWithWeights(
        video_dim=VIDEO_DIM, audio_dim=AUDIO_DIM,
        fusion_dim=FUSION_DIM, num_heads=NUM_HEADS,
    ).to(DEVICE).eval()

    video_h = torch.randn(BATCH, 1569, VIDEO_DIM, device=DEVICE)
    audio_h = torch.randn(BATCH, 32, AUDIO_DIM, device=DEVICE)

    with torch.no_grad():
        fusion(video_h, audio_h)

    v2a_w = fusion.last_v2a_weights  # (B, T_v, T_a)
    a2v_w = fusion.last_a2v_weights  # (B, T_a, T_v)

    # Uniform-Referenz: alle Weights = 1/T
    t_a = v2a_w.shape[-1]
    t_v = a2v_w.shape[-1]
    uniform_v2a = torch.full_like(v2a_w, 1.0 / t_a)
    uniform_a2v = torch.full_like(a2v_w, 1.0 / t_v)

    # KL-Divergenz zur Uniform-Verteilung (höher = weniger uniform = besser)
    kl_v2a = F.kl_div(v2a_w.log().clamp(min=-100), uniform_v2a, reduction="batchmean").item()
    kl_a2v = F.kl_div(a2v_w.log().clamp(min=-100), uniform_a2v, reduction="batchmean").item()

    # Max-Abweichung von uniform
    max_dev_v2a = (v2a_w - uniform_v2a).abs().max().item()
    max_dev_a2v = (a2v_w - uniform_a2v).abs().max().item()

    assert max_dev_v2a > 1e-5, "V→A Attention-Weights sind vollständig uniform!"
    assert max_dev_a2v > 1e-5, "A→V Attention-Weights sind vollständig uniform!"

    print(f"  ✓ V→A Attention: max Abweichung von uniform = {max_dev_v2a:.6f}  (KL = {kl_v2a:.2e})")
    print(f"  ✓ A→V Attention: max Abweichung von uniform = {max_dev_a2v:.6f}  (KL = {kl_a2v:.2e})")
    print()
    print("  Hinweis: Nach dem Training sollten diese Werte deutlich größer sein.")
    print("  Sehr kleine Werte nach dem Training → Attention kollabiert zu Mean-Pooling.")


# ── Test 5: Asynchronitäts-Nachweis (bekannte Limitation dokumentieren) ───────

def test_document_token_level_behavior():
    """Dokumentiert das bewusste Design: Token-Level-Attention ist global (nicht temporal).

    Was das bedeutet:
      - Chunk-Ebene:  Video und Audio decken exakt dieselbe Zeitspanne ab (Test 1).
      - Token-Ebene:  Jeder Video-Token kann jeden Audio-Token attendieren.
                      Es gibt keine erzwungene "Frame i ↔ Audio-Segment i"-Bindung.

    Das ist bei Multimodal-Transformern (CLIP, AV-HuBERT, etc.) Standard und
    ausreichend für die Frage "passt das Audio grundsätzlich zum Video?".

    Für frame-genaue Lippensync-Erkennung wäre lokale / temporale Attention
    (z.B. Sliding-Window) sinnvoller — das ist ein möglicher nächster Schritt.
    """
    print("\nTest 5: Token-Level-Attention — Verhalten dokumentieren")

    t_v = 1569   # VideoMAE-base Tokens (inkl. CLS)
    t_a = 32     # Wav2Vec2-base Tokens für 10240 Samples

    video_duration_s = NUM_FRAMES / TARGET_FPS           # 0.64s
    audio_duration_s = AUDIO_SAMPLES_PER_CHUNK / SAMPLE_RATE  # 0.64s

    # Zeitauflösung pro Token
    # VideoMAE: tubelet_size=2 → 8 Zeitschritte, 14×14=196 spatial patches
    # → nicht sinnvoll einen Token einem Frame zuzuordnen
    audio_token_duration_ms = (audio_duration_s / t_a) * 1000

    print(f"  Video-Tokens:          {t_v} (inkl. CLS, über {video_duration_s:.2f}s)")
    print(f"  Audio-Tokens:          {t_a} (je ~{audio_token_duration_ms:.1f}ms)")
    print(f"  Attention-Typ:         global (jeder Token → jeden Token)")
    print(f"  Chunk-Alignment:       ✓ (beide Fenster = {video_duration_s:.2f}s)")
    print(f"  Token-Alignment:       — (global, kein Forcing)")
    print()
    print("  → Ausreichend für Deepfake-Erkennung auf Chunk-Ebene.")
    print("  → Für frame-genaue Lippensync: temporale Attention erwägen.")
    assert True  # Dieser Test ist immer grün — er dokumentiert nur das Verhalten.


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_chunk_duration_alignment,
        test_audio_sensitivity,
        test_video_sensitivity,
        test_attention_weights_not_uniform,
        test_document_token_level_behavior,
    ]

    passed, failed = 0, 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FEHLER: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Ergebnis: {passed}/{len(tests)} Tests bestanden", end="")
    print("  ✓" if failed == 0 else f"  — {failed} fehlgeschlagen")
    print(f"{'='*60}\n")
    sys.exit(0 if failed == 0 else 1)