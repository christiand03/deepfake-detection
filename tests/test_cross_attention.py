"""Smoke-Test für CrossAttentionFusion & MultimodalDeepfakeModule.

Prüft, ob:
  1. CrossAttentionFusion allein korrekte Shapes produziert
  2. MultimodalDeepfakeModule einen Forward-Pass übersteht
  3. Attention-Weights nicht degeneriert sind (keine NaN/Inf, nicht uniform)
  4. Gradienten durch den Fusion-Head fließen (Backprop funktioniert)
  5. Backbones wirklich eingefroren sind wenn freeze_backbone=True

Kein HDF5, kein Hydra, kein Training nötig.

Ausführen:
    python tests/test_cross_attention.py
"""

import sys

import pytest
import torch

from src.models.multimodal_module import CrossAttentionFusion, MultimodalDeepfakeModule

# ── Konstanten (müssen zur Preprocessing-Config passen) ───────────────────────
BATCH_SIZE = 2
NUM_FRAMES = 16
IMG_SIZE = 224
AUDIO_LEN = 10_240
VIDEO_DIM = 768  # VideoMAE-base hidden size
AUDIO_DIM = 768  # Wav2Vec2-base hidden size
FUSION_DIM = 512
NUM_HEADS = 8
NUM_CLASSES = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'=' * 60}")
print(f"  Device: {DEVICE}")
print(f"{'=' * 60}\n")


# ── Hilfsfunktionen ────────────────────────────────────────────────────────────


def make_video_hidden(batch=BATCH_SIZE, t_v=1568, d=VIDEO_DIM):
    """Dummy VideoMAE hidden states (B, T_v, D_v).
    T_v = 1568 Patch-Tokens für VideoMAE-base mit 16 Frames (kein CLS-Token)."""
    return torch.randn(batch, t_v, d, device=DEVICE)


def make_audio_hidden(batch=BATCH_SIZE, t_a=32, d=AUDIO_DIM):
    """Dummy Wav2Vec2 hidden states (B, T_a, D_a).
    T_a ≈ 32 für 10240 Audio-Samples (CNN stride ~320)."""
    return torch.randn(batch, t_a, d, device=DEVICE)


def make_pixel_values(batch=BATCH_SIZE):
    """Dummy Video-Input für den vollen Forward-Pass (B, T, C, H, W)."""
    return torch.randn(batch, NUM_FRAMES, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)


def make_input_values(batch=BATCH_SIZE):
    """Dummy Audio-Input für den vollen Forward-Pass (B, samples)."""
    return torch.randn(batch, AUDIO_LEN, device=DEVICE)


def make_labels(batch=BATCH_SIZE):
    return torch.randint(0, NUM_CLASSES, (batch,), device=DEVICE)


# ── Test 1: CrossAttentionFusion Shapes ───────────────────────────────────────


def test_fusion_output_shape():
    print("Test 1: CrossAttentionFusion — Output-Shape")
    fusion = CrossAttentionFusion(
        video_dim=VIDEO_DIM,
        audio_dim=AUDIO_DIM,
        fusion_dim=FUSION_DIM,
        num_heads=NUM_HEADS,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    video_h = make_video_hidden()
    audio_h = make_audio_hidden()

    logits = fusion(video_h, audio_h)

    assert logits.shape == (BATCH_SIZE, NUM_CLASSES), f"Erwartet ({BATCH_SIZE}, {NUM_CLASSES}), bekommen {logits.shape}"
    print(f"  ✓ Logits-Shape korrekt: {logits.shape}")


# ── Test 1b: Alle fusion_mode-Varianten (Ablation) ────────────────────────────


@pytest.mark.parametrize("mode", ["cross_attention", "concat", "video_only", "audio_only"])
def test_fusion_mode_output_shape(mode):
    fusion = CrossAttentionFusion(
        video_dim=VIDEO_DIM, audio_dim=AUDIO_DIM, fusion_dim=FUSION_DIM,
        num_heads=NUM_HEADS, num_classes=NUM_CLASSES, fusion_mode=mode,
    ).to(DEVICE).eval()
    logits = fusion(make_video_hidden(), make_audio_hidden())
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)
    assert not torch.isnan(logits).any() and not torch.isinf(logits).any()


def test_fusion_mode_invalid_raises():
    with pytest.raises(ValueError, match="fusion_mode"):
        CrossAttentionFusion(fusion_mode="bogus")


def test_fusion_mode_single_modality_ignores_dropped_input():
    """video_only must ignore audio: changing audio leaves logits unchanged."""
    torch.manual_seed(0)
    fusion = CrossAttentionFusion(
        video_dim=VIDEO_DIM, audio_dim=AUDIO_DIM, fusion_dim=FUSION_DIM,
        num_heads=NUM_HEADS, num_classes=NUM_CLASSES, fusion_mode="video_only",
    ).to(DEVICE).eval()
    v = make_video_hidden()
    out_a = fusion(v, make_audio_hidden())
    out_b = fusion(v, make_audio_hidden())  # different audio, same video
    assert torch.allclose(out_a, out_b, atol=1e-5), "video_only must not depend on audio"


# ── Test 2: Keine NaN / Inf im Output ─────────────────────────────────────────


def test_no_nan_inf():
    print("\nTest 2: CrossAttentionFusion — Keine NaN/Inf")
    fusion = CrossAttentionFusion(
        video_dim=VIDEO_DIM,
        audio_dim=AUDIO_DIM,
        fusion_dim=FUSION_DIM,
        num_heads=NUM_HEADS,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    logits = fusion(make_video_hidden(), make_audio_hidden())

    assert not torch.isnan(logits).any(), "NaN in Logits!"
    assert not torch.isinf(logits).any(), "Inf in Logits!"
    print(f"  ✓ Logits sauber: min={logits.min():.3f}  max={logits.max():.3f}")


# ── Test 3: Backpropagation durch Fusion-Head ─────────────────────────────────


def test_fusion_backprop():
    print("\nTest 3: CrossAttentionFusion — Backpropagation")
    fusion = CrossAttentionFusion(
        video_dim=VIDEO_DIM,
        audio_dim=AUDIO_DIM,
        fusion_dim=FUSION_DIM,
        num_heads=NUM_HEADS,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    video_h = make_video_hidden()
    audio_h = make_audio_hidden()
    labels = make_labels()

    logits = fusion(video_h, audio_h)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    # Prüfe, ob mindestens ein Parameter wirklich Gradienten hat
    grads = [p.grad for p in fusion.parameters() if p.grad is not None]
    assert len(grads) > 0, "Kein einziger Parameter hat einen Gradienten!"

    # Prüfe ob Gradienten ungleich Null sind
    total_grad_norm = sum(g.abs().sum().item() for g in grads)
    assert total_grad_norm > 0, "Alle Gradienten sind 0 — kein Lernfortschritt möglich!"
    print(f"  ✓ Gradienten fließen  (∑|grad| = {total_grad_norm:.4f})")
    print(f"  ✓ Loss               = {loss.item():.4f}")


# ── Test 4: Freeze-Backbones funktioniert ─────────────────────────────────────


def test_freeze_backbones():
    print("\nTest 4: MultimodalDeepfakeModule — freeze_backbone=True")
    model = MultimodalDeepfakeModule(
        freeze_backbone=True,
        optimizer=lambda params: torch.optim.AdamW(params, lr=1e-4),
    ).to(DEVICE)

    frozen_video = [p for p in model.video_backbone.parameters() if p.requires_grad]
    frozen_audio = [p for p in model.audio_backbone.parameters() if p.requires_grad]
    fusion_trainable = [p for p in model.fusion.parameters() if p.requires_grad]

    assert len(frozen_video) == 0, f"{len(frozen_video)} Video-Backbone-Parameter sind NICHT eingefroren!"
    assert len(frozen_audio) == 0, f"{len(frozen_audio)} Audio-Backbone-Parameter sind NICHT eingefroren!"
    assert len(fusion_trainable) > 0, "Fusion-Head hat keine trainierbaren Parameter!"

    # Frozen backbones must stay in eval() even after model.train() — sonst
    # laufen Dropout/Stochastic-Depth bei der Feature-Extraktion (Train/Eval-Mismatch).
    model.train()
    assert model.video_backbone.training is False, "Video-Backbone ist nach train() NICHT im eval-Modus!"
    assert model.audio_backbone.training is False, "Audio-Backbone ist nach train() NICHT im eval-Modus!"
    assert model.fusion.training is True, "Fusion-Head sollte nach train() im Trainings-Modus sein!"

    print(f"  ✓ Video-Backbone eingefroren  ({sum(p.numel() for p in model.video_backbone.parameters()):,} params)")
    print(f"  ✓ Audio-Backbone eingefroren  ({sum(p.numel() for p in model.audio_backbone.parameters()):,} params)")
    print(f"  ✓ Fusion-Head trainierbar     ({sum(p.numel() for p in model.fusion.parameters()):,} params)")
    print("  ✓ Backbones bleiben nach model.train() im eval-Modus")


# ── Test 5: Voller Forward-Pass mit echten Backbones ──────────────────────────


def test_full_forward_pass():
    print("\nTest 5: MultimodalDeepfakeModule — Voller Forward-Pass")
    model = (
        MultimodalDeepfakeModule(
            freeze_backbone=True,
            optimizer=lambda params: torch.optim.AdamW(params, lr=1e-4),
        )
        .to(DEVICE)
        .eval()
    )

    pixel_values = make_pixel_values()
    input_values = make_input_values()

    with torch.no_grad():
        logits = model(pixel_values, input_values)

    assert logits.shape == (BATCH_SIZE, NUM_CLASSES), f"Erwartet ({BATCH_SIZE}, {NUM_CLASSES}), bekommen {logits.shape}"
    probs = torch.softmax(logits, dim=1)
    assert (probs.sum(dim=1) - 1.0).abs().max() < 1e-5, "Softmax-Summe != 1"

    print(f"  ✓ Logits-Shape: {logits.shape}")
    print(f"  ✓ Softmax-Probs (Fake): {probs[:, 1].tolist()}")


# ── Test 6: Unfreeze funktioniert ─────────────────────────────────────────────


def test_unfreeze_backbones():
    print("\nTest 6: MultimodalDeepfakeModule — unfreeze_backbone()")
    model = MultimodalDeepfakeModule(
        freeze_backbone=True,
        optimizer=lambda params: torch.optim.AdamW(params, lr=1e-4),
    ).to(DEVICE)

    model.unfreeze_backbone()

    trainable_video = [p for p in model.video_backbone.parameters() if p.requires_grad]
    assert len(trainable_video) > 0, "Video-Backbone bleibt eingefroren nach unfreeze!"
    # Audio CNN feature extractor must stay frozen even after unfreeze (invariant).
    cnn_trainable = [p for p in model.audio_backbone.feature_extractor.parameters() if p.requires_grad]
    assert len(cnn_trainable) == 0, "Audio-CNN-Feature-Extractor darf nach unfreeze NICHT trainierbar sein!"
    print(f"  ✓ Video-Backbone nach unfreeze: {len(trainable_video)} trainierbare Parameter")
    print("  ✓ Audio-CNN bleibt eingefroren (Invariante)")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_fusion_output_shape,
        test_no_nan_inf,
        test_fusion_backprop,
        test_freeze_backbones,
        test_full_forward_pass,
        test_unfreeze_backbones,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FEHLER: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Ergebnis: {passed}/{len(tests)} Tests bestanden", end="")
    print("  ✓" if failed == 0 else f"  — {failed} fehlgeschlagen")
    print(f"{'=' * 60}\n")
    sys.exit(0 if failed == 0 else 1)
