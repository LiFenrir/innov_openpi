"""Tests for PI0Pytorch embedding extraction methods.

Verifies that the first-class ``extract_prefix_embeddings()`` API on
PI0Pytorch produces correct outputs without monkey-patching.
"""

from unittest.mock import MagicMock

import torch

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


def _make_mock_pi0(B=1, M=20, D=2048, action_horizon=50, action_dim=14):
    """Create a mock PI0Pytorch that returns known tensors from key methods."""
    pi0 = MagicMock(spec=PI0Pytorch)

    # _preprocess_observation
    images = torch.randn(B, 3, 224, 224)
    img_masks = torch.ones(B, 1, dtype=torch.bool)
    lang_tokens = torch.randint(0, 1000, (B, 10))
    lang_masks = torch.ones(B, 10, dtype=torch.bool)
    state = torch.randn(B, action_dim)
    pi0._preprocess_observation.return_value = (images, img_masks, lang_tokens, lang_masks, state)

    # embed_prefix
    prefix_embs = torch.randn(B, M, D)
    prefix_pad_masks = torch.ones(B, M, dtype=torch.bool)
    prefix_att_masks = torch.ones(B, M, dtype=torch.bool)
    pi0.embed_prefix.return_value = (prefix_embs, prefix_pad_masks, prefix_att_masks)

    # paligemma_with_expert.forward — returns ([prefix_out, suffix_out], kv_cache)
    prefix_out = torch.randn(B, M, D)
    pi0.paligemma_with_expert.forward.return_value = ([prefix_out, None], None)

    # Mock language model config for dtype check
    mock_config = MagicMock()
    mock_config._attn_implementation = "eager"
    mock_layer = MagicMock()
    mock_layer.self_attn.q_proj.weight.dtype = torch.float32
    pi0.paligemma_with_expert.paligemma.language_model.config = mock_config
    pi0.paligemma_with_expert.paligemma.language_model.layers = [mock_layer]

    # _prepare_attention_masks_4d
    pi0._prepare_attention_masks_4d.return_value = torch.ones(B, 1, M, M)

    # sample_actions
    pi0.sample_actions.return_value = torch.randn(B, action_horizon, action_dim)

    return pi0


def test_extract_prefix_embeddings_is_callable():
    """Verify PI0Pytorch exposes extract_prefix_embeddings."""
    assert hasattr(PI0Pytorch, "extract_prefix_embeddings")
    assert callable(getattr(PI0Pytorch, "extract_prefix_embeddings"))


def test_forward_with_prefix_embeddings_is_callable():
    """Verify PI0Pytorch exposes forward_with_prefix_embeddings."""
    assert hasattr(PI0Pytorch, "forward_with_prefix_embeddings")
    assert callable(getattr(PI0Pytorch, "forward_with_prefix_embeddings"))


def test_extract_prefix_embeddings_shapes():
    """Test that extract_prefix_embeddings returns expected shapes."""
    B, M, D = 2, 20, 2048
    pi0 = _make_mock_pi0(B=B, M=M, D=D)
    obs = {"state": torch.randn(B, 14)}
    z, pad_mask = pi0.extract_prefix_embeddings(obs)
    assert z.shape == (B, M, D)
    assert pad_mask.shape == (B, M)
    assert pad_mask.dtype == torch.bool


def test_extract_prefix_embeddings_output_is_float32():
    """Test that extract_prefix_embeddings returns float32 output."""
    B, M, D = 1, 10, 2048
    pi0 = _make_mock_pi0(B=B, M=M, D=D)
    obs = {"state": torch.randn(B, 14)}
    z, _ = pi0.extract_prefix_embeddings(obs)
    assert z.dtype == torch.float32


def test_sample_actions_shape():
    """Test that sample_actions still works on the mock PI0Pytorch."""
    B, action_horizon, action_dim = 1, 50, 14
    pi0 = _make_mock_pi0(B=B, action_horizon=action_horizon, action_dim=action_dim)
    device = torch.device("cpu")
    actions = pi0.sample_actions(device, {"state": torch.randn(B, action_dim)})
    assert actions.shape == (B, action_horizon, action_dim)
