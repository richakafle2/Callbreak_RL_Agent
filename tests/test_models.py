"""
tests/test_models.py
---------------------
Unit tests for neural network model components.
"""

import pytest
import torch
from models.actor_critic import ActorCritic, BidHead, PlayHead, ValueHead
from models.encoder import MLPEncoder
from utils.state_encoder import OBS_DIM


class TestMLPEncoder:
    def test_output_shape(self):
        enc = MLPEncoder(obs_dim=OBS_DIM, embed_dim=128, hidden_dim=256)
        x = torch.zeros(4, OBS_DIM)
        out = enc(x)
        assert out.shape == (4, 128)


class TestActorCritic:
    def test_forward_play_phase(self):
        model = ActorCritic(obs_dim=OBS_DIM, encoder_type="mlp")
        obs = torch.zeros(2, OBS_DIM)
        mask = torch.ones(2, 52, dtype=torch.bool)
        log_probs, value = model(obs, mask, phase="play")
        assert log_probs.shape == (2, 52)
        assert value.shape == (2, 1)

    def test_forward_bid_phase(self):
        model = ActorCritic(obs_dim=OBS_DIM, encoder_type="mlp")
        obs = torch.zeros(2, OBS_DIM)
        mask = torch.ones(2, 13, dtype=torch.bool)
        log_probs, value = model(obs, mask, phase="bid")
        assert log_probs.shape == (2, 13)

    def test_illegal_actions_masked(self):
        """Masked-out actions should have log_prob ≈ -inf."""
        model = ActorCritic(obs_dim=OBS_DIM, encoder_type="mlp")
        obs = torch.zeros(1, OBS_DIM)
        mask = torch.zeros(1, 52, dtype=torch.bool)
        mask[0, 0] = True   # only card 0 is legal
        log_probs, _ = model(obs, mask, phase="play")
        assert log_probs[0, 0].item() > -1e8
        assert log_probs[0, 1].item() < -1e8

    def test_get_action_and_value_shapes(self):
        model = ActorCritic(obs_dim=OBS_DIM, encoder_type="mlp")
        obs = torch.zeros(3, OBS_DIM)
        mask = torch.ones(3, 52, dtype=torch.bool)
        action, log_prob, entropy, value = model.get_action_and_value(obs, mask)
        assert action.shape == (3,)
        assert log_prob.shape == (3,)
        assert entropy.shape == (3,)
        assert value.shape == (3,)
