import math

import torch
from torch import nn

from stable_worldmodel.protocols import Dynamics
from stable_worldmodel.wm.vjepa import (
    BoundedLogVariance,
    VJEPA,
    gaussian_nll,
    reparameterize,
    unit_gaussian_kl,
)


class _EncoderOutput:
    def __init__(self, hidden):
        self.last_hidden_state = hidden


class _ToyEncoder(nn.Module):
    def __init__(self, input_dim=3, latent_dim=4):
        super().__init__()
        self.proj = nn.Linear(input_dim, latent_dim)

    def forward(self, pixels, interpolate_pos_encoding=True):
        pooled = pixels.mean(dim=(-2, -1))
        cls = self.proj(pooled)
        return _EncoderOutput(cls[:, None])


class _ToyPredictor(nn.Module):
    def __init__(self, latent_dim=4, num_frames=3):
        super().__init__()
        self.num_frames = num_frames
        self.proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, embedding, action):
        return self.proj(embedding + action)


def _model(latent_dim=4):
    return VJEPA(
        encoder=_ToyEncoder(latent_dim=latent_dim),
        predictor=_ToyPredictor(latent_dim=latent_dim),
        action_encoder=nn.Linear(2, latent_dim),
        projector=nn.Linear(latent_dim, latent_dim),
        pred_mean_head=nn.Linear(latent_dim, latent_dim),
        pred_log_var_head=BoundedLogVariance(latent_dim, latent_dim),
        target_log_var_head=BoundedLogVariance(latent_dim, latent_dim),
    )


def test_vjepa_satisfies_dynamics_protocol():
    assert isinstance(_model(), Dynamics)


def test_bounded_log_variance_initializes_at_unit_variance():
    head = BoundedLogVariance(4, 4, -10.0, 5.0)
    actual = head(torch.randn(3, 4))
    torch.testing.assert_close(
        actual, torch.zeros_like(actual), atol=1e-6, rtol=0
    )


def test_gaussian_losses_match_torch_distributions():
    torch.manual_seed(0)
    target = torch.randn(2, 3, 4)
    mean = torch.randn(2, 3, 4)
    log_var = torch.randn(2, 3, 4).clamp(-2, 2)

    distribution = torch.distributions.Normal(
        mean, torch.exp(0.5 * log_var)
    )
    torch.testing.assert_close(
        gaussian_nll(target, mean, log_var), -distribution.log_prob(target)
    )

    prior = torch.distributions.Normal(
        torch.zeros_like(mean), torch.ones_like(mean)
    )
    expected_kl = torch.distributions.kl_divergence(distribution, prior)
    torch.testing.assert_close(unit_gaussian_kl(mean, log_var), expected_kl)


def test_reparameterize_uses_supplied_epsilon():
    mean = torch.tensor([[1.0, 2.0]])
    log_var = torch.tensor([[0.0, math.log(4.0)]])
    epsilon = torch.tensor([[3.0, 4.0]])
    expected = torch.tensor([[4.0, 10.0]])
    torch.testing.assert_close(
        reparameterize(mean, log_var, epsilon), expected
    )


def test_forward_shapes_and_gradient_ownership():
    torch.manual_seed(0)
    model = _model()
    pixels = torch.randn(2, 4, 3, 8, 8)
    actions = torch.randn(2, 3, 2)

    context = model.encode({'pixels': pixels[:, :3], 'action': actions})
    target, target_mean, target_log_var = model.sample_target(pixels[:, 1:])
    pred_mean, pred_log_var = model.predict_distribution(
        context['emb'], context['act_emb']
    )

    outputs = target, target_mean, target_log_var, pred_mean, pred_log_var
    for value in outputs:
        assert value.shape == (2, 3, 4)
        assert torch.isfinite(value).all()

    loss = gaussian_nll(target, pred_mean, pred_log_var).mean()
    loss = loss + 0.01 * unit_gaussian_kl(
        target_mean, target_log_var
    ).mean()
    loss.backward()

    assert all(
        parameter.grad is None
        for parameter in model.target_encoder.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.target_projector.parameters()
    )
    modules_with_grad = (
        model.encoder,
        model.predictor,
        model.pred_mean_head,
        model.pred_log_var_head,
        model.target_log_var_head,
    )
    for module in modules_with_grad:
        assert any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
            for parameter in module.parameters()
        )


def test_target_mean_is_ema_updated_and_stays_in_eval_mode():
    model = _model()
    source = next(model.encoder.parameters())
    target = next(model.target_encoder.parameters())
    before = target.detach().clone()

    with torch.no_grad():
        source.add_(2.0)
    model.update_target(decay=0.75)
    torch.testing.assert_close(target, before + 0.5)

    model.train()
    assert not model.target_encoder.training
    assert not model.target_projector.training
    assert all(
        not parameter.requires_grad
        for parameter in model.target_encoder.parameters()
    )


def test_goal_encoding_uses_ema_target_modules():
    model = _model()
    pixels = torch.randn(2, 1, 3, 8, 8)

    with torch.no_grad():
        model.encoder.proj.weight.add_(1.0)

    online = model.encode({'pixels': pixels.clone()})['emb']
    goal = model.encode_goal({'pixels': pixels.clone()})['emb']
    expected = model.infer_target(pixels)[0]

    torch.testing.assert_close(goal, expected)
    assert not torch.allclose(goal, online)


def test_mean_rollout_preserves_context_and_predicts_each_candidate_step():
    torch.manual_seed(0)
    model = _model()
    batch_size, candidates, context_len, horizon = 2, 3, 3, 4
    info = {
        'pixels': torch.randn(batch_size, candidates, context_len, 3, 8, 8),
        'action_history': torch.randn(batch_size, candidates, 2, 2),
    }
    action_sequence = torch.randn(batch_size, candidates, horizon, 2)

    output = model.rollout(info, action_sequence)
    assert output['predicted_emb'].shape == (
        batch_size,
        candidates,
        context_len + horizon,
        4,
    )
    assert output['action'].shape == (batch_size, candidates, context_len, 2)
    assert torch.isfinite(output['predicted_emb']).all()


def test_stochastic_rollout_samples_each_candidate_step():
    torch.manual_seed(0)
    model = _model()
    batch_size, candidates, context_len, horizon = 1, 3, 3, 4
    pixels = torch.randn(batch_size, 1, context_len, 3, 8, 8)
    info = {
        'pixels': pixels.expand(-1, candidates, -1, -1, -1, -1),
        'action_history': torch.zeros(batch_size, candidates, 2, 2),
    }
    actions = torch.zeros(batch_size, candidates, horizon, 2)

    torch.manual_seed(123)
    first = model.rollout_stochastic(dict(info), actions)['predicted_emb']
    torch.manual_seed(123)
    second = model.rollout_stochastic(dict(info), actions)['predicted_emb']

    torch.testing.assert_close(first, second)
    assert first.shape == (1, candidates, context_len + horizon, 4)
    assert not torch.allclose(
        first[:, 0, context_len:], first[:, 1, context_len:]
    )
