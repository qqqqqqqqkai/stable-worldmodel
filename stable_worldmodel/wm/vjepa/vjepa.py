from copy import deepcopy

import torch
from einops import rearrange
from torch import nn

from .module import reparameterize


class VJEPA(nn.Module):
    """Pixels-only variational JEPA with EMA target-mean encoders."""

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        action_encoder: nn.Module,
        projector: nn.Module | None = None,
        pred_mean_head: nn.Module | None = None,
        pred_log_var_head: nn.Module | None = None,
        target_log_var_head: nn.Module | None = None,
        ema_decay: float = 0.99,
        **kwargs,
    ) -> None:
        super().__init__()
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError('ema_decay must be in [0, 1)')

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_mean_head = pred_mean_head or nn.Identity()
        self.pred_log_var_head = pred_log_var_head or nn.Identity()
        self.target_log_var_head = target_log_var_head or nn.Identity()
        self.ema_decay = float(ema_decay)

        self.target_encoder = deepcopy(self.encoder)
        self.target_projector = deepcopy(self.projector)
        self._freeze_target_mean()

    def _freeze_target_mean(self) -> None:
        self.target_encoder.requires_grad_(False)
        self.target_projector.requires_grad_(False)
        self.target_encoder.eval()
        self.target_projector.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        self.target_projector.eval()
        return self

    @staticmethod
    def _encode_pixels(
        pixels: torch.Tensor,
        encoder: nn.Module,
        projector: nn.Module,
    ) -> torch.Tensor:
        pixels = pixels.to(next(encoder.parameters()).dtype)
        batch_size = pixels.size(0)
        pixels = rearrange(pixels, 'b t ... -> (b t) ...')
        output = encoder(pixels, interpolate_pos_encoding=True)
        embedding = projector(output.last_hidden_state[:, 0])
        return rearrange(embedding, '(b t) d -> b t d', b=batch_size)

    def encode(self, info: dict) -> dict:
        """Encode observations for context or goal inference."""
        info['emb'] = self._encode_pixels(
            info['pixels'], self.encoder, self.projector
        )
        if 'action' in info:
            info['act_emb'] = self.action_encoder(info['action'])
        return info

    def encode_goal(self, info: dict) -> dict:
        """Encode planning goals in the EMA target latent space."""
        info['emb'] = self._encode_pixels(
            info['pixels'], self.target_encoder, self.target_projector
        )
        return info

    def infer_target(self, pixels: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return q mean and learned input-dependent log variance."""
        with torch.no_grad():
            mean = self._encode_pixels(
                pixels, self.target_encoder, self.target_projector
            )
        log_var = self.target_log_var_head(mean.detach())
        return mean, log_var

    def predict_distribution(
        self, emb: torch.Tensor, act_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.predictor(emb, act_emb)
        hidden_flat = rearrange(hidden, 'b t d -> (b t) d')
        mean = self.pred_mean_head(hidden_flat)
        log_var = self.pred_log_var_head(hidden_flat)
        mean = rearrange(mean, '(b t) d -> b t d', b=emb.size(0))
        log_var = rearrange(log_var, '(b t) d -> b t d', b=emb.size(0))
        return mean, log_var

    def sample_target(
        self,
        pixels: torch.Tensor,
        epsilon: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_var = self.infer_target(pixels)
        return reparameterize(mean, log_var, epsilon), mean, log_var

    @torch.no_grad()
    def update_target(self, decay: float | None = None) -> None:
        """Update the target-mean encoder after an optimizer step."""
        decay = self.ema_decay if decay is None else float(decay)
        if not 0.0 <= decay <= 1.0:
            raise ValueError('EMA decay must be in [0, 1]')

        for online, target in (
            (self.encoder, self.target_encoder),
            (self.projector, self.target_projector),
        ):
            for source, destination in zip(
                online.parameters(), target.parameters(), strict=True
            ):
                destination.lerp_(source, 1.0 - decay)
            for source, destination in zip(
                online.buffers(), target.buffers(), strict=True
            ):
                if torch.is_floating_point(destination):
                    destination.lerp_(source, 1.0 - decay)
                else:
                    destination.copy_(source)

        self._freeze_target_mean()

    def rollout(
        self,
        info: dict,
        action_sequence: torch.Tensor,
        history_size: int | None = None,
    ) -> dict:
        """Autoregressive mean rollout compatible with the LeWM planner."""
        if history_size is None:
            history_size = getattr(self.predictor, 'num_frames', 3)

        assert 'pixels' in info, 'pixels not in info_dict'
        n_context = info['pixels'].size(2)
        batch_size, n_candidates, horizon = action_sequence.shape[:3]
        action_history = info.get('action_history')
        if action_history is None:
            action_history = action_sequence.new_zeros(
                batch_size,
                n_candidates,
                0,
                action_sequence.size(-1),
            )
        assert action_history.size(2) == n_context - 1, (
            f'action_history must hold H-1={n_context - 1} executed blocks, '
            f'got {action_history.size(2)}'
        )
        info['action'] = torch.cat(
            [action_history, action_sequence[:, :, :1]], dim=2
        )

        if 'emb' not in info:
            initial = {
                key: value[:, 0]
                for key, value in info.items()
                if torch.is_tensor(value)
            }
            initial = self.encode(initial)
            info['emb'] = (
                initial['emb']
                .detach()
                .unsqueeze(1)
                .expand(batch_size, n_candidates, -1, -1)
            )

        initial_emb = rearrange(info['emb'], 'b s ... -> (b s) ...')
        past_flat = rearrange(action_history, 'b s ... -> (b s) ...')
        candidates_flat = rearrange(action_sequence, 'b s ... -> (b s) ...')
        all_action_emb = self.action_encoder(
            torch.cat([past_flat, candidates_flat], dim=1)
        )

        embeddings = list(initial_emb.unbind(dim=1))
        for step in range(horizon):
            lower = max(0, n_context + step - history_size)
            context = torch.stack(embeddings[lower:], dim=1)
            actions = all_action_emb[:, lower : n_context + step]
            mean, _ = self.predict_distribution(context, actions)
            embeddings.append(mean[:, -1])

        rollout = torch.stack(embeddings, dim=1)
        info['predicted_emb'] = rearrange(
            rollout,
            '(b s) ... -> b s ...',
            b=batch_size,
            s=n_candidates,
        )
        return info

    def rollout_stochastic(
        self,
        info: dict,
        action_sequence: torch.Tensor,
        history_size: int | None = None,
    ) -> dict:
        """Sample one latent trajectory for every candidate action sequence.

        This implements the stochastic dynamics rollout used by VJEPA-MPC:
        each candidate receives independent Gaussian noise at every horizon
        step, and the sampled state is recursively fed back as context.
        """
        if history_size is None:
            history_size = getattr(self.predictor, 'num_frames', 3)

        assert 'pixels' in info, 'pixels not in info_dict'
        n_context = info['pixels'].size(2)
        batch_size, n_candidates, horizon = action_sequence.shape[:3]
        action_history = info.get('action_history')
        if action_history is None:
            action_history = action_sequence.new_zeros(
                batch_size,
                n_candidates,
                0,
                action_sequence.size(-1),
            )
        assert action_history.size(2) == n_context - 1, (
            f'action_history must hold H-1={n_context - 1} executed blocks, '
            f'got {action_history.size(2)}'
        )
        info['action'] = torch.cat(
            [action_history, action_sequence[:, :, :1]], dim=2
        )

        if 'emb' not in info:
            initial = {
                key: value[:, 0]
                for key, value in info.items()
                if torch.is_tensor(value)
            }
            initial = self.encode(initial)
            info['emb'] = (
                initial['emb']
                .detach()
                .unsqueeze(1)
                .expand(batch_size, n_candidates, -1, -1)
            )

        initial_emb = rearrange(info['emb'], 'b s ... -> (b s) ...')
        past_flat = rearrange(action_history, 'b s ... -> (b s) ...')
        candidates_flat = rearrange(action_sequence, 'b s ... -> (b s) ...')
        all_action_emb = self.action_encoder(
            torch.cat([past_flat, candidates_flat], dim=1)
        )

        embeddings = list(initial_emb.unbind(dim=1))
        for step in range(horizon):
            lower = max(0, n_context + step - history_size)
            context = torch.stack(embeddings[lower:], dim=1)
            actions = all_action_emb[:, lower : n_context + step]
            mean, log_var = self.predict_distribution(context, actions)
            sampled = reparameterize(mean[:, -1], log_var[:, -1])
            embeddings.append(sampled)

        rollout = torch.stack(embeddings, dim=1)
        info['predicted_emb'] = rearrange(
            rollout,
            '(b s) ... -> b s ...',
            b=batch_size,
            s=n_candidates,
        )
        return info


__all__ = ['VJEPA']
