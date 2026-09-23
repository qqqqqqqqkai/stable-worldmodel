"""Pluggable planning objectives, decoupled from the world model.

An :class:`~stable_worldmodel.protocols.Objective` scores a populated
``info_dict`` (after the world model has rolled out candidate actions) and
returns a per-candidate cost tensor of shape ``(B, S)``. Objectives are plain,
composable objects: swapping or combining costs no longer requires subclassing
the world model. The ``Objective`` protocol itself lives in
:mod:`stable_worldmodel.protocols`; this module holds concrete objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_worldmodel.protocols import Objective


class GoalMSE(nn.Module):
    """Last-step MSE between predicted and goal embeddings.

    Reads ``predicted_emb`` ``(B, S, T, ...)`` and ``goal_emb``
    ``(B, T_goal, ...)``; returns per-candidate cost ``(B, S)``. The comparison
    is against the *last* predicted step and the *last* goal frame.

    Rank-agnostic: the time axis is addressed positionally (dim 2 of the
    prediction, dim 1 of the goal) rather than from the right, so latents that
    carry extra trailing axes work too — ``PreJEPA``'s pixel embedding has a
    patch axis, ``(B, S, T, patches, dim)``, and indexing from the right would
    silently slice patches instead of time.

    Point ``pred_key``/``goal_key`` at one source of a split latent (e.g.
    ``predicted_pixels_emb`` / ``pixels_goal_emb``) and combine the terms with
    :class:`WeightedSum` to score models whose latent concatenates several
    sources; see the planning guide.

    Behavior-preserving extraction of ``LeWM.criterion`` (reproduces it
    bit-for-bit), so an existing model migrates to the ``ShootingCostEvaluator`` seam
    without changing results.

    Args:
        pred_key: ``info_dict`` key holding the rollout predictions.
        goal_key: ``info_dict`` key holding the goal embedding.
        reduction: How the squared error is collapsed over every axis after
            the candidate axis. ``'sum'`` (the default) matches
            ``LeWM.criterion``; ``'mean'`` matches the former
            ``PreJEPA.criterion``. This choice also sets the *relative* weight
            of sources with different shapes, so it matters when composing
            per-source terms: summing over a 196x384 pixel embedding and a
            10-dim proprio embedding weights them ~7500x apart, whereas
            averaging puts them on a comparable scale.
    """

    def __init__(
        self,
        pred_key: str = 'predicted_emb',
        goal_key: str = 'goal_emb',
        reduction: str = 'sum',
    ) -> None:
        super().__init__()
        if reduction not in ('sum', 'mean'):
            raise ValueError(
                f"reduction must be 'sum' or 'mean', got {reduction!r}"
            )
        self.pred_key = pred_key
        self.goal_key = goal_key
        self.reduction = reduction

    def forward(self, info_dict: dict) -> torch.Tensor:
        pred_emb = info_dict[self.pred_key]
        goal_emb = info_dict[self.goal_key][:, None, -1:].expand_as(pred_emb)
        err = F.mse_loss(
            pred_emb[:, :, -1:],
            goal_emb[:, :, -1:].detach(),
            reduction='none',
        )
        dims = tuple(range(2, pred_emb.ndim))
        return (
            err.sum(dim=dims)
            if self.reduction == 'sum'
            else err.mean(dim=dims)
        )


class CumulativeGoalMSE(GoalMSE):
    """Sum goal distance over predicted rollout steps.

    The initial context is excluded using ``history_size``.  This matches the
    cumulative trajectory cost in VJEPA-MPC while retaining GoalMSE's choice
    of keys and latent-dimension reduction.
    """

    def __init__(self, history_size: int = 3, **kwargs) -> None:
        super().__init__(**kwargs)
        if history_size < 1:
            raise ValueError('history_size must be positive')
        self.history_size = history_size

    def forward(self, info_dict: dict) -> torch.Tensor:
        pred_emb = info_dict[self.pred_key][:, :, self.history_size :]
        if pred_emb.size(2) == 0:
            raise ValueError('rollout contains no predicted steps')
        goal = info_dict[self.goal_key][:, None, -1:]
        goal = goal.expand(
            pred_emb.size(0),
            pred_emb.size(1),
            pred_emb.size(2),
            *pred_emb.shape[3:],
        )
        err = F.mse_loss(pred_emb, goal.detach(), reduction='none')
        feature_dims = tuple(range(3, pred_emb.ndim))
        step_cost = (
            err.sum(dim=feature_dims)
            if self.reduction == 'sum'
            else err.mean(dim=feature_dims)
        )
        return step_cost.sum(dim=2)


class ControlPenalty(nn.Module):
    """L2 penalty on the action candidates themselves.

    Reads ``action_candidates`` (shape ``(B, S, H, action_dim)``) from the
    ``info_dict`` — the ``ShootingCostEvaluator`` stores them there before scoring —
    and returns a per-candidate cost ``(B, S)``. Demonstrates a cost the world
    model never had to know about.
    """

    def __init__(self, action_key: str = 'action_candidates') -> None:
        super().__init__()
        self.action_key = action_key

    def forward(self, info_dict: dict) -> torch.Tensor:
        actions = info_dict[self.action_key]
        return actions.pow(2).sum(dim=tuple(range(2, actions.ndim)))


class WeightedSum(nn.Module):
    """Linear combination of objectives: ``sum(w * term(info_dict))``.

    Lets us assemble multi-term costs (goal distance + control penalty +
    constraints) without touching the model or the solver.
    """

    def __init__(self, terms: list[tuple[float, Objective]]) -> None:
        super().__init__()
        if not terms:
            raise ValueError('WeightedSum requires at least one term')
        self.weights = [weight for weight, _ in terms]
        self.objectives = nn.ModuleList(term for _, term in terms)

    def forward(self, info_dict: dict) -> torch.Tensor:
        total = None
        for weight, term in zip(self.weights, self.objectives):
            value = weight * term(info_dict)
            total = value if total is None else total + value
        return total
