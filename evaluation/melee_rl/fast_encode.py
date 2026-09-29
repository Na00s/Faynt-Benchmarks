"""Sync-free encoder forward for the actor (5 Sep 2026, ``/rl-speedup`` lever 2).

``SlippiEncoder.forward`` (``model.py:940-1005``) validates its categorical inputs with eleven
``bool(tensor.any())`` checks per call (``model.py:542, 777, 787, 925``): each is a device-to-host round trip
that drains the CUDA queue, and together they are 11 of the actor frame's sync points (the bench of 5 Sep
2026).  It also builds two one-hots per player that the learned-embedding configuration never reads
(``model.py:799, 810``).  :func:`fast_encode` runs the same orchestration with Ali's own modules and helpers
(``_field``, ``_scaled_float``, ``_embed_items``, ``_controller_embedding``, the embeddings) minus those
checks and dead tensors, so the output is bit-identical (``tests/rl/test_fast_encode.py``) and the call can
sit inside a CUDA graph.  The actor's inputs are validated upstream: the environment tree
(``validate_frame_tree``) and the actor's own samples (in range by construction).  Nothing in ``model.py``
changes; the configurations the RL policy does not use (player-name conditioning, the controller RNN) are
refused rather than reimplemented.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from controller_codec import ControllerLabels
from model import _RAW_STAGE_TO_LIBMELEE, SlippiEncoder, _as_prefix_tensor, _get, _has


def _stage(encoder: SlippiEncoder, game: Any, prefix: torch.Size, device: torch.device) -> torch.Tensor:
    """``SlippiEncoder._stage`` without its range check."""

    if _has(game, "stage_id"):
        raw = _as_prefix_tensor(_get(game, "stage_id"), prefix, device).long()
        stage = torch.full_like(raw, -1)
        for raw_value, mapped in _RAW_STAGE_TO_LIBMELEE.items():
            stage = torch.where(raw == raw_value, mapped, stage)
        return stage
    return _as_prefix_tensor(_get(game, "stage"), prefix, device).long()


def _embed_player_or_nana(
    encoder: SlippiEncoder, player: Any, prefix: torch.Size, device: torch.device, *, nana: bool
) -> torch.Tensor:
    """``SlippiEncoder._embed_player_or_nana`` without its two range checks and two dead one-hots."""

    config = encoder.config
    default = 0
    percent = encoder._field(player, prefix, device, "percent", default=default).long()
    percent = torch.remainder(percent, 1 << 16)
    facing_raw = encoder._field(player, prefix, device, "facing", "facing_direction", default=default)
    facing = torch.where(facing_raw.bool() if facing_raw.dtype == torch.bool else facing_raw > 0, 1.0, -1.0)
    x = encoder._field(player, prefix, device, "x", "x_position", default=default)
    y = encoder._field(player, prefix, device, "y", "y_position", default=default)
    action = (
        encoder._field(player, prefix, device, "action", "action_state_id", default=default)
        .long()
        .clamp(0, encoder.ACTION_SIZE - 1)
    )
    invulnerable = encoder._field(player, prefix, device, "invulnerable", default=default).bool()
    character = encoder._field(player, prefix, device, "character", "character_id", default=default).long()
    jumps = encoder._field(player, prefix, device, "jumps_left", "jumps_remaining", default=default).long()
    shield = encoder._field(player, prefix, device, "shield_strength", "shield_value", default=default)
    on_ground = encoder._field(player, prefix, device, "on_ground", default=default).bool()

    if config.use_learned_action:
        action_embed = encoder.action_embedding(action)
        if config.use_character_action_joint:
            joint_index = character * encoder.ACTION_SIZE + action
            action_embed = action_embed + encoder.character_action_embedding(joint_index)
        if config.hybrid_embed:
            action_embed = torch.cat((action_embed, F.one_hot(action, encoder.ACTION_SIZE).float()), dim=-1)
    else:
        action_embed = F.one_hot(action, encoder.ACTION_SIZE).float()

    if config.use_learned_character:
        character_embed = encoder.character_embedding(character)
        if config.hybrid_embed:
            character_embed = torch.cat(
                (character_embed, F.one_hot(character, encoder.CHARACTER_SIZE).float()), dim=-1
            )
    else:
        character_embed = F.one_hot(character, encoder.CHARACTER_SIZE).float()

    parts = [
        encoder._scaled_float(percent, 0.01),
        facing.unsqueeze(-1),
        encoder._scaled_float(x, 0.05),
        encoder._scaled_float(y, 0.05),
        action_embed,
        invulnerable.float().unsqueeze(-1),
        character_embed,
        F.one_hot(jumps, encoder.JUMPS_SIZE).float(),
        encoder._scaled_float(shield, 0.01),
        on_ground.float().unsqueeze(-1),
    ]
    if nana:
        exists = encoder._field(player, prefix, device, "exists", default=False).bool()
        parts.append(exists.float().unsqueeze(-1))
    return torch.cat([part.float() for part in parts], dim=-1)


def _embed_player(
    encoder: SlippiEncoder, player: Any, prefix: torch.Size, device: torch.device, *, with_nana: bool
) -> torch.Tensor:
    parts = [_embed_player_or_nana(encoder, player, prefix, device, nana=False)]
    if with_nana:
        nana = _get(player, "nana", "follower", default={})
        parts.append(_embed_player_or_nana(encoder, nana, prefix, device, nana=True))
    return torch.cat(parts, dim=-1)


def fast_encode(encoder: SlippiEncoder, game_state: Any, controller: ControllerLabels) -> torch.Tensor:
    """``encoder(game_state, controller)`` bit for bit, without host round trips (see the module docstring).

    ``game_state`` is a frame tree (mapping) or a ``GameStateBatch``; ``controller`` the previous-action
    labels with the tree's leading shape.
    """

    config = encoder.config
    if config.condition_on_player_name:
        raise NotImplementedError("fast_encode does not cover player-name conditioning")
    if encoder.controller_rnn is not None:
        raise NotImplementedError("fast_encode does not cover the controller RNN")
    labels = ControllerLabels(controller.buttons.long(), controller.main_stick.long())
    prefix = labels.buttons.shape
    device = labels.buttons.device
    self_player = encoder._player_from_game(game_state, self_player=True)
    opponent = encoder._player_from_game(game_state, self_player=False)
    stage = _stage(encoder, game_state, prefix, device)
    parts = [
        _embed_player(encoder, self_player, prefix, device, with_nana=config.use_self_nana),
        _embed_player(encoder, opponent, prefix, device, with_nana=True),
        F.one_hot(stage, encoder.STAGE_SIZE).float(),
    ]
    if config.use_randall:
        randall = _get(game_state, "randall", default=None)
        if randall is None:
            raise ValueError("fast_encode needs the randall leaves (the canonical frame tree carries them)")
        randall_x = _as_prefix_tensor(_get(randall, "x"), prefix, device)
        randall_y = _as_prefix_tensor(_get(randall, "y"), prefix, device)
        parts.extend((encoder._scaled_float(randall_x, 0.05), encoder._scaled_float(randall_y, 0.05)))
    if config.use_fod_platforms:
        fod = _get(game_state, "fod_platforms", "fod", default={})
        left = _as_prefix_tensor(_get(fod, "left", default=0), prefix, device)
        right = _as_prefix_tensor(_get(fod, "right", default=0), prefix, device)
        parts.extend((encoder._scaled_float(left, 0.05), encoder._scaled_float(right, 0.05)))
    if config.use_items:
        parts.append(encoder._embed_items(game_state, prefix, device))
    parts.append(encoder._controller_embedding(labels))
    result = torch.cat([part.float() for part in parts], dim=-1)
    if result.shape[-1] != encoder.output_dim:
        raise RuntimeError(f"encoder produced {result.shape[-1]} features, expected {encoder.output_dim}")
    return result


__all__ = ["fast_encode"]
