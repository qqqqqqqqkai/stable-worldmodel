"""Check whether VJEPA online and EMA parameters change across checkpoints.

This is a parameter-level diagnostic for the EMA update/save/load chain.  It
does not train or modify checkpoints.  For each supplied checkpoint it reports
parameter counts and compares every branch with the first checkpoint:

* ``encoder`` and ``projector`` are the trainable online branches;
* ``target_encoder`` and ``target_projector`` are the EMA branches.

If online parameters change while target parameters remain exactly constant,
the EMA callback or checkpoint serialization path is broken.  If both change,
but target changes are much smaller, that is consistent with a working EMA.

Example::

    python scripts/diagnostics/vjepa_ema_timeline.py \
        --checkpoint epoch1=vjepa_beta2_epoch1/weights_epoch_1.pt \
        --checkpoint epoch25=vjepa_beta2_epoch25/weights_epoch_25.pt \
        --checkpoint epoch63=vjepa_beta2_epoch63/weights_epoch_63.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import stable_worldmodel as swm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--checkpoint', action='append', required=True, metavar='LABEL=PATH'
    )
    return parser.parse_args()


def parse_entries(entries: list[str]) -> list[tuple[str, str]]:
    parsed = []
    for entry in entries:
        label, separator, path = entry.partition('=')
        if not separator or not label or not path:
            raise ValueError(f'Expected LABEL=PATH, got {entry!r}')
        parsed.append((label, path))
    return parsed


def branch_parameters(model, prefix: str) -> dict[str, torch.Tensor]:
    """Return CPU copies so later comparisons cannot mutate model state."""
    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if name == prefix or name.startswith(prefix + '.')
    }


def compare(reference, current) -> tuple[float, float, float, int]:
    names = sorted(set(reference) & set(current))
    if not names:
        raise ValueError('No matching parameters found between checkpoints')
    deltas = torch.cat([
        (current[name] - reference[name]).reshape(-1) for name in names
    ])
    return (
        deltas.norm().item(),
        deltas.abs().mean().item(),
        deltas.abs().max().item(),
        int(torch.count_nonzero(deltas).item()),
    )


def main() -> None:
    entries = parse_entries(parse_args().checkpoint)
    loaded = []
    for label, checkpoint in entries:
        if not Path(checkpoint).exists() and '/' not in checkpoint:
            # load_pretrained also resolves names relative to STABLEWM_HOME.
            pass
        model = swm.wm.utils.load_pretrained(checkpoint)
        loaded.append((label, checkpoint, model))

    branches = [
        'encoder',
        'projector',
        'target_encoder',
        'target_projector',
    ]
    reference = {
        branch: branch_parameters(loaded[0][2], branch)
        for branch in branches
    }

    print(f'reference: {loaded[0][0]} ({loaded[0][1]})')
    print('label\tbranch\tparam_count\tl2_delta\tmean_abs_delta\tmax_abs_delta\tnonzero')
    for label, checkpoint, model in loaded:
        for branch in branches:
            current = branch_parameters(model, branch)
            l2, mean_abs, max_abs, nonzero = compare(
                reference[branch], current
            )
            count = sum(value.numel() for value in current.values())
            print(
                f'{label}\t{branch}\t{count}\t{l2:.8e}\t'
                f'{mean_abs:.8e}\t{max_abs:.8e}\t{nonzero}'
            )


if __name__ == '__main__':
    main()
