"""Compare VJEPA representation diversity across training checkpoints.

The same randomly selected clips are encoded at several training stages.  The
comparison answers whether representation collapse was already present in the
model structure/input path or developed during optimization.  It is read-only:
checkpoints and datasets are never written.

Example::

    python scripts/diagnostics/vjepa_collapse_timeline.py \
        --checkpoint random=vjepa_beta2_epoch63/weights_epoch_63.pt \
        --checkpoint epoch1=vjepa_beta2_epoch1/weights_epoch_1.pt \
        --checkpoint epoch63=vjepa_beta2_epoch63/weights_epoch_63.pt \
        --dataset /stable_wm/stablewm_data/datasets/pusht_expert_train.lance

The ``random`` label is special: it loads the first checkpoint's architecture,
resets trainable parameters, and makes the EMA copies equal to the online
copies.  Thus it is a random-initialization control, not a trained checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torchvision.transforms import v2 as transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--checkpoint',
        action='append',
        required=True,
        metavar='LABEL=PATH',
        help='Checkpoint entry; repeat for each training stage.',
    )
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--num-samples', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--img-size', type=int, default=224)
    return parser.parse_args()


def parse_entries(entries: list[str]) -> list[tuple[str, str]]:
    parsed = []
    for entry in entries:
        label, separator, path = entry.partition('=')
        if not separator or not label or not path:
            raise ValueError(f'Expected LABEL=PATH, got {entry!r}')
        parsed.append((label, path))
    return parsed


def make_transform(img_size: int):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize((img_size, img_size)),
    ])


def randomize_model(model: torch.nn.Module) -> torch.nn.Module:
    """Reset the architecture while keeping online and EMA branches aligned."""
    model = copy.deepcopy(model)
    model.apply(lambda module: module.reset_parameters()
                if hasattr(module, 'reset_parameters') else None)
    with torch.no_grad():
        model.target_encoder.load_state_dict(model.encoder.state_dict())
        model.target_projector.load_state_dict(model.projector.state_dict())
    return model


def measure(model, dataset, transform, indices, device):
    online_raw, ema_raw = [], []
    online_latent, ema_latent = [], []
    model = model.to(device).eval()
    with torch.no_grad():
        for index in indices:
            pixels = dataset[index]['pixels']
            pixels = torch.stack([transform(frame) for frame in pixels])
            pixels = pixels.to(device)
            online_output = model.encoder(
                pixels, interpolate_pos_encoding=True
            )
            ema_output = model.target_encoder(
                pixels, interpolate_pos_encoding=True
            )
            online_cls = online_output.last_hidden_state[:, 0]
            ema_cls = ema_output.last_hidden_state[:, 0]
            online_raw.append(online_cls.cpu())
            ema_raw.append(ema_cls.cpu())
            online_latent.append(model.projector(online_cls).cpu())
            ema_latent.append(model.target_projector(ema_cls).cpu())

    values = [
        torch.cat(value).float()
        for value in (online_raw, ema_raw, online_latent, ema_latent)
    ]
    return tuple(value.std(dim=0).mean().item() for value in values)


def main() -> None:
    args = parse_args()
    entries = parse_entries(args.checkpoint)
    if not Path(args.dataset).exists():
        raise FileNotFoundError(f'Dataset not found: {args.dataset}')

    device = torch.device(args.device)
    dataset = swm.data.load_dataset(
        args.dataset,
        keys_to_cache=['pixels'],
    )
    num_samples = min(args.num_samples, len(dataset))
    indices = random.Random(args.seed).sample(range(len(dataset)), num_samples)
    transform = make_transform(args.img_size)

    first_model = swm.wm.utils.load_pretrained(entries[0][1])
    print(f'dataset: {args.dataset}')
    print(f'samples: {num_samples}')
    print(f'sampling seed: {args.seed}')
    print(f'index range: {min(indices)}..{max(indices)}')
    print(
        'label\tonline_encoder_std\tema_encoder_std\t'
        'online_projector_std\tema_projector_std'
    )

    for label, checkpoint in entries:
        model = (
            randomize_model(first_model)
            if label == 'random'
            else swm.wm.utils.load_pretrained(checkpoint)
        )
        stats = measure(model, dataset, transform, indices, device)
        print(
            f'{label}\t' + '\t'.join(f'{value:.8f}' for value in stats)
        )


if __name__ == '__main__':
    main()
