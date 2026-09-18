"""Diagnose the online/EMA latent-space gap in a trained VJEPA model.

This script is intentionally read-only.  It compares the same PushT pixel
clips through the online encoder/projector and the EMA target
encoder/projector.  Running the comparison before and after the projector
helps distinguish an encoder-space problem from a projector (for example
BatchNorm state) problem.

Example::

    python scripts/diagnostics/vjepa_latent_gap.py \
        --checkpoint vjepa_beta2_epoch63/weights_epoch_63.pt \
        --dataset /stable_wm/stablewm_data/datasets/pusht_expert_train.lance \
        --num-samples 32
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--num-samples', type=int, default=32)
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed for deterministic sampling across the full dataset.',
    )
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--img-size', type=int, default=224)
    return parser.parse_args()


def make_transform(img_size: int):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize((img_size, img_size)),
    ])


def collect_features(model, dataset, transform, indices, device):
    online_raw = []
    ema_raw = []
    online_latent = []
    ema_latent = []

    with torch.no_grad():
        for index in indices:
            pixels = dataset[index]['pixels']
            # Lance returns a temporal clip.  Transform each frame before
            # adding the batch dimension expected by _encode_pixels().
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

    return (
        torch.cat(online_raw).float(),
        torch.cat(ema_raw).float(),
        torch.cat(online_latent).float(),
        torch.cat(ema_latent).float(),
    )


def report(name: str, online: torch.Tensor, ema: torch.Tensor) -> None:
    """Print diagnostics that separate scale, direction, and collapse."""
    difference = online - ema
    online_norm = online.norm(dim=-1)
    ema_norm = ema.norm(dim=-1)
    relative_l2 = difference.norm(dim=-1) / online_norm.clamp_min(1e-8)
    cosine = F.cosine_similarity(online, ema, dim=-1)
    centered_online = online - online.mean(dim=0, keepdim=True)
    centered_ema = ema - ema.mean(dim=0, keepdim=True)
    centered_cosine = F.cosine_similarity(
        centered_online, centered_ema, dim=-1
    )

    print(f'\n===== {name} =====')
    print(f'vectors: {online.shape[0]}')
    print(f'dimension: {online.shape[-1]}')
    print(f'online norm mean: {online_norm.mean():.6f}')
    print(f'ema norm mean: {ema_norm.mean():.6f}')
    print(f'online feature std across inputs: {online.std(dim=0).mean():.6f}')
    print(f'ema feature std across inputs: {ema.std(dim=0).mean():.6f}')
    print(f'online/ema cosine mean: {cosine.mean():.6f}')
    print(f'centered online/ema cosine mean: {centered_cosine.mean():.6f}')
    print(f'relative L2 mean: {relative_l2.mean():.6f}')
    print(f'coordinate MSE: {difference.square().mean():.6f}')


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError('--num-samples must be positive')
    if not Path(args.dataset).exists():
        raise FileNotFoundError(f'Dataset not found: {args.dataset}')

    device = torch.device(args.device)
    model = swm.wm.utils.load_pretrained(args.checkpoint)
    model = model.to(device).eval()
    dataset = swm.data.load_dataset(
        args.dataset,
        keys_to_cache=['pixels'],
    )
    num_samples = min(args.num_samples, len(dataset))
    # Dense video datasets contain strongly overlapping adjacent clips.
    # Sampling dataset[0:N] would therefore underestimate representation
    # diversity and could falsely look like collapse.  Draw indices from the
    # full table with a fixed seed so runs remain reproducible.
    indices = random.Random(args.seed).sample(range(len(dataset)), num_samples)
    transform = make_transform(args.img_size)
    features = collect_features(
        model, dataset, transform, indices, device
    )

    print(f'checkpoint: {args.checkpoint}')
    print(f'dataset: {args.dataset}')
    print(f'samples: {num_samples}')
    print(f'sampling seed: {args.seed}')
    print(f'index range: {min(indices)}..{max(indices)}')
    report('encoder CLS before projector', features[0], features[1])
    report('latent after projector', features[2], features[3])

    print('\nInterpretation:')
    print('- EMA feature std near zero indicates input-insensitive collapse.')
    print('- Low cosine with large relative L2 indicates direction mismatch,')
    print('  not merely a harmless scale difference.')
    print('- If the mismatch appears only after projector, inspect projector')
    print('  parameters and normalization buffers first.')


if __name__ == '__main__':
    main()
