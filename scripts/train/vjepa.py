import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from omegaconf import OmegaConf, open_dict
from stable_pretraining import data as dt

from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.loss import SIGReg
from stable_worldmodel.wm.utils import save_pretrained
from stable_worldmodel.wm.vjepa.module import gaussian_nll, unit_gaussian_kl


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(
        **imagenet_stats, source=source, target=target
    )
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class TargetEMACallback(Callback):
    """Update the target-mean encoder after each optimizer step."""

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        pl_module.model.update_target()


class SaveCkptCallback(Callback):
    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        if epoch % self.epoch_interval == 0:
            self._save(pl_module.model, epoch)
        if epoch == trainer.max_epochs and epoch % self.epoch_interval != 0:
            self._save(pl_module.model, epoch)

    def _save(self, model, epoch):
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )


def vjepa_forward(self, batch, stage, cfg):
    """Compute the single-sample variational JEPA objective."""
    context_length = cfg.wm.history_size
    prediction_offset = cfg.wm.num_preds
    beta = cfg.loss.beta
    sigreg_cfg = cfg.loss.get('sigreg', {})
    sigreg_enabled = sigreg_cfg.get('enabled', False)
    sigreg_weight = sigreg_cfg.get('weight', 0.0)

    action = torch.nan_to_num(batch['action'], 0.0)
    context = {
        'pixels': batch['pixels'][:, :context_length],
        'action': action[:, :context_length],
    }
    context = self.model.encode(context)

    target_pixels = batch['pixels'][:, prediction_offset:]
    target, target_mean, target_log_var = self.model.sample_target(
        target_pixels
    )
    pred_mean, pred_log_var = self.model.predict_distribution(
        context['emb'], context['act_emb']
    )

    nll = gaussian_nll(target, pred_mean, pred_log_var)
    kl = unit_gaussian_kl(target_mean, target_log_var)
    nll_loss = nll.mean()
    kl_loss = kl.mean()
    if sigreg_enabled:
        sigreg_loss = self.sigreg(context['emb'].transpose(0, 1))
    else:
        sigreg_loss = nll_loss.new_zeros(())
    loss = nll_loss + beta * kl_loss + sigreg_weight * sigreg_loss

    lower = self.model.pred_log_var_head.log_var_min
    upper = self.model.pred_log_var_head.log_var_max
    tolerance = 0.01 * (upper - lower)
    metrics = {
        'loss': loss,
        'nll_loss': nll_loss,
        'kl_loss': kl_loss,
        'weighted_kl_loss': beta * kl_loss,
        'sigreg_loss': sigreg_loss,
        'weighted_sigreg_loss': sigreg_weight * sigreg_loss,
        'pred_mse': (pred_mean.float() - target.float()).square().mean(),
        'target_mean_std': target_mean.float().std(),
        'pred_mean_std': pred_mean.float().std(),
        'target_log_var_mean': target_log_var.float().mean(),
        'target_log_var_min': target_log_var.float().min(),
        'target_log_var_max': target_log_var.float().max(),
        'pred_log_var_mean': pred_log_var.float().mean(),
        'pred_log_var_min': pred_log_var.float().min(),
        'pred_log_var_max': pred_log_var.float().max(),
        'pred_log_var_lower_saturation': (
            pred_log_var.float() <= lower + tolerance
        )
        .float()
        .mean(),
        'pred_log_var_upper_saturation': (
            pred_log_var.float() >= upper - tolerance
        )
        .float()
        .mean(),
    }
    self.log_dict(
        {f'{stage}/{key}': value.detach() for key, value in metrics.items()},
        on_step=True,
        sync_dist=True,
    )
    return metrics


@hydra.main(version_base=None, config_path='./config', config_name='vjepa')
def run(cfg):
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    print(
        f'Loading dataset "{dataset_name}" from '
        f'{"local cache: " + cache_dir if cache_dir else "default location"}'
    )
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )

    transforms = [
        get_img_preprocessor(
            source='pixels', target='pixels', img_size=cfg.img_size
        )
    ]
    with open_dict(cfg):
        for column in cfg.data.dataset.keys_to_load:
            if column.startswith('pixels'):
                continue
            transforms.append(get_column_normalizer(dataset, column, column))

        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim('action')
        )

    dataset.transform = spt.data.transforms.Compose(*transforms)
    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=generator,
    )
    full_train_size, full_val_size = len(train_set), len(val_set)
    train_set = swm.data.take_dataset_fraction(
        train_set, cfg.train_data_fraction
    )
    val_set = swm.data.take_dataset_fraction(val_set, cfg.val_data_fraction)
    print(
        f'Dataset split: train={len(train_set)}/{full_train_size}, '
        f'val={len(val_set)}/{full_val_size}'
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, **cfg.loader, generator=generator
    )
    val_cfg = {**cfg.loader, 'shuffle': False, 'drop_last': False}
    val_loader = torch.utils.data.DataLoader(val_set, **val_cfg)

    model = hydra.utils.instantiate(cfg.model)
    total_steps = cfg.trainer.max_epochs * len(train_loader)
    optimizers = {
        'model_opt': {
            'modules': 'model',
            'optimizer': dict(cfg.optimizer),
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(1, int(0.01 * total_steps)),
                'max_steps': total_steps,
            },
            'interval': 'epoch',
        }
    }
    module = spt.Module(
        model=model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(vjepa_forward, cfg=cfg),
        optim=optimizers,
    )
    data_module = spt.data.DataModule(
        train=train_loader, val=val_loader
    )

    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as config_file:
        OmegaConf.save(cfg, config_file)

    loggers = []
    if cfg.wandb.enabled:
        loggers.append(WandbLogger(**cfg.wandb.config))
    if cfg.tensorboard.enabled:
        loggers.append(
            TensorBoardLogger(
                save_dir=swm.data.utils.get_cache_dir(
                    sub_folder='tensorboard'
                ),
                name=cfg.output_model_name,
                version=run_id or None,
                default_hp_metric=False,
            )
        )
    for experiment_logger in loggers:
        experiment_logger.log_hyperparams(OmegaConf.to_container(cfg))
    logger = loggers if len(loggers) > 1 else (loggers[0] if loggers else None)

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[
            TargetEMACallback(),
            SaveCkptCallback(
                run_name=cfg.output_model_name,
                cfg=cfg.model,
                epoch_interval=1,
            ),
        ],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )
    ckpt_path = run_dir / f'{cfg.output_model_name}_weights.ckpt'
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )
    manager()


if __name__ == '__main__':
    run()
