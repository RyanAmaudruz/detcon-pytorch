import argparse
import os
import shutil

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from detcon.datasets import VOCSSLDataModule
from detcon.datasets.imagenette_module import ImagenetteDataModule
from detcon.datasets.s2c_data_module import S2cDataModule
from detcon.datasets.transforms import default_ssl_augs
from detcon.models import DetConB
import datetime


def main(cfg_path: str, cfg: DictConfig) -> None:
    # pl.seed_everything(0, workers=True)
    module = DetConB(**cfg.module)
    [print((k, v))  for k, v in cfg.module.items()]

    meta_df = pd.read_csv("/gpfs/scratch1/shared/ramaudruz/s2c_un/ssl4eo_s2_l1c_full_extract_metadata.csv")
    temp_var = meta_df['patch_id'].astype(str)
    meta_df['patch_id'] = temp_var.map(lambda x: (7 - len(x)) * '0' + x)
    meta_df['file_name'] = meta_df['patch_id'] + '/' + meta_df['timestamp']
    num_images = meta_df.shape[0]

    datamodule = S2cDataModule(
        train_transforms=default_ssl_augs,
        batch_size=cfg['datamodule']['batch_size'],
        meta_df=meta_df,
        num_workers=cfg['datamodule']['num_workers'],
        num_images=num_images
    )

    # datamodule = ImagenetteDataModule(
    #     train_transforms=default_ssl_augs,
    #     batch_size=cfg['datamodule']['batch_size'],
    #     meta_df=None,
    #     num_workers=16,
    #     num_images=None
    # )
    # datamodule = VOCSSLDataModule(**cfg.datamodule)

    timestamp = datetime.datetime.now().__str__().split('.')[0][:-3].replace(' ', '_').replace(':', '-')
    print(f'Timestamp: {timestamp}')
    checkpoint_dir = f'/gpfs/work5/0/prjs0790/data/run_outputs/checkpoints/odin/run_{timestamp}/'
    os.makedirs(checkpoint_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='ckp-{epoch:02d}',
        save_top_k=-1,
        verbose=True,
        every_n_epochs=1
    )
    callbacks = [checkpoint_callback]

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=WandbLogger(log_model="all"),
        # accumulate_grad_batches=20,
        gradient_clip_val=1,
        **cfg.trainer
    )
    trainer.fit(model=module, datamodule=datamodule)
    shutil.copyfile(cfg_path, os.path.join(checkpoint_dir, "config.yaml"))

class FakeArgs:
    cfg = 'conf/pretrain_new.yaml'


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--cfg", type=str, required=True, help="Path to config.yaml file"
    # )
    # args = parser.parse_args()
    # [print((k, getattr(args, k)))  for k in dir(args) if not k.startswith('_')]

    args = FakeArgs()

    cfg = OmegaConf.load(args.cfg)
    main(args.cfg, cfg)
