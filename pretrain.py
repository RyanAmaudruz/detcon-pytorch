import argparse
import os
import shutil

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import pandas as pd

from detcon.datasets import VOCSSLDataModule
from detcon.datasets.s2c_data_module import S2cDataModule
from detcon.datasets.transforms import default_ssl_augs
from detcon.models import DetConB


def main(cfg_path: str, cfg: DictConfig) -> None:
    pl.seed_everything(0, workers=True)
    module = DetConB(**cfg.module)

    meta_df = pd.read_csv("/gpfs/scratch1/shared/ramaudruz/s2c_un/ssl4eo_s2_l1c_full_extract_metadata.csv")
    temp_var = meta_df['patch_id'].astype(str)
    meta_df['patch_id'] = temp_var.map(lambda x: (7 - len(x)) * '0' + x)
    meta_df['file_name'] = meta_df['patch_id'] + '/' + meta_df['timestamp']
    num_images = meta_df.shape[0]

    datamodule = S2cDataModule(
        train_transforms=default_ssl_augs,
        batch_size=32,
        meta_df=meta_df,
        num_workers=16,
        num_images=num_images
    )
    # datamodule = VOCSSLDataModule(**cfg.datamodule)
    trainer = pl.Trainer(**cfg.trainer)
    trainer.fit(model=module, datamodule=datamodule)
    shutil.copyfile(cfg_path, os.path.join(trainer.logger.log_dir, "config.yaml"))

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
