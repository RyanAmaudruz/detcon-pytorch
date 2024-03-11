import pandas as pd
import pytorch_lightning as pl
import random
import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional

# from data.imagenet import logger
import numpy as np
import h5py

S2C_MEAN = [1605.57504906, 1390.78157673, 1314.8729939, 1363.52445545, 1549.44374991, 2091.74883118, 2371.7172463, 2299.90463006, 2560.29504086, 830.06605044, 22.10351321, 2177.07172323, 1524.06546312]

S2C_STD = [786.78685367, 850.34818441, 875.06484736, 1138.84957046, 1122.17775652, 1161.59187054, 1274.39184232, 1248.42891965, 1345.52684884, 577.31607053, 51.15431158, 1336.09932639, 1136.53823676]

S2C_MEAN_NEW = [x / 10000.0 * 255.0 for x in S2C_MEAN]

S2C_STD_NEW = [x / 10000.0 * 255.0 for x in S2C_STD]

class S2cDataModule(pl.LightningDataModule):

    def __init__(self,
                 num_workers: int,
                 batch_size: int,
                 meta_df: pd.DataFrame,
                 train_transforms,
                 num_images: int,
                 val_transforms = None,
                 size_val_set: int = 10):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.size_val_set = size_val_set
        self.meta_df = meta_df
        self.patch_id_list = self.meta_df['patch_id'].unique().tolist()
        self.num_images = len(self.patch_id_list)
        # self.num_images = num_images
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.im_train = None
        self.im_val = None
        self.file_list = meta_df['file_name'].tolist()
        random.shuffle(self.file_list)
        # logger.info(f"Found {len(self.file_list)} many images")
        print(f"Found {len(self.file_list)} many images")

    def __len__(self):
        # return len(self.file_list)
        return len(self.patch_id_list)

    def setup(self, stage: Optional[str] = None):
        # Split test set in val an test
        if stage == 'fit' or stage is None:
            self.im_train = UnlabelledSc2(self.patch_id_list, self.train_transforms)
            assert len(self.im_train) == self.num_images
            print(f"Train size {len(self.im_train)}")
        else:
            raise NotImplementedError("There is no dedicated val/test set.")
        # logger.info(f"Data Module setup at stage {stage}")
        print(f"Data Module setup at stage {stage}")

    def train_dataloader(self):
        return DataLoader(self.im_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          drop_last=True, pin_memory=True)


class UnlabelledSc2(Dataset):

    def __init__(self, file_list, transforms):
        self.file_names = file_list
        self.transform = transforms

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        patch_id = self.file_names[idx]
        with h5py.File('/gpfs/scratch1/shared/ramaudruz/s2c_un/s2c_264_light_new.h5', 'r') as f:
            data = np.array(f.get(patch_id))

        # data = data.astype('float32')

        # for i, (s2c_mean, s2c_std) in enumerate(zip(S2C_MEAN_NEW, S2C_STD_NEW)):
        #     data[:, i, :, :] = (data[:, i, :, :] - s2c_mean) / s2c_std
        # img_path = self.file_names[idx]
        # image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(data)
        return image

