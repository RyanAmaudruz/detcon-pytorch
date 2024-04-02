import time

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as pth_transforms
import random
from copy import deepcopy
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

import pytorch_lightning as pl
import multiprocessing
from torchvision import datasets, transforms

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode

from detcon.datasets.s2c_data_module import S2cDataModule, RandomBrightness, RandomContrast, RandomSaturation, \
    RandomHue, ToGray, GaussianBlur, Solarize
from detcon.datasets.transforms import default_ssl_augs
from dino.vision_transformer import vit_small


# MLP class for projector and predictor
def MLP(dim=384, projection_size=256, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


class RandomApply(torch.nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


class NetWrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.projector = MLP()
        self.activation = {}
        self.hook_registered = False

    def _register_hook(self):
        self.net.layer4.register_forward_hook(self._get_activation('h'))

    def _get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

    def get_representation(self, inputs):
        if not self.hook_registered:
            self._register_hook()
        _ = self.net(inputs)
        return self.activation['h']

    def get_masks(self, inputs):
        n_clusters = 5
        image_size = 224
        device = 'cuda'
        features = self.net(inputs)
        # features = self.get_representation(inputs)

        fea_wo_cls = features[:, 1:, :]

        cluster_pred_list = []

        for i in range(fea_wo_cls.shape[0]):


            kmeans = KMeans(
                init="random",
                n_clusters=n_clusters,
                n_init=3,
                max_iter=500,
                random_state=42
            )
            fea_wo_cls_cpu =  fea_wo_cls[i].cpu()

            cluster_pred = kmeans.fit_predict(fea_wo_cls_cpu)

            cluster_pred_list.append(cluster_pred.reshape(1, -1))


        clusters_conc = np.concatenate(cluster_pred_list, axis=0)

        masks = torch.from_numpy(clusters_conc).reshape(-1, 1, 14, 14).to(device)

        resized_masks = F.interpolate(masks.float(), size=(image_size, image_size), mode='nearest')

        binary_masks = F.one_hot(resized_masks.long(), num_classes=n_clusters)


        # masks = []
        # for feature in features:
        #
        #     num_channels, num_x, num_y = feature.shape
        #     normalized_feature = feature.reshape(num_channels, num_x * num_y)
        #     normalized_feature = normalized_feature.T.cpu()
        #     nn.functional.normalize(normalized_feature, p = 2, dim = 1)
        #
        #     kmeans = KMeans(
        #         init="random",
        #         n_clusters=2,
        #         n_init=10,
        #         max_iter=1000,
        #         random_state=42)
        #
        #     labels = kmeans.fit_predict(normalized_feature)
        #     labels = labels.reshape(1, num_x, num_y)
        #
        #     # Extend the mask to the original size of the image, so that we can feed
        #     # both the image and the mask through the image augmentation pipeline.
        #     # This is necessary such that both the image and the mask are augmented
        #     # in the exact same way (due to the inherent randomness of augmentation).
        #     labels = nn.functional.interpolate(torch.FloatTensor(labels).unsqueeze(0),
        #                                        size=(image_size, image_size), mode="nearest")[0]
        #
        #     expanded_mask = torch.FloatTensor(labels).repeat(inputs[0].shape[0], 1, 1).to(device)
        #
        #     masks.append(expanded_mask)
        # masks = torch.stack(masks)

        return binary_masks.permute(0, 1, 4, 2, 3).reshape(-1, n_clusters, image_size, image_size)
        # return binary_masks


    def forward(self, inputs, masks_obj1, masks_obj2):
        features = self.net(inputs)
        features = features[:, 1:, :]

        features_per = features.permute(0, 2, 1).reshape(-1, 384, 14, 14)

        features_full = F.interpolate(features_per, size=(224, 224), mode='nearest')

        mask1_mean = (masks_obj1[:, None, :, :] * features_full).sum(axis=(2, 3)) / (masks_obj1[:, None, :, :]).sum(axis=(2, 3))
        mask2_mean = (masks_obj2[:, None, :, :] * features_full).sum(axis=(2, 3)) / (masks_obj2[:, None, :, :]).sum(axis=(2, 3))

        return self.projector(mask1_mean), self.projector(mask2_mean)







        # mask_pooled_features_obj1 = []
        # mask_pooled_features_obj2 = []
        #
        #
        #
        # for idx, feature in enumerate(features):
        #     num_channels, num_x, num_y = feature.shape
        #     feature = feature.reshape(num_channels, num_x * num_y)
        #
        #     # The mask was previously resized to the initial size of images so that
        #     # it can be fed to the augmentation pipeline. Now, we want to apply the
        #     # masks to the feature output by the network, so we have to resize the
        #     # masks to their original sizes in order to be able to apply them.
        #     resized_mask_obj1 = nn.functional.interpolate(masks_obj1[idx][0].unsqueeze(0).unsqueeze(0),
        #                                                   size=(num_x, num_y), mode="nearest")[0]
        #     resized_mask_obj2 = nn.functional.interpolate(masks_obj2[idx][0].unsqueeze(0).unsqueeze(0),
        #                                                   size=(num_x, num_y), mode="nearest")[0]
        #
        #
        #     resized_mask_obj1 = resized_mask_obj1.repeat(num_channels, 1, 1)
        #     resized_mask_obj2 = resized_mask_obj2.repeat(num_channels, 1, 1)
        #
        #     resized_mask_obj1 = resized_mask_obj1.reshape(num_channels, num_x * num_y)
        #     resized_mask_obj2 = resized_mask_obj2.reshape(num_channels, num_x * num_y)
        #
        #     mask_pooled_feature_obj1 = torch.mean(torch.mul(feature, resized_mask_obj1), dim=(1))
        #     mask_pooled_feature_obj2 = torch.mean(torch.mul(feature, resized_mask_obj2), dim=(1))
        #
        #     mask_pooled_features_obj1.append(mask_pooled_feature_obj1)
        #     mask_pooled_features_obj2.append(mask_pooled_feature_obj2)
        #
        # mask_pooled_features_obj1 = torch.stack(mask_pooled_features_obj1)
        # mask_pooled_features_obj2 = torch.stack(mask_pooled_features_obj2)
        #
        # return self.projector(mask_pooled_features_obj1), self.projector(mask_pooled_features_obj2)


class Odin(torch.nn.Module):
    def __init__(self, net, moving_average_decay = 0.99):
        super().__init__()

        image_size = 256
        device = 'cuda'

        # DEFAULT_AUG = torch.nn.Sequential(
        #     RandomApply(
        #         pth_transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
        #         p = 0.3
        #     ),
        #     pth_transforms.RandomGrayscale(p=0.2),
        #     pth_transforms.RandomHorizontalFlip(),
        #     RandomApply(
        #         pth_transforms.GaussianBlur((3, 3), (1.0, 2.0)),
        #         p = 0.2
        #     ),
        #     pth_transforms.RandomResizedCrop((image_size, image_size)),
        #     pth_transforms.Normalize(
        #         mean=torch.tensor([0.485, 0.456, 0.406]),
        #         std=torch.tensor([0.229, 0.224, 0.225])),
        # )
        global_crops_scale = (0.4, 1.0)
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                RandomBrightness(0.4),
                RandomContrast(0.4),
                RandomSaturation(0.2),
                RandomHue(0.1)
            ], p=0.8),
            transforms.RandomApply([ToGray(13)], p=0.2),
        ])

        DEFAULT_AUG = transforms.Compose([
            # transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            transforms.RandomApply([Solarize()], p=0.2),
            # normalize,
        ])

        self.cropping = transforms.RandomResizedCrop(size=(224, 224), scale=(0.7, 0.9))

        self.augment1 = DEFAULT_AUG
        self.augment2 = DEFAULT_AUG

        self.theta_encoder = NetWrapper(net)
        self.theta_predictor = MLP(256, 256, 4096).to(device)

        self.tau_encoder = deepcopy(self.theta_encoder).to(device)
        set_requires_grad(self.tau_encoder, False)

        self.csi_encoder = deepcopy(self.theta_encoder).to(device)
        set_requires_grad(self.csi_encoder, False)

        self.theta_encoder = self.theta_encoder.to(device)

        self.target_ema_updater = EMA(moving_average_decay)

    def update_moving_average(self):
        assert self.tau_encoder is not None, 'target encoder has not been created yet'
        assert self.csi_encoder is not None, 'target encoder has not been created yet'

        update_moving_average(self.target_ema_updater, self.tau_encoder, self.theta_encoder)
        update_moving_average(self.target_ema_updater, self.csi_encoder, self.theta_encoder)


    def forward(self, x):
        tau_encoder_masks = self.tau_encoder.get_masks(x)

        _, n_masks, img_size1, img_size2 = tau_encoder_masks.shape

        x_aug1 = self.augment1(x)
        x_aug2 = self.augment2(x)

        x_aug1_plus_mask = torch.concat([x_aug1, tau_encoder_masks], axis=1)
        x_aug2_plus_mask = torch.concat([x_aug2, tau_encoder_masks], axis=1)


        x_aug1_plus_mask_cro = self.cropping(x_aug1_plus_mask)
        x_aug2_plus_mask_cro = self.cropping(x_aug2_plus_mask)

        x_aug1 = x_aug1_plus_mask_cro[:, :-n_masks, :, :]
        x_aug2 = x_aug2_plus_mask_cro[:, :-n_masks, :, :]

        masks_1 = x_aug1_plus_mask_cro[:, -n_masks:, :, :]
        masks_2 = x_aug2_plus_mask_cro[:, -n_masks:, :, :]


        masks_1_sum = masks_1.float().sum(axis=(2, 3))
        masks_2_sum = masks_2.float().sum(axis=(2, 3))

        mask1_prob = (masks_1_sum + 10-5) / masks_1_sum.sum(1)[:, None]
        mask2_prob = (masks_2_sum + 10-5) / masks_2_sum.sum(1)[:, None]

        selected_masks1 = torch.multinomial(mask1_prob, 1)
        selected_masks2 = torch.multinomial(mask2_prob, 1)

        # selected_masks1 = torch.multinomial(mask1_prob, 2)
        # selected_masks2 = torch.multinomial(mask2_prob, 2)

        # mask_1_to_use_a = torch.stack([masks_1[i, m, :, :] for i, m in enumerate(selected_masks1[:, 0])]).reshape(-1, img_size1, img_size2)
        # mask_1_to_use_b = torch.stack([masks_1[i, m, :, :] for i, m in enumerate(selected_masks1[:, 0])]).reshape(-1, img_size1, img_size2)
        #
        # mask_2_to_use_a = torch.stack([masks_2[i, m, :, :] for i, m in enumerate(selected_masks2[:, 0])]).reshape(-1, img_size1, img_size2)
        # mask_2_to_use_b = torch.stack([masks_2[i, m, :, :] for i, m in enumerate(selected_masks2[:, 0])]).reshape(-1, img_size1, img_size2)

        mask_1_to_use = torch.stack([masks_1[i, m, :, :] for i, m in enumerate(selected_masks1)]).reshape(-1, img_size1, img_size2)
        mask_2_to_use = torch.stack([masks_2[i, m, :, :] for i, m in enumerate(selected_masks2)]).reshape(-1, img_size1, img_size2)

        z_theta_obj1_view1, z_theta_obj2_view1 = self.theta_encoder(x_aug1, mask_1_to_use, 1 - mask_1_to_use)
        pred_obj1_view1 = self.theta_predictor(z_theta_obj1_view1)
        pred_obj2_view1 = self.theta_predictor(z_theta_obj2_view1)

        z_theta_obj1_view2, z_theta_obj2_view2 = self.theta_encoder(x_aug2, mask_2_to_use, 1 - mask_2_to_use)
        pred_obj1_view2 = self.theta_predictor(z_theta_obj1_view2)
        pred_obj2_view2 = self.theta_predictor(z_theta_obj2_view2)

        with torch.no_grad():
            z_csi_obj1_view2, z_csi_obj2_view2 = self.csi_encoder(x_aug1, mask_1_to_use, 1 - mask_1_to_use)
            z_csi_obj1_view1, z_csi_obj2_view1 = self.csi_encoder(x_aug2, mask_2_to_use, 1 - mask_2_to_use)

        loss_obj1 = (
            loss_fn(pred_obj1_view1, z_csi_obj1_view2, z_csi_obj2_view2) +
            loss_fn(pred_obj1_view2, z_csi_obj1_view1, z_csi_obj2_view1)
        ) / 2
        loss_obj2 = (
            loss_fn(pred_obj2_view1, z_csi_obj2_view2, z_csi_obj1_view2) +
            loss_fn(pred_obj2_view2, z_csi_obj2_view1, z_csi_obj1_view1)
        ) / 2

        return (loss_obj1 + loss_obj2).mean()


        # view_masks_one = self.augment1(torch.concat([x, tau_encoder_masks]))
        # view_masks_two = self.augment2(torch.concat([x, tau_encoder_masks]))
        #
        # view_one = view_masks_one[0: len(x)]
        # view_one_masks = view_masks_one[len(x):]
        #
        # view_two = view_masks_two[0: len(x)]
        # view_two_masks = view_masks_two[len(x):]

        # z_theta_obj1_view1, z_theta_obj2_view1 = self.theta_encoder(view_one, view_one_masks, 1 - view_one_masks)
        # pred_obj1_view1 = self.theta_predictor(z_theta_obj1_view1)
        # pred_obj2_view1 = self.theta_predictor(z_theta_obj2_view1)
        #
        # z_theta_obj1_view2, z_theta_obj2_view2 = self.theta_encoder(view_two, view_two_masks, 1 - view_two_masks)
        # pred_obj1_view2 = self.theta_predictor(z_theta_obj1_view1)
        # pred_obj2_view2 = self.theta_predictor(z_theta_obj2_view1)
        #
        #
        # with torch.no_grad():
        #     z_csi_obj1_view2, z_csi_obj2_view2 = self.csi_encoder(view_two, view_two_masks, 1 - view_two_masks)
        #     z_csi_obj1_view1, z_csi_obj2_view1 = self.csi_encoder(view_one, view_one_masks, 1 - view_one_masks)
        #
        # # Every instance in the batch from pred_obj1 must be similar to z_csi_obj1 (same obj, diff. views, same img).
        # # But each one must be different from z_csi_obj2 (diff. object, diff. views, same image).
        # # Bacause the theta and csi encoders are different, we have to repeat the same steps for z_csi_obj1
        # #  from the second view too.
        # # Additionally, the same procedure must be done for pred_obj2 as well.
        #
        # loss_obj1 = (loss_fn(pred_obj1_view1, z_csi_obj1_view2, z_csi_obj2_view2) +
        #              loss_fn(pred_obj1_view2, z_csi_obj1_view1, z_csi_obj2_view1)) / 2
        # loss_obj2 = (loss_fn(pred_obj2_view1, z_csi_obj2_view2, z_csi_obj1_view2) +
        #              loss_fn(pred_obj2_view2, z_csi_obj2_view1, z_csi_obj1_view1)) / 2
        #
        # return (loss_obj1 + loss_obj2).mean()

def loss_fn(pred_obj1, z_csi_obj1, z_csi_obj2):
    pred_obj1 = F.normalize(pred_obj1, dim=-1, p=2)
    z_csi_obj1 = F.normalize(z_csi_obj1, dim=-1, p=2)
    z_csi_obj2 = F.normalize(z_csi_obj2, dim=-1, p=2)
    similar_instances = (pred_obj1 * z_csi_obj1).sum(dim=-1)
    dissimilar_instances = (pred_obj1 * z_csi_obj2).sum(dim=-1)

    return -torch.log(torch.exp(similar_instances) / torch.exp(similar_instances + dissimilar_instances))


class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = Odin(net)
        self.count = 0

    def forward(self, images):
        self.count += 1
        if self.count != 0 and self.count % 1000 == 0:
            torch.save({
                'step': self.count,
                'model_state_dict': self.learner.theta_encoder.state_dict(),
            }, f'odin_checkpoint_{self.count}.pth')
        return self.learner(images)

    def training_step(self, images, _):
        loss = self.forward(images)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    def on_before_zero_grad(self, _):
        self.learner.update_moving_average()

if __name__ == '__main__':

    BATCH_SIZE = 12
    EPOCHS     = 60
    LR         = 3e-4
    NUM_GPUS   = 1
    NUM_WORKERS = multiprocessing.cpu_count()
    device = 'cuda'

    model = vit_small(patch_size=16, num_classes=21, in_chans=13)

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

    train_loader = datamodule
    # train_loader = DataLoader(dataset.to(device), batch_size=BATCH_SIZE, shuffle=True)
    # model = model.to(device)
    model.train()
    odin_model = SelfSupervisedLearner(model)

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        gpus=NUM_GPUS,
        accumulate_grad_batches=1,
        sync_batchnorm=True
    )

    trainer.fit(odin_model, train_loader)





