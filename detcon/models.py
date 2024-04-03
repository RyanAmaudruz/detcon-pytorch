from typing import Dict, Sequence, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from einops import rearrange
# from torch_ema import ExponentialMovingAverage
from sklearn.cluster import KMeans
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import os
import pickle

from detcon.swin.swin_transformer import SwinTransformer

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from torchvision.transforms import InterpolationMode

from detcon.datasets.s2c_data_module import GaussianBlur, Solarize, RandomBrightness, RandomContrast, RandomSaturation, \
    RandomHue, ToGray

from detcon.losses import DetConBLoss
from dino.vision_transformer import vit_small


class MLP(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            # nn.SyncBatchNorm(hidden_dim),
            # nn.BatchNorm3d(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )


class Encoder(nn.Sequential):
    def __init__(self, backbone: str = "resnet50", pretrained: bool = False) -> None:

        model = vit_small(patch_size=16, num_classes=0, in_chans=13)
        # model = getattr(torchvision.models, backbone)(pretrained)
        # self.emb_dim = model.fc.in_features
        self.emb_dim = model.embed_dim
        model.fc = nn.Identity()
        model.avgpool = nn.Identity()
        super().__init__(*list(model.children()))


class MaskPooling(nn.Module):
    def __init__(
        self, num_classes: int, num_samples: int = 16, downsample: int = 32
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.mask_ids = torch.arange(num_classes)
        self.pool = nn.AvgPool2d(kernel_size=downsample, stride=downsample)

    # def pool_masks(self, masks: torch.Tensor) -> torch.Tensor:
    #     """Create binary masks and performs mask pooling
    #
    #     Args:
    #         masks: (b, 1, h, w)
    #
    #     Returns:
    #         masks: (b, num_classes, d)
    #     """
    #     if masks.ndim < 4:
    #         masks = masks.unsqueeze(dim=1)
    #
    #     masks = masks == self.mask_ids[None, :, None, None].to(masks.device)
    #     masks = self.pool(masks.to(torch.float))
    #     masks = rearrange(masks, "b c h w -> b c (h w)")
    #     masks = torch.argmax(masks, dim=1)
    #     masks = torch.eye(self.num_classes).to(masks.device)[masks]
    #     masks = rearrange(masks, "b d c -> b c d")
    #     return masks

    def pool_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Create binary masks and performs mask pooling

        Args:
            masks: (b, 1, h, w)

        Returns:
            masks: (b, num_classes, d)
        """
        if masks.ndim < 4:
            masks = masks.unsqueeze(dim=1)

        # masks = masks == self.mask_ids[None, :, None, None].to(masks.device)
        masks = self.pool(masks.to(torch.float))
        masks = rearrange(masks, "b c h w -> b c (h w)")
        masks = torch.argmax(masks, dim=1)
        masks = torch.eye(self.num_classes).to(masks.device)[masks]
        masks = rearrange(masks, "b d c -> b c d")
        return masks

    # def pool_masks(self, masks: torch.Tensor) -> torch.Tensor:
    #     """Create binary masks and performs mask pooling
    #
    #     Args:
    #         masks: (b, 1, h, w)
    #
    #     Returns:
    #         masks: (b, num_classes, d)
    #     """
    #     if masks.ndim < 4:
    #         masks = masks.unsqueeze(dim=1)
    #
    #     # masks = masks == self.mask_ids[None, :, None, None].to(masks.device)
    #     masks = self.pool(masks.to(torch.float))
    #     masks = rearrange(masks, "b c h w -> b c (h w)")
    #     masks = torch.argmax(masks, dim=1)
    #     masks = torch.eye(self.num_classes).to(masks.device)[masks]
    #     masks = rearrange(masks, "b d c -> b c d")
    #     return masks

    # def sample_masks(self, masks: torch.Tensor) -> torch.Tensor:
    #     """Samples which binary masks to use in the loss.
    #
    #     Args:
    #         masks: (b, num_classes, d)
    #
    #     Returns:
    #         masks: (b, num_samples, d)
    #     """
    #     bs = masks.shape[0]
    #
    #     # pooled_masks = self.pool(masks)
    #
    #     # masks_sum = masks.sum((2, 3)) + 1e-11
    #     # masks_sum2 = masks_sum / masks_sum.sum(dim=1, keepdim=True)
    #     # mask_ids = torch.multinomial(masks_sum2, num_samples=self.num_samples)
    #     # sampled_masks = torch.stack([masks[b][mask_ids[b]] for b in range(bs)])
    #
    #     mask_exists = torch.greater(masks.sum(dim=-1), 1e-3)
    #     sel_masks = mask_exists.to(torch.float) + 1e-11
    #
    #     mask_ids = torch.multinomial(sel_masks, num_samples=self.num_samples)
    #     sampled_masks = torch.stack([masks[b][mask_ids[b]] for b in range(bs)])
    #
    #
    #     # mask_exists = torch.greater(masks.sum((2, 3)), 1e-3)
    #     # sel_masks = mask_exists.to(torch.float) + 1e-11
    #     #
    #     #
    #     # mask_exists = torch.greater(masks.sum(dim=-1), 1e-3)
    #     # sel_masks = mask_exists.to(torch.float) + 1e-11
    #     # # torch.multinomial handles normalizing
    #     # sel_masks = sel_masks / sel_masks.sum(dim=1, keepdim=True)
    #     # sel_masks = torch.softmax(sel_masks, dim=-1)
    #     # mask_ids = torch.multinomial(sel_masks, num_samples=self.num_samples)
    #     # sampled_masks = torch.stack([masks[b][mask_ids[b]] for b in range(bs)])
    #     return sampled_masks, mask_ids

    def sample_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Samples which binary masks to use in the loss.

        Args:
            masks: (b, num_classes, d)

        Returns:
            masks: (b, num_samples, d)
        """
        bs = masks.shape[0]

        # pooled_masks = self.pool(masks)

        masks_sum = masks.sum((2, 3)) + 1e-11
        masks_sum2 = masks_sum / masks_sum.sum(dim=1, keepdim=True)
        mask_ids = torch.multinomial(masks_sum2, num_samples=self.num_samples)
        sampled_masks = torch.stack([masks[b][mask_ids[b]] for b in range(bs)])


        # mask_exists = torch.greater(masks.sum((2, 3)), 1e-3)
        # sel_masks = mask_exists.to(torch.float) + 1e-11
        #
        #
        # mask_exists = torch.greater(masks.sum(dim=-1), 1e-3)
        # sel_masks = mask_exists.to(torch.float) + 1e-11
        # # torch.multinomial handles normalizing
        # sel_masks = sel_masks / sel_masks.sum(dim=1, keepdim=True)
        # sel_masks = torch.softmax(sel_masks, dim=-1)
        # mask_ids = torch.multinomial(sel_masks, num_samples=self.num_samples)
        # sampled_masks = torch.stack([masks[b][mask_ids[b]] for b in range(bs)])
        return sampled_masks, mask_ids

    def sample_masks2(self, masks: torch.Tensor) -> torch.Tensor:

        bs = masks.shape[0]

        # Check for existing masks
        mask_exists = (masks > 1e-3).any(-1).any(-1)

        # Create mask weights (add small epsilon for numerical stability)
        sel_masks = mask_exists.float() + 1e-8
        sel_masks = sel_masks / sel_masks.sum(dim=1, keepdim=True)
        sel_masks = torch.log(sel_masks)

        # Sample mask indices using categorical distribution
        dist = torch.distributions.Categorical(logits=sel_masks)
        mask_ids = dist.sample(sample_shape=(bs, self.num_samples))

        # Gather sampled masks
        smpl_masks = torch.stack(
            [masks[b][mask_ids[b]] for b in range(bs)]
        )
        return smpl_masks, mask_ids

    # def forward(self, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     binary_masks = self.pool_masks(masks)
    #     sampled_masks, sampled_mask_ids = self.sample_masks(binary_masks)
    #     area = sampled_masks.sum(dim=-1, keepdim=True)
    #     sampled_masks = sampled_masks / torch.maximum(area, torch.tensor(1.0))
    #     return sampled_masks, sampled_mask_ids




    def forward(self, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # binary_masks = self.pool_masks(masks)
        pooled_masks = self.pool(masks)
        sampled_masks, sampled_mask_ids = self.sample_masks(pooled_masks)
        area = sampled_masks.sum(dim=(2, 3), keepdim=True)
        sampled_masks = sampled_masks / torch.maximum(area, torch.tensor(1.0))
        return sampled_masks, sampled_mask_ids


class Network(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = False,
        # hidden_dim: int = 128,
        hidden_dim: int = 300,
        output_dim: int = 256,
        num_classes: int = 10,
        downsample: int = 32,
        num_samples: int = 16,
    ) -> None:
        super().__init__()
        # self.encoder = Encoder(backbone, pretrained)
        # self.encoder = vit_small(patch_size=16, num_classes=21, in_chans=13)
        self.encoder = SwinTransformer(
            img_size=256,
            patch_size=16,
            in_chans=13,
            # embed_dim=384,
            embed_dim=768,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            window_size=8,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            norm_befor_mlp='ln',
        )



# self.encoder = vit_small(patch_size=16, num_classes=21, in_chans=3)
#         if pretrained:
#             load_pretrained_weights(
#                 self.encoder,
#                 '/gpfs/work5/0/prjs0790/data/run_outputs/checkpoints/ssl4eo_ssl/ssl_s2c_new_transforms/checkpoint0095.pth',
#                 'teacher'
#             )
        self.projector = MLP(self.encoder.embed_dim, 300, output_dim)
        self.mask_pool = MaskPooling(num_classes, num_samples, downsample)

    # def forward(self, x: torch.Tensor, masks: torch.Tensor) -> Sequence[torch.Tensor]:
    #     m, mids = self.mask_pool(masks)
    #     e = self.encoder(x)
    #     e = e[:, 1:, :]
    #     # e = rearrange(e, "b c h w -> b (h w) c")
    #     e = m @ e
    #     p = self.projector(e)
    #     return e, p, m, mids


    def forward(self, x: torch.Tensor, masks: torch.Tensor) -> Sequence[torch.Tensor]:
        m, mids = self.mask_pool(masks)
        nb, ns = m.shape[:2]
        e = self.encoder(x)
        e = e[:, 1:, :]
        # e = rearrange(e, "b c h w -> b (h w) c")
        e = m.reshape(nb, ns, -1) @ e
        p = self.projector(e)
        return e, p, m, mids


class Custom_Transform(torch.nn.Module):
    """
    TODO
    """

    def __init__(self, size, scale=(0.4, 0.85), ratio=(3/4, 4/3), interpolation='BILINEAR'):
        super().__init__()
        self.size = (size, size)

        self.interpolation = InterpolationMode[interpolation]
        self.scale = scale
        self.ratio = ratio
        self.corner_map = {0: 3, 1: 2, 3: 0, 2: 1}
        self.selected_corner = None

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if self.selected_corner is None:
            self.selected_corner = np.random.randint(0, 4)
            img = self.crop_corner(img, self.selected_corner)
        else:
            img = self.crop_opposite_corner(img, self.selected_corner)
            self.selected_corner = None
        return img


    def crop_corner(self, img, selected_corner):
        h, w = self.get_params(img, self.scale, self.ratio)
        if selected_corner == 0:
            i = 0
            j = 0
        elif selected_corner == 1:
            i = img.shape[-2] - h
            j = 0
        elif selected_corner == 2:
            i = 0
            j = img.shape[-1] - w
        elif selected_corner == 3:
            i = img.shape[-2] - h
            j = img.shape[-1] - w
        else:
            raise ValueError('Case should not occur!')
        return torchvision.transforms.functional.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def crop_opposite_corner(self, img, selected_corner):
        opposite_corner = self.corner_map[selected_corner]
        return self.crop_corner(img, opposite_corner)

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (CV Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.shape[-2] * img.shape[-1]
            target_area = np.random.uniform(*scale) * area
            aspect_ratio = np.random.uniform(*ratio)

            w = int(round(np.sqrt(target_area * aspect_ratio)))
            h = int(round(np.sqrt(target_area / aspect_ratio)))

            if np.random.random() < 0.5:
                w, h = h, w

            if w <= img.shape[-2] and h <= img.shape[-1]:
                return h, w

        # Fallback
        w = min(img.shape[0], img.shape[1])
        return w, w

    def __repr__(self):
        interpolate_str = self.interpolation
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string

class DetConB(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 21,
        num_samples: int = 5,
        backbone: str = "resnet50",
        pretrained: bool = False,
        downsample: int = 32,
        proj_hidden_dim: int = 300,
        proj_dim: int = 256,
        loss_fn: nn.Module = DetConBLoss(),
    ) -> None:
        super().__init__()
        self.step_count = 0
        self.save_hyperparameters(ignore=["loss_fn"])
        self.loss_fn = loss_fn
        self.num_classes = num_classes
        self.online_network = Network(
            backbone=backbone,
            pretrained=pretrained,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_dim,
            num_classes=num_classes,
            downsample=downsample,
            num_samples=num_samples,
        )
        self.ema_network = Network(
            backbone=backbone,
            pretrained=pretrained,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_dim,
            num_classes=num_classes,
            downsample=downsample,
            num_samples=num_samples,
        )

        for p in self.ema_network.parameters():
            p.requires_grad = False

        # self.ema = ExponentialMovingAverage(self.network.parameters(), decay=0.995)
        self.predictor = MLP(256, 256, 256)
        # self.enc_mlp = MLP(384, 384, 384)


        flip_and_color_jitter = transforms.Compose([
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                # RandomBrightness(0.8),
                # RandomContrast(0.8),
                # RandomSaturation(0.8),
                # RandomHue(0.2)
                RandomBrightness(0.4),
                RandomContrast(0.4),
                RandomSaturation(0.2),
                RandomHue(0.1)
            ], p=0.8),
            transforms.RandomApply([ToGray(13)], p=0.2),
        ])


        # flip_and_color_jitter = transforms.Compose([
        #     # transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandomApply(transforms.ColorJitter(0.4, 0.4, 0.2, 0.1), p=0.8),
        #     transforms.RandomApply([ToGray(3)], p=0.2),
        # ])


        DEFAULT_AUG = transforms.Compose([
            # transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=InterpolationMode.BICUBIC),
            # flip_and_color_jitter,
            flip_and_color_jitter,
            # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            # transforms.RandomApply([ToGray(13)], p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            transforms.RandomApply([Solarize()], p=0.2),
            # transforms.RandomApply([torchvision.transforms.RandomSolarize(0.5)], p=0.2),
            # normalize,
        ])
        self.augment1 = DEFAULT_AUG
        self.augment2 = DEFAULT_AUG
        self.crop_flip = transforms.Compose([
            # transforms.RandomResizedCrop(size=(224, 224), scale=(0.08, 1)),
            Custom_Transform(224),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip()
        ])

    def configure_optimizers(self) -> torch.optim.Optimizer:
        proj_params = {'params': [p for n, p in self.named_parameters() if 'projector' in n], 'lr': 1e-3}
        pred_params = {'params': [p for n, p in self.named_parameters() if 'predictor' in n], 'lr': 1e-3}
        encoder_params = {'params': [p for n, p in self.named_parameters() if 'encoder' in n], 'lr': 1e-4}
        # enc_mlp_params = {'params': [p for n, p in self.named_parameters() if 'enc_mlp' in n], 'lr': 1e-3}

        return torch.optim.Adam(
            # [proj_params, pred_params, encoder_params, enc_mlp_params],
            [proj_params, pred_params, encoder_params],
            lr=1e-3
        )
        # return torch.optim.Adam(self.parameters(), lr=1e-3)

    def on_before_zero_grad(self, *args, **kwargs):
        # """See https://forums.pytorchlightning.ai/t/adopting-exponential-moving-average-ema-for-pl-pipeline/488"""  # noqa: E501
        # self.ema.to(device=next(self.network.parameters()).device)
        # self.ema.update(self.network.parameters())
        with torch.no_grad():
            m = 0.995
            for param_q, param_k in zip(self.online_network.parameters(), self.ema_network.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)


    def forward(self, x: torch.Tensor, y: torch.Tensor, ema=False) -> Sequence[torch.Tensor]:
        if ema:
            return self.ema_network(x, y)
        else:
            return self.online_network(x, y)

    def binarise_masks(self, mask: torch.Tensor):
        return F.one_hot(mask.argmax(1), num_classes=self.num_classes).permute(0, 3, 1, 2).float()

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        if self.step_count == 0:
            for n, p in self.online_network.encoder.named_parameters():
                if n not in ['head.weight', 'head.bias']:
                    p.requires_grad = False
        elif self.step_count == 100:
            for n, p in self.online_network.encoder.named_parameters():
                p.requires_grad = True

        self.step_count += 1

        features = self.ema_network.encoder(batch)

        # features = self.enc_mlp(features)

        fea_wo_cls = features[:, 1:, :]
        masks = self.get_masks_from_features(fea_wo_cls)

        _, n_masks, img_size1, img_size2 = masks.shape

        x_aug1 = self.augment1(batch)
        x_aug2 = self.augment2(batch)

        x_aug1_plus_mask = torch.concat([x_aug1, masks], axis=1)
        x_aug2_plus_mask = torch.concat([x_aug2, masks], axis=1)

        x_aug1_plus_mask_cro = self.crop_flip(x_aug1_plus_mask)
        x_aug2_plus_mask_cro = self.crop_flip(x_aug2_plus_mask)

        x_aug1 = x_aug1_plus_mask_cro[:, :-n_masks, :, :]
        x_aug2 = x_aug2_plus_mask_cro[:, :-n_masks, :, :]

        masks_1 = x_aug1_plus_mask_cro[:, -n_masks:, :, :]
        masks_2 = x_aug2_plus_mask_cro[:, -n_masks:, :, :]

        # masks_1 = self.binarise_masks(masks_1)
        # masks_2 = self.binarise_masks(masks_2)

        # (x1, x2), (y1, y2) = batch["image"], batch["mask"]

        # encode and project
        _, p1, _, ids1 = self(x_aug1, masks_1)
        _, p2, _, ids2 = self(x_aug2, masks_2)

        # # ema encode and project
        # with self.ema.average_parameters():
        _, ema_p1, _, ema_ids1 = self(x_aug1, masks_1, ema=True)
        _, ema_p2, _, ema_ids2 = self(x_aug2, masks_2, ema=True)

        # predict
        q1, q2 = self.predictor(p1), self.predictor(p2)

        # compute loss
        loss = self.loss_fn(
            pred1=q1,
            pred2=q2,
            target1=ema_p1.detach(),
            target2=ema_p2.detach(),
            pind1=ids1,
            pind2=ids2,
            tind1=ema_ids1,
            tind2=ema_ids2,
            step_count=self.step_count
        )
        self.log("temperature", self.loss_fn.temperature)
        self.log("loss", loss)
        # if self.step_count % 50 == 0:
        #     # print("temperature: " + str(self.loss_fn.temperature))
        #     print("loss: " + str(loss))


        return loss

    def get_masks_from_features(self, fea_wo_cls):
        image_size = 448
        device = 'cuda'
        cluster_pred_list = []
        for i in range(fea_wo_cls.shape[0]):
            kmeans = KMeans(
                init="random",
                n_clusters=self.num_classes,
                n_init=1,
                max_iter=1000,
                random_state=42
            )
            fea_wo_cls_cpu =  fea_wo_cls[i].detach().cpu()
            cluster_pred = kmeans.fit_predict(fea_wo_cls_cpu)
            cluster_pred_list.append(cluster_pred.reshape(1, -1))

        clusters_conc = np.concatenate(cluster_pred_list, axis=0)
        n_patch_side = int(image_size / 16)
        masks = torch.from_numpy(clusters_conc).reshape(-1, 1, n_patch_side, n_patch_side).to(device)
        resized_masks = F.interpolate(masks.float(), size=(image_size, image_size), mode='nearest')
        binary_masks = F.one_hot(resized_masks.long(), num_classes=self.num_classes)
        return binary_masks.permute(0, 1, 4, 2, 3).reshape(-1, self.num_classes, image_size, image_size)


def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            temp_state_dict = {
                k.replace('teacher.', ''): v for k, v in state_dict.items()
                if k.startswith('teacher') and 'projection_head.' not in k and 'prototypes.' not in k
            }
            if temp_state_dict:
                state_dict = temp_state_dict
            else:
                temp_state_dict = {
                    k.replace('module.momentum_encoder.', ''): v for k, v in state_dict.items()
                    if k.startswith('module.momentum_encoder.') and 'projection_head.' not in k and 'prototypes.' not in k
                }
                if temp_state_dict:
                    state_dict = temp_state_dict
                else:
                    state_dict = {
                        k.replace('momentum_encoder.', ''): v for k, v in state_dict.items()
                        if k.startswith('momentum_encoder.') and 'projection_head.' not in k and 'prototypes.' not in k
                    }
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            state_dict = {k[4:]: v for k, v in state_dict.items() if k.startswith('net.')}
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        raise ValueError('Checkpoint file does not exist!')
