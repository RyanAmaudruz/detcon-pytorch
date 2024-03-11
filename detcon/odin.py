
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as pth_transforms
import random
from copy import deepcopy
from sklearn.cluster import KMeans

# MLP class for projector and predictor
def MLP(dim=512, projection_size=256, hidden_size=4096):
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
        features = self.get_representation(inputs)

        masks = []

        for feature in features:

            num_channels, num_x, num_y = feature.shape
            normalized_feature = feature.reshape(num_channels, num_x * num_y)
            normalized_feature = normalized_feature.T.cpu()
            nn.functional.normalize(normalized_feature, p = 2, dim = 1)

            kmeans = KMeans(
                init="random",
                n_clusters=2,
                n_init=10,
                max_iter=1000,
                random_state=42)

            labels = kmeans.fit_predict(normalized_feature)
            labels = labels.reshape(1, num_x, num_y)

            # Extend the mask to the original size of the image, so that we can feed
            # both the image and the mask through the image augmentation pipeline.
            # This is necessary such that both the image and the mask are augmented
            # in the exact same way (due to the inherent randomness of augmentation).
            labels = nn.functional.interpolate(torch.FloatTensor(labels).unsqueeze(0),
                                               size=(image_size, image_size), mode="nearest")[0]

            expanded_mask = torch.FloatTensor(labels).repeat(inputs[0].shape[0], 1, 1).to(device)

            masks.append(expanded_mask)

        masks = torch.stack(masks)
        return masks


    def forward(self, inputs, masks_obj1, masks_obj2):
        features = self.get_representation(inputs)

        mask_pooled_features_obj1 = []
        mask_pooled_features_obj2 = []

        for idx, feature in enumerate(features):
            num_channels, num_x, num_y = feature.shape
            feature = feature.reshape(num_channels, num_x * num_y)

            # The mask was previously resized to the initial size of images so that
            # it can be fed to the augmentation pipeline. Now, we want to apply the
            # masks to the feature output by the network, so we have to resize the
            # masks to their original sizes in order to be able to apply them.
            resized_mask_obj1 = nn.functional.interpolate(masks_obj1[idx][0].unsqueeze(0).unsqueeze(0),
                                                          size=(num_x, num_y), mode="nearest")[0]
            resized_mask_obj2 = nn.functional.interpolate(masks_obj2[idx][0].unsqueeze(0).unsqueeze(0),
                                                          size=(num_x, num_y), mode="nearest")[0]


            resized_mask_obj1 = resized_mask_obj1.repeat(num_channels, 1, 1)
            resized_mask_obj2 = resized_mask_obj2.repeat(num_channels, 1, 1)

            resized_mask_obj1 = resized_mask_obj1.reshape(num_channels, num_x * num_y)
            resized_mask_obj2 = resized_mask_obj2.reshape(num_channels, num_x * num_y)

            mask_pooled_feature_obj1 = torch.mean(torch.mul(feature, resized_mask_obj1), dim=(1))
            mask_pooled_feature_obj2 = torch.mean(torch.mul(feature, resized_mask_obj2), dim=(1))

            mask_pooled_features_obj1.append(mask_pooled_feature_obj1)
            mask_pooled_features_obj2.append(mask_pooled_feature_obj2)

        mask_pooled_features_obj1 = torch.stack(mask_pooled_features_obj1)
        mask_pooled_features_obj2 = torch.stack(mask_pooled_features_obj2)

        return self.projector(mask_pooled_features_obj1), self.projector(mask_pooled_features_obj2)


class Odin(torch.nn.Module):
    def __init__(self, net, moving_average_decay = 0.99):
        super().__init__()

        image_size = 256
        device = 'cuda'

        DEFAULT_AUG = torch.nn.Sequential(
            RandomApply(
                pth_transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p = 0.3
            ),
            pth_transforms.RandomGrayscale(p=0.2),
            pth_transforms.RandomHorizontalFlip(),
            RandomApply(
                pth_transforms.GaussianBlur((3, 3), (1.0, 2.0)),
                p = 0.2
            ),
            pth_transforms.RandomResizedCrop((image_size, image_size)),
            pth_transforms.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
        )

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

        view_masks_one = self.augment1(torch.concat([x, tau_encoder_masks]))
        view_masks_two = self.augment2(torch.concat([x, tau_encoder_masks]))

        view_one = view_masks_one[0: len(x)]
        view_one_masks = view_masks_one[len(x):]

        view_two = view_masks_two[0: len(x)]
        view_two_masks = view_masks_two[len(x):]

        z_theta_obj1_view1, z_theta_obj2_view1 = self.theta_encoder(view_one, view_one_masks, 1 - view_one_masks)
        pred_obj1_view1 = self.theta_predictor(z_theta_obj1_view1)
        pred_obj2_view1 = self.theta_predictor(z_theta_obj2_view1)

        z_theta_obj1_view2, z_theta_obj2_view2 = self.theta_encoder(view_two, view_two_masks, 1 - view_two_masks)
        pred_obj1_view2 = self.theta_predictor(z_theta_obj1_view1)
        pred_obj2_view2 = self.theta_predictor(z_theta_obj2_view1)


        with torch.no_grad():
            z_csi_obj1_view2, z_csi_obj2_view2 = self.csi_encoder(view_two, view_two_masks, 1 - view_two_masks)
            z_csi_obj1_view1, z_csi_obj2_view1 = self.csi_encoder(view_one, view_one_masks, 1 - view_one_masks)

        # Every instance in the batch from pred_obj1 must be similar to z_csi_obj1 (same obj, diff. views, same img).
        # But each one must be different from z_csi_obj2 (diff. object, diff. views, same image).
        # Bacause the theta and csi encoders are different, we have to repeat the same steps for z_csi_obj1
        #  from the second view too.
        # Additionally, the same procedure must be done for pred_obj2 as well.

        loss_obj1 = (loss_fn(pred_obj1_view1, z_csi_obj1_view2, z_csi_obj2_view2) +
                     loss_fn(pred_obj1_view2, z_csi_obj1_view1, z_csi_obj2_view1)) / 2
        loss_obj2 = (loss_fn(pred_obj2_view1, z_csi_obj2_view2, z_csi_obj1_view2) +
                     loss_fn(pred_obj2_view2, z_csi_obj2_view1, z_csi_obj1_view1)) / 2

        return (loss_obj1 + loss_obj2).mean()

def loss_fn(pred_obj1, z_csi_obj1, z_csi_obj2):
    pred_obj1 = F.normalize(pred_obj1, dim=-1, p=2)
    z_csi_obj1 = F.normalize(z_csi_obj1, dim=-1, p=2)
    z_csi_obj2 = F.normalize(z_csi_obj2, dim=-1, p=2)
    similar_instances = (pred_obj1 * z_csi_obj1).sum(dim=-1)
    dissimilar_instances = (pred_obj1 * z_csi_obj2).sum(dim=-1)

    return -torch.log(torch.exp(similar_instances) / torch.exp(similar_instances + dissimilar_instances))

