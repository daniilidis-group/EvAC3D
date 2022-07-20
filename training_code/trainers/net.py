import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
import numpy as np
import pdb
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid
from utils import gen_event_images, get_contour, draw_contours
from models import ContourNet
from pytorch_lightning.callbacks import Callback
import torchvision.transforms.functional as TF
from datasets import gen_discretized_event_volume, normalize_event_volume
import matplotlib

class Net(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Net")
        parser.add_argument('--width', type=int, default=224)
        parser.add_argument('--height', type=int, default=224)
        parser.add_argument('--lr', type=float, default=2e-4)
        parser.add_argument('--decoder_layers', type=int, default=5)
        parser.add_argument('--hidden_size', type=int, default=256)
        parser.add_argument('--bottleneck_size', type=int, default=128)
        parser.add_argument('--big_encoder', action="store_true")
        return parent_parser

    def __init__(self, num_input_channels=20,
                            width=224, 
                            height=224,
                            single_scale_flow=False, 
                            lr=2e-4,
                            batch_size=16, 
                            bottleneck_size=128,
                            hidden_size=256,
                            decoder_layers=5,
                            use_tp=False,
                            big_encoder=False,
                            delta_t=-1,
                            **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.width = width
        self.height = height
        self.bottleneck_size = bottleneck_size
        self.lr = lr
        self.num_input_channels = num_input_channels
        self.use_tp = use_tp
        self.big_encoder = big_encoder
        self.hidden_size = hidden_size

        # segmentaion/contour network
        self.contour_net = ContourNet(num_input_channels,
                                        output_dim=2,
                                        point_input_dim=4 if self.use_tp else 2,
                                        big_encoder=self.big_encoder,
                                        bottleneck_size=self.bottleneck_size,
                                        hidden_size=self.hidden_size,
                                        decoder_layers=decoder_layers,
                                        batch_size=batch_size)
    def forward(self, x):
        return self.net(x)

    def step(self, batch, batch_idx, mode="train"):
        return self.point_step(batch, batch_idx, mode=mode)

    def point_step(self, batch, batch_idx, mode="train"):

        plot = (mode=="val" or self.global_step % 100 == 0)
        batched_data = batch

        vis_data = {}
        input_data = batched_data['event_volume']

        event_images = gen_event_images(input_data, mode)
        vis_data["event_images"] = event_images
        vis_data["contour_image"] = batched_data["contour_image"]

        input_data = F.interpolate(input_data, [self.width, self.height])
        loss = 0

        samples = batched_data['events'] if self.use_tp \
                else batched_data['events'][:, :, :2]

        output = self.contour_net(input_data, samples=samples)
        # compute classification loss
        bce = nn.BCEWithLogitsLoss()
        class_loss = bce(output[:, :1, :].squeeze(1), 
                        batched_data['labels'].squeeze(-1))

        self.log('{}/class_loss'.format(mode), class_loss)
        loss += class_loss

        # report accuracy
        labels = batched_data['labels'].long().squeeze(-1)
        pred_labels = (output[:, :1, :].squeeze(1) > 0.).long()
        accuracy = (pred_labels == labels).sum() / torch.numel(pred_labels)
        self.log('{}/pred_accuracy'.format(mode), accuracy)

        # neg_accuracy
        neg = labels < .5
        neg_accuracy = (pred_labels[neg] == labels[neg]).sum()\
                                / torch.numel(pred_labels[neg])
        self.log('{}/neg_pred_accuracy'.format(mode), neg_accuracy)
        
        # pos_accuracy
        pos = labels > .5
        pos_accuracy = (pred_labels[pos] == labels[pos]).sum()\
                                / torch.numel(pred_labels[pos])
        self.log('{}/pos_pred_accuracy'.format(mode), pos_accuracy)

        # plot
        if plot:
            vis_data = self.plot_classify(input_data, batched_data, 
                                                vis_data, mode=mode)
            self.vis(vis_data, mode=mode)
        self.log('{}/loss'.format(mode), loss)

        return loss

    def plot_classify(self, input_data, batched_data, vis_data, mode="train"):
        image_size = [250, 250]
        xx, yy = torch.meshgrid(torch.arange(image_size[1]), torch.arange(image_size[0]))
        xx = (xx - image_size[1]//2) / (image_size[1]//2)
        yy = (yy - image_size[0]//2) / (image_size[0]//2)

        xy = torch.stack((xx.flatten(), yy.flatten()), dim=1).cuda()
        with torch.no_grad():
            all_contour_vis = []

            if not self.use_tp:
                for b_idx in range(input_data.shape[0]):
                    batch_size = image_size[0] * image_size[1]
                    vis_image = torch.zeros([image_size[0], image_size[1]]).bool()
                    x_batch_ind = (torch.floor(xy[..., 0] * image_size[1]/2) + \
                                                            image_size[1]/2).long()
                    y_batch_ind = (torch.floor(xy[..., 1] * image_size[0]/2) + \
                                                            image_size[0]/2).long()
                    samples=xy.unsqueeze(0).float()
                    event_volume = input_data[b_idx, ...].unsqueeze(0)
                    pred_labels = self.contour_net(event_volume, samples)[0, 0, :] > 0.

                    vis_image[y_batch_ind, x_batch_ind] = pred_labels.cpu()
                    all_contour_vis.append((255*vis_image).byte().unsqueeze(0))
                all_contour_vis = torch.stack(all_contour_vis)
                vis_data["pred_pixel_image"] = all_contour_vis

            # get the positive points and plot
            image_size = [250, 250]
            num_events = batched_data['events'].shape[1]

            num_pos_samples = batched_data['num_pos_samples'][0]

            pos_events = batched_data['events'][:, :num_pos_samples, :].cpu()
            neg_events = batched_data['events'][:, num_pos_samples:, :].cpu()

            all_gt_images = []
            for b_idx in range(input_data.shape[0]):
                gt_image = torch.zeros([3, image_size[0], image_size[1]])
                x_coord = torch.floor(pos_events[b_idx, :, 0] * image_size[1]/2 +  \
                                                            image_size[1]/2).long()
                y_coord = torch.floor(pos_events[b_idx, :, 1] * image_size[0]/2 +  \
                                                            image_size[0]/2).long()
                gt_image[0, y_coord, x_coord] = 1

                x_coord_neg = torch.floor(neg_events[b_idx, :, 0] * image_size[1]/2 \
                                                    + image_size[1]/2).long()
                y_coord_neg = torch.floor(neg_events[b_idx, :, 1] * image_size[0]/2 \
                                                    + image_size[0]/2).long()
                gt_image[1, y_coord_neg, x_coord_neg] = 1
                all_gt_images.append(gt_image)

            all_gt_images = (torch.stack(all_gt_images, axis=0)*255).byte()
            vis_data["gt_contour_image"] = all_gt_images

        return vis_data

    def validation_step(self, batch, batch_idx):
        val_loss = self.step(batch, batch_idx, mode="val")
        self.log("val_loss", val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        return self.test(batch, batch_idx, mode="test")

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="train")

    def vis(self, vis_data, mode='train'):
        nrow=4
        for k, v in vis_data.items():
            grid = make_grid(v, nrow=nrow)
            self.logger.experiment.add_image("{}/{}_vis".format(mode, k),
                                             grid, global_step=self.global_step)
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return self.optimizer

