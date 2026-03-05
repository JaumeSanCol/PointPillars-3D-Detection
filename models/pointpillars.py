import torch
import torch.nn as nn

from models.pillar_encoder import PillarEncoder
from models.backbone_2d import Backbone2D
from models.detection_head import DetectionHead


class PointPillars(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        # the encoder turns raw pillars into a pseudo image with 64 channels
        self.pillar_encoder = PillarEncoder(
            num_point_features=9,
            num_output_features=64
        )

        # the backbone processes the pseudo image and outputs 256 channels
        self.backbone = Backbone2D(num_input_channels=64)

        # the detection head takes the 256 channel feature map and predicts boxes
        self.detection_head = DetectionHead(
            in_channels=256,
            num_classes=num_classes,
            num_anchors=2
        )

    def forward(self, pillars, pillar_indices, spatial_shape):
        # step 1: convert raw pillars into a 2d pseudo image
        pseudo_image = self.pillar_encoder(pillars, pillar_indices, spatial_shape)

        # step 2: extract multi scale features from the pseudo image
        features = self.backbone(pseudo_image)

        # step 3: predict classes boxes and directions from the feature map
        class_preds, box_preds, dir_preds = self.detection_head(features)

        return class_preds, box_preds, dir_preds