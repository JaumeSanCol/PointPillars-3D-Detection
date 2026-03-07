import torch
import torch.nn as nn
import numpy as np

from utils.anchors_generator import AnchorGenerator
from utils.anchors_matcher import match_anchors


class PointPillarsLoss:
    def __init__(self, num_classes=3, grid_h=220, grid_w=250):
        self.num_classes = num_classes
        self.cls_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.box_loss_fn = nn.SmoothL1Loss(reduction='none')
        self.dir_loss_fn = nn.CrossEntropyLoss(reduction='none')

        # weights to balance the three losses against each other
        self.cls_weight = 1.0
        self.box_weight = 2.0
        self.dir_weight = 0.2

        # generate all anchors once since they dont change between batches
        anchor_gen = AnchorGenerator(grid_h=220, grid_w=250)
        self.anchors = anchor_gen.generate()

    def compute(self, class_preds, box_preds, dir_preds, gt_boxes_batch):
        """
        class_preds:    (batch, num_classes*2, H, W)
        box_preds:      (batch, 14, H, W)
        dir_preds:      (batch, 4, H, W)
        gt_boxes_batch: list of (M, 7) arrays, one per sample in the batch
        """
        batch_size = class_preds.shape[0]
        device = class_preds.device

        total_cls_loss = torch.tensor(0.0, device=device)
        total_box_loss = torch.tensor(0.0, device=device)
        total_dir_loss = torch.tensor(0.0, device=device)

        for b in range(batch_size):
            gt_boxes = np.array(gt_boxes_batch[b], dtype=np.float32)

            # match anchors to ground truth boxes for this sample
            labels, matched_gt = match_anchors(self.anchors, gt_boxes)

            # flatten the predictions from (C, H, W) to (H*W*num_anchors, C)
            # so each row corresponds to one anchor
            cls_flat = class_preds[b].permute(1, 2, 0).reshape(-1, self.num_classes)
            box_flat = box_preds[b].permute(1, 2, 0).reshape(-1, 7)
            dir_flat = dir_preds[b].permute(1, 2, 0).reshape(-1, 2)

            labels_tensor = torch.tensor(labels, device=device)

            # get positive and negative anchor indices
            pos_mask = labels_tensor == 1
            neg_mask = labels_tensor == 0

            # hard negative mining, keep at most 3x more negatives than positives
            num_pos = pos_mask.sum().item()
            num_neg = min(neg_mask.sum().item(), num_pos * 3)

            # pick the hardest negatives by taking the highest class loss
            if num_neg > 0 and num_pos > 0:
                with torch.no_grad():
                    neg_cls_loss = self.cls_loss_fn(
                        cls_flat[neg_mask],
                        torch.zeros(neg_mask.sum(), dtype=torch.long, device=device)
                    )
                    # sort by loss descending and keep the hardest ones
                    _, hard_neg_idx = neg_cls_loss.sort(descending=True)
                    hard_neg_idx = hard_neg_idx[:num_neg]

            # classification loss on positives and hard negatives
            if num_pos > 0:
                pos_cls_targets = torch.tensor(
                    [gt_boxes_batch[b][matched_gt[i]][7] if len(gt_boxes_batch[b][0]) > 7
                     else 0
                     for i, flag in enumerate(pos_mask.cpu().numpy()) if flag],
                    dtype=torch.long, device=device
                )
                cls_loss_pos = self.cls_loss_fn(
                    cls_flat[pos_mask], pos_cls_targets
                ).mean()

                # box regression loss only on positive anchors
                pos_anchors = torch.tensor(
                    self.anchors[pos_mask.cpu().numpy().astype(bool)],
                    dtype=torch.float32, device=device
                )
                pos_gt = torch.tensor(
                    gt_boxes[matched_gt[pos_mask.cpu().numpy().astype(bool)]],
                    dtype=torch.float32, device=device
                )

                # encode the regression targets as deltas from anchor to gt
                box_targets = encode_box_targets(pos_anchors, pos_gt)
                box_loss = self.box_loss_fn(
                    box_flat[pos_mask], box_targets
                ).mean()

                # direction loss on positive anchors
                dir_targets = (pos_gt[:, 6] > 0).long()
                dir_loss = self.dir_loss_fn(
                    dir_flat[pos_mask], dir_targets
                ).mean()

                total_cls_loss += cls_loss_pos
                total_box_loss += box_loss
                total_dir_loss += dir_loss

        total_loss = (
            self.cls_weight * total_cls_loss +
            self.box_weight * total_box_loss +
            self.dir_weight * total_dir_loss
        )

        return total_loss, total_cls_loss, total_box_loss, total_dir_loss


def encode_box_targets(anchors, gt_boxes):
    """
    instead of regressing absolute coordinates we regress deltas
    this makes the learning problem much easier for the network
    anchors:  (N, 7) with x y z l w h rot
    gt_boxes: (N, 7) with the same format
    """
    # diagonal of the anchor base used to normalize position deltas
    diagonal = torch.sqrt(anchors[:, 3] ** 2 + anchors[:, 4] ** 2)

    dx = (gt_boxes[:, 0] - anchors[:, 0]) / diagonal
    dy = (gt_boxes[:, 1] - anchors[:, 1]) / diagonal
    dz = (gt_boxes[:, 2] - anchors[:, 2]) / anchors[:, 5]
    dl = torch.log(gt_boxes[:, 3] / anchors[:, 3])
    dw = torch.log(gt_boxes[:, 4] / anchors[:, 4])
    dh = torch.log(gt_boxes[:, 5] / anchors[:, 5])
    drot = gt_boxes[:, 6] - anchors[:, 6]

    return torch.stack([dx, dy, dz, dl, dw, dh, drot], dim=1)