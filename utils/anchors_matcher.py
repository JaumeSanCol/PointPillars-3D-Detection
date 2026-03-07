import numpy as np


def compute_iou_bev(anchors, gt_boxes):
    """
    anchors:  (N, 7) with x y z length width height orientation
    gt_boxes: (M, 7) with the same format
    returns:  (N, M) iou matrix
    """
    # pull out the corners of each anchor as axis aligned boxes in bev
    # we ignore orientation for iou calculation to keep things simple
    anchors_x1 = anchors[:, 0] - anchors[:, 3] / 2
    anchors_x2 = anchors[:, 0] + anchors[:, 3] / 2
    anchors_y1 = anchors[:, 1] - anchors[:, 4] / 2
    anchors_y2 = anchors[:, 1] + anchors[:, 4] / 2

    gt_x1 = gt_boxes[:, 0] - gt_boxes[:, 3] / 2
    gt_x2 = gt_boxes[:, 0] + gt_boxes[:, 3] / 2
    gt_y1 = gt_boxes[:, 1] - gt_boxes[:, 4] / 2
    gt_y2 = gt_boxes[:, 1] + gt_boxes[:, 4] / 2

    # for each anchor gt pair find the overlapping rectangle
    # we use broadcasting to compute all pairs at once
    inter_x1 = np.maximum(anchors_x1[:, None], gt_x1[None, :])
    inter_x2 = np.minimum(anchors_x2[:, None], gt_x2[None, :])
    inter_y1 = np.maximum(anchors_y1[:, None], gt_y1[None, :])
    inter_y2 = np.minimum(anchors_y2[:, None], gt_y2[None, :])

    # clip to zero so non overlapping boxes dont give negative areas
    inter_w = np.maximum(inter_x2 - inter_x1, 0)
    inter_h = np.maximum(inter_y2 - inter_y1, 0)
    inter_area = inter_w * inter_h

    anchors_area = (anchors_x2 - anchors_x1) * (anchors_y2 - anchors_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)

    # union is the sum of both areas minus the intersection counted twice
    union_area = anchors_area[:, None] + gt_area[None, :] - inter_area

    iou = inter_area / union_area
    return iou


def match_anchors(anchors, gt_boxes, pos_threshold=0.5, neg_threshold=0.35):
    """
    anchors:       (N, 7) all anchors in the grid
    gt_boxes:      (M, 7) ground truth boxes for this sample
    returns:
        labels:    (N,) 1 for positive, 0 for negative, -1 for ignore
        matched_gt:(N,) index of the matched gt box for each positive anchor
    """
    num_anchors = len(anchors)

    # start everything as ignored
    labels = np.full(num_anchors, -1, dtype=np.int32)
    matched_gt = np.full(num_anchors, -1, dtype=np.int32)

    if len(gt_boxes) == 0:
        # no objects in this scene, mark everything as negative
        labels[:] = 0
        return labels, matched_gt

    # compute iou between all anchors and all gt boxes
    iou_matrix = compute_iou_bev(anchors, gt_boxes)

    # for each anchor find the gt box it overlaps most with
    best_gt_iou = iou_matrix.max(axis=1)
    best_gt_idx = iou_matrix.argmax(axis=1)

    # mark negatives first then overwrite with positives
    labels[best_gt_iou < neg_threshold] = 0
    labels[best_gt_iou >= pos_threshold] = 1
    matched_gt[labels == 1] = best_gt_idx[labels == 1]

    # make sure every gt box has at least one positive anchor
    # even if no anchor exceeded the threshold
    best_anchor_per_gt = iou_matrix.argmax(axis=0)
    labels[best_anchor_per_gt] = 1
    matched_gt[best_anchor_per_gt] = np.arange(len(gt_boxes))

    return labels, matched_gt