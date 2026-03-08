import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from models.pointpillars import PointPillars
from utils.pillarization import Pillarization
from utils.anchors_generator import AnchorGenerator


def decode_box_predictions(box_preds, anchors):
    """
    reverses the delta encoding done during training to get absolute coordinates
    box_preds: (N, 7) raw predictions from the model
    anchors:   (N, 7) the anchors each prediction is relative to
    """
    diagonal = np.sqrt(anchors[:, 3] ** 2 + anchors[:, 4] ** 2)

    x   = box_preds[:, 0] * diagonal + anchors[:, 0]
    y   = box_preds[:, 1] * diagonal + anchors[:, 1]
    z   = box_preds[:, 2] * anchors[:, 5] + anchors[:, 2]
    l   = np.exp(box_preds[:, 3]) * anchors[:, 3]
    w   = np.exp(box_preds[:, 4]) * anchors[:, 4]
    h   = np.exp(box_preds[:, 5]) * anchors[:, 5]
    rot = box_preds[:, 6] + anchors[:, 6]

    return np.stack([x, y, z, l, w, h, rot], axis=1)


def run_inference(bin_path, checkpoint_path, score_threshold=0.3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the model and the trained weights
    model = PointPillars(num_classes=3).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    # pillarize the raw point cloud
    pillarizer = Pillarization()
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    pillars, pillar_indices, spatial_shape = pillarizer(points)

    pillars_tensor = torch.tensor(pillars, dtype=torch.float32).unsqueeze(0).to(device)
    indices_tensor = torch.tensor(pillar_indices, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        class_preds, box_preds, dir_preds = model(
            pillars_tensor, indices_tensor, spatial_shape
        )

    # flatten predictions from (1, C, H, W) to (H*W*num_anchors, ...)
    cls_flat = class_preds[0].permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
    box_flat = box_preds[0].permute(1, 2, 0).reshape(-1, 7).cpu().numpy()

    # get the confidence score for each anchor using softmax
    scores = np.exp(cls_flat) / np.exp(cls_flat).sum(axis=1, keepdims=True)
    max_scores = scores.max(axis=1)
    max_classes = scores.argmax(axis=1)
    print(f"score stats - min: {max_scores.min():.3f} max: {max_scores.max():.3f} mean: {max_scores.mean():.3f}")
    # keep only predictions above the confidence threshold
    keep = max_scores > score_threshold
    if keep.sum() == 0:
        print("no detections above threshold")
        return [], points

    # generate anchors and decode the surviving predictions
    anchor_gen = AnchorGenerator(grid_h=220, grid_w=250)
    anchors = anchor_gen.generate()

    filtered_boxes   = decode_box_predictions(box_flat[keep], anchors[keep])
    filtered_scores  = max_scores[keep]
    filtered_classes = max_classes[keep]

    detections = []
    for i in range(len(filtered_boxes)):
        detections.append({
            'box':    filtered_boxes[i],
            'score':  filtered_scores[i],
            'class':  filtered_classes[i]
        })

    return detections, points


def visualize_detections(detections, points, save_path=None):
    pillarizer = Pillarization()

    mask = (
        (points[:, 0] >= pillarizer.x_range[0]) & (points[:, 0] < pillarizer.x_range[1]) &
        (points[:, 1] >= pillarizer.y_range[0]) & (points[:, 1] < pillarizer.y_range[1])
    )
    points = points[mask]

    density_map, _, _ = np.histogram2d(
        points[:, 0], points[:, 1],
        bins=[pillarizer.grid_w, pillarizer.grid_h],
        range=[pillarizer.x_range, pillarizer.y_range]
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(
        np.log1p(density_map.T),
        origin='lower',
        extent=[
            pillarizer.x_range[0], pillarizer.x_range[1],
            pillarizer.y_range[0], pillarizer.y_range[1]
        ],
        cmap='viridis',
        aspect='auto'
    )

    class_names  = ['Car', 'Pedestrian', 'Cyclist']
    class_colors = ['red', 'blue', 'green']

    for det in detections:
        x, y, z, l, w, h, rot = det['box']
        color = class_colors[det['class']]
        score = det['score']

        corner_x = x - l / 2
        corner_y = y - w / 2

        rect = patches.Rectangle(
            (corner_x, corner_y), l, w,
            linewidth=1.5,
            edgecolor=color,
            facecolor='none'
        )
        transform = plt.matplotlib.transforms.Affine2D().rotate_around(x, y, rot)
        rect.set_transform(transform + ax.transData)
        ax.add_patch(rect)

        # show the confidence score next to each box
        ax.text(x, y, f'{score:.2f}', color=color, fontsize=6)

    handles = [
        patches.Patch(color=c, label=n)
        for n, c in zip(class_names, class_colors)
    ]
    ax.legend(handles=handles, loc='upper right')
    ax.set_xlabel('x (meters) - forward')
    ax.set_ylabel('y (meters) - lateral')
    ax.set_title('PointPillars Predictions')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"saved to {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == '__main__':
    detections, points = run_inference(
        bin_path='data/kitti/train/velodyne/000110.bin',
        checkpoint_path='checkpoints/model_epoch_10.pth',
        score_threshold=0.9999
    )

    print(f"found {len(detections)} detections")
    visualize_detections(
        detections, points,
        save_path='assets/predictions_000110.png'
    )