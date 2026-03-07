import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.pillarization import Pillarization
import os


def load_bin(bin_path):
    # read the raw point cloud from a kitti .bin file
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points


def plot_bev_density(points, save_path=None):
    """
    plots the point cloud from above colored by point density per cell
    points: (N, 4) raw lidar points
    """
    pillarizer = Pillarization()

    # filter points to our region of interest
    mask = (
        (points[:, 0] >= pillarizer.x_range[0]) & (points[:, 0] < pillarizer.x_range[1]) &
        (points[:, 1] >= pillarizer.y_range[0]) & (points[:, 1] < pillarizer.y_range[1])
    )
    points = points[mask]

    # build a 2d histogram counting how many points fall in each cell
    density_map, x_edges, y_edges = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=[pillarizer.grid_w, pillarizer.grid_h],
        range=[pillarizer.x_range, pillarizer.y_range]
    )

    fig, ax = plt.subplots(figsize=(12, 10))

    # log scale makes sparse and dense areas both visible
    img = ax.imshow(
        np.log1p(density_map.T),
        origin='lower',
        extent=[
            pillarizer.x_range[0], pillarizer.x_range[1],
            pillarizer.y_range[0], pillarizer.y_range[1]
        ],
        cmap='viridis',
        aspect='auto'
    )

    plt.colorbar(img, ax=ax, label='log(point count + 1)')
    ax.set_xlabel('x (meters) - forward')
    ax.set_ylabel('y (meters) - lateral')
    ax.set_title('LiDAR Point Cloud - Bird Eye View Density Map')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_bev_with_labels(points, labels, save_path=None):
    """
    plots the bev density map with ground truth bounding boxes on top
    points: (N, 4) raw lidar points
    labels: list of [x, y, z, h, w, l, rotation] boxes
    """
    pillarizer = Pillarization()

    mask = (
        (points[:, 0] >= pillarizer.x_range[0]) & (points[:, 0] < pillarizer.x_range[1]) &
        (points[:, 1] >= pillarizer.y_range[0]) & (points[:, 1] < pillarizer.y_range[1])
    )
    points = points[mask]

    density_map, _, _ = np.histogram2d(
        points[:, 0],
        points[:, 1],
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

    # color each class differently
    class_colors = {
        'Car': 'red',
        'Pedestrian': 'blue',
        'Cyclist': 'green'
    }

    for label in labels:
        x, y, z, h, w, l, rot = label[:7]
        class_name = label[7] if len(label) > 7 else 'Car'
        color = class_colors.get(class_name, 'red')

        # draw a rotated rectangle for each bounding box
        corner_x = x - l / 2
        corner_y = y - w / 2

        rect = patches.Rectangle(
            (corner_x, corner_y), l, w,
            linewidth=1.5,
            edgecolor=color,
            facecolor='none',
            label=class_name
        )

        # apply rotation around the box center
        transform = plt.matplotlib.transforms.Affine2D().rotate_around(x, y, rot)
        rect.set_transform(transform + ax.transData)
        ax.add_patch(rect)

    # build a clean legend without duplicate entries
    handles = [
        patches.Patch(color=c, label=n)
        for n, c in class_colors.items()
    ]
    ax.legend(handles=handles, loc='upper right')

    ax.set_xlabel('x (meters) - forward')
    ax.set_ylabel('y (meters) - lateral')
    ax.set_title('LiDAR BEV with Ground Truth Bounding Boxes')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"saved to {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == '__main__':
    # point to a sample from your kitti data
    sample_id = '000000'
    bin_path   = f'data/kitti/train/velodyne/{sample_id}.bin'
    label_path = f'data/kitti/train/label_2/{sample_id}.txt'

    points = load_bin(bin_path)

    # save the density map
    os.makedirs('assets', exist_ok=True)
    plot_bev_density(points, save_path='assets/bev_density.png')

    # load labels and save the annotated version
    from utils.kitti_dataset import KittiDataset
    dataset = KittiDataset(data_dir='data/kitti', split='train')
    labels  = dataset.parse_kitti_labels(label_path)
    plot_bev_with_labels(points, labels, save_path='assets/bev_with_labels.png')