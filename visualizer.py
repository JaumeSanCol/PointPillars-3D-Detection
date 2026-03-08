import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.pillarization import Pillarization
from utils.kitti_dataset import KittiDataset
import os

import json
import torch
from models.pillar_encoder import PillarEncoder


def plot_height_colored(points, save_path=None):
    """
    plots the point cloud from above colored by z height
    gives a much better sense of the 3d structure than density alone
    """
    pillarizer = Pillarization()

    mask = (
        (points[:, 0] >= pillarizer.x_range[0]) & (points[:, 0] < pillarizer.x_range[1]) &
        (points[:, 1] >= pillarizer.y_range[0]) & (points[:, 1] < pillarizer.y_range[1]) &
        (points[:, 2] >= pillarizer.z_range[0]) & (points[:, 2] < pillarizer.z_range[1])
    )
    points = points[mask]

    fig, ax = plt.subplots(figsize=(12, 10))

    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=points[:, 2],
        cmap='plasma',
        s=0.3,
        vmin=pillarizer.z_range[0],
        vmax=pillarizer.z_range[1]
    )

    plt.colorbar(scatter, ax=ax, label='height (meters)')
    ax.set_xlabel('x (meters) - forward')
    ax.set_ylabel('y (meters) - lateral')
    ax.set_title('LiDAR Point Cloud - Colored by Height')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.tick_params(colors='white')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')
        print(f"saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_encoder_output(points, save_path=None):
    """
    runs the pillar encoder on a real point cloud and visualizes
    the resulting pseudo image as a heatmap, one plot per channel group
    """
    pillarizer = Pillarization()
    pillars, pillar_indices, spatial_shape = pillarizer(points)

    pillars_tensor = torch.tensor(pillars, dtype=torch.float32).unsqueeze(0)
    indices_tensor = torch.tensor(pillar_indices, dtype=torch.long).unsqueeze(0)

    encoder = PillarEncoder(num_point_features=9, num_output_features=64)
    encoder.eval()

    with torch.no_grad():
        pseudo_image = encoder(pillars_tensor, indices_tensor, spatial_shape)

    # pseudo image shape is (1, 64, H, W), drop the batch dim
    pseudo_image = pseudo_image.squeeze(0).numpy()

    # show the mean across all 64 channels as a single heatmap
    mean_activation = pseudo_image.mean(axis=0)

    fig, ax = plt.subplots(figsize=(12, 10))

    img = ax.imshow(
        mean_activation,
        origin='lower',
        cmap='inferno',
        aspect='auto'
    )

    plt.colorbar(img, ax=ax, label='mean activation')
    ax.set_title('Pillar Encoder Output - Mean Activation across 64 Channels')
    ax.set_xlabel('grid x')
    ax.set_ylabel('grid y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_loss_history(loss_history_path='assets/loss_history.json', save_path=None):
    """
    reads the loss history dumped by train.py and plots all four curves
    """
    with open(loss_history_path, 'r') as f:
        history = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Loss History', fontsize=14)

    keys   = ['total', 'cls', 'box', 'dir']
    titles = ['Total Loss', 'Classification Loss', 'Box Regression Loss', 'Direction Loss']
    colors = ['white', 'cyan', 'orange', 'lime']
    fig.patch.set_facecolor('#1a1a2e')

    for ax, key, title, color in zip(axes.flatten(), keys, titles, colors):
        ax.plot(history[key], color=color, linewidth=0.8, alpha=0.9)
        ax.set_title(title, color='white')
        ax.set_xlabel('batch', color='white')
        ax.set_ylabel('loss', color='white')
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        print(f"saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_pipeline_diagram(save_path=None):
    """
    draws a simple architecture diagram showing the data flow
    through the full pointpillars pipeline
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')

    blocks = [
        (1.0,  'Raw\nPoint Cloud\n(N, 4)',        '#e94560'),
        (3.2,  'Pillarization\n(500x440 grid)',   '#0f3460'),
        (5.4,  'Pillar Encoder\nPseudo Image\n(64, H, W)', '#0f3460'),
        (7.6,  'Backbone 2D\nMulti-scale\n(256, H/2, W/2)', '#0f3460'),
        (9.8,  'Detection Head\nClass + Box\n+ Direction',  '#0f3460'),
        (12.0, 'Predictions\n3D Boxes',            '#e94560'),
    ]

    for x, label, color in blocks:
        rect = plt.Rectangle(
            (x - 0.9, 2.0), 1.8, 2.0,
            facecolor=color, edgecolor='white', linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(
            x, 3.0, label,
            ha='center', va='center',
            color='white', fontsize=8, fontweight='bold'
        )

    # draw arrows between blocks
    for i in range(len(blocks) - 1):
        x_start = blocks[i][0] + 0.9
        x_end   = blocks[i + 1][0] - 0.9
        ax.annotate(
            '', xy=(x_end, 3.0), xytext=(x_start, 3.0),
            arrowprops=dict(arrowstyle='->', color='white', lw=1.5)
        )

    ax.set_title(
        'PointPillars Pipeline', color='white', fontsize=14, pad=20
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        print(f"saved to {save_path}")
    else:
        plt.show()

    plt.close()

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
def cam_to_lidar(label):
    # kitti labels are in camera coords, we need lidar coords for bev
    x_cam, y_cam, z_cam = label[0], label[1], label[2]
    h, w, l, rot = label[3], label[4], label[5], label[6]

    x_lidar =  z_cam
    y_lidar = -x_cam
    z_lidar = -y_cam + 1.65

    # rotation also needs to be adjusted between coordinate systems
    rot_lidar = -rot - np.pi / 2

    return [x_lidar, y_lidar, z_lidar, h, w, l, rot_lidar]

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
        label = cam_to_lidar(label)
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
    os.makedirs('assets', exist_ok=True)

    sample_ids = ['000010', '000110', '000050']
    dataset = KittiDataset(data_dir='data/kitti', split='train')

    for sample_id in sample_ids:
        bin_path   = f'data/kitti/train/velodyne/{sample_id}.bin'
        label_path = f'data/kitti/train/label_2/{sample_id}.txt'
        points = load_bin(bin_path)
        labels = dataset.parse_kitti_labels(label_path)

        plot_bev_density(points,     save_path=f'assets/bev_density_{sample_id}.png')
        plot_bev_with_labels(points, labels, save_path=f'assets/bev_labels_{sample_id}.png')
        plot_height_colored(points,  save_path=f'assets/bev_height_{sample_id}.png')
        plot_encoder_output(points,  save_path=f'assets/encoder_output_{sample_id}.png')

    plot_loss_history(save_path='assets/loss_history.png')
    plot_pipeline_diagram(save_path='assets/pipeline_diagram.png')

    print("all assets generated")