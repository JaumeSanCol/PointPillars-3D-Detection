import numpy as np


class AnchorGenerator:
    def __init__(self, grid_h, grid_w):
        # the grid dimensions come from the pillarization parameters
        self.grid_h = grid_h
        self.grid_w = grid_w

        # anchor definitions for each class in kitti
        # each row is length width height z_center orientation
        self.anchor_configs = {
            'Car': [
                [3.9, 1.6, 1.56, -1.0, 0],
                [3.9, 1.6, 1.56, -1.0, 1.5708]
            ],
            'Pedestrian': [
                [0.6, 0.8, 1.73, -0.6, 0],
                [0.6, 0.8, 1.73, -0.6, 1.5708]
            ],
            'Cyclist': [
                [0.6, 1.76, 1.73, -0.6, 0],
                [0.6, 1.76, 1.73, -0.6, 1.5708]
            ]
        }

    def generate(self):
        # we will stack anchors for all classes into one big array
        all_anchors = []

        for class_name, configs in self.anchor_configs.items():
            for config in configs:
                length, width, height, z_center, orientation = config

                # place one anchor at the center of every cell in the grid
                # linspace gives us the center coordinate of each cell
                x_centers = np.linspace(0, self.grid_w - 1, self.grid_w) + 0.5
                y_centers = np.linspace(0, self.grid_h - 1, self.grid_h) + 0.5

                # build a meshgrid so we get one anchor per cell
                xs, ys = np.meshgrid(x_centers, y_centers)
                xs = xs.flatten()
                ys = ys.flatten()

                # repeat the box dimensions for every cell position
                n = len(xs)
                anchors = np.column_stack([
                    xs,
                    ys,
                    np.full(n, z_center),
                    np.full(n, length),
                    np.full(n, width),
                    np.full(n, height),
                    np.full(n, orientation)
                ])
                all_anchors.append(anchors)

        # stack everything into one array of shape (total_anchors, 7)
        return np.vstack(all_anchors)