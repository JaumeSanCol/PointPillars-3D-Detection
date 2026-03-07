import numpy as np
class AnchorGenerator:
    def __init__(self, grid_h, grid_w):
        self.grid_h = grid_h
        self.grid_w = grid_w

        # just two orientations per cell, class agnostic
        self.anchor_configs = [
            [3.9, 1.6, 1.56, -1.0, 0],
            [3.9, 1.6, 1.56, -1.0, 1.5708]
        ]

    def generate(self):
        all_anchors = []

        for config in self.anchor_configs:
            length, width, height, z_center, orientation = config

            x_centers = np.linspace(0, self.grid_w - 1, self.grid_w) + 0.5
            y_centers = np.linspace(0, self.grid_h - 1, self.grid_h) + 0.5

            xs, ys = np.meshgrid(x_centers, y_centers)
            xs = xs.flatten()
            ys = ys.flatten()

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

        return np.vstack(all_anchors)