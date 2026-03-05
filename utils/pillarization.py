import numpy as np


class Pillarization:
    def __init__(
        self,
        x_range=(0, 70.4), # 70 meters in front of the car, 0 behind we add0.4 to make it fit nicely into 0.16m cells
        y_range=(-40, 40),
        z_range=(-3, 1),
        cell_size=0.16,
        max_points_per_pillar=100,
        max_pillars=12000
    ):
        # the spatial boundaries of the scene we care about
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

        # how big each cell is in meters
        self.cell_size = cell_size

        # cap on how many points we keep per pillar
        self.max_points_per_pillar = max_points_per_pillar

        # cap on total number of non empty pillars per sample
        self.max_pillars = max_pillars

        # figure out how many cells fit in each direction
        self.grid_w = int((x_range[1] - x_range[0]) / cell_size)  # columns
        self.grid_h = int((y_range[1] - y_range[0]) / cell_size)  # rows
    
    def __call__(self, points):
        # points comes in as (N, 4) with x y z reflectance

        # throw away anything outside our region of interest
        mask = (
            (points[:, 0] >= self.x_range[0]) & (points[:, 0] < self.x_range[1]) &
            (points[:, 1] >= self.y_range[0]) & (points[:, 1] < self.y_range[1]) &
            (points[:, 2] >= self.z_range[0]) & (points[:, 2] < self.z_range[1])
        )
        points = points[mask]

        # figure out which cell each point belongs to
        grid_x = ((points[:, 0] - self.x_range[0]) / self.cell_size).astype(int)
        grid_y = ((points[:, 1] - self.y_range[0]) / self.cell_size).astype(int)
        
        # group points by their (grid_x, grid_y) cell
        pillar_dict = {}
        for i, (gx, gy) in enumerate(zip(grid_x, grid_y)):
            key = (gx, gy)
            if key not in pillar_dict:
                pillar_dict[key] = []
            pillar_dict[key].append(i)

        # build the output arrays
        num_pillars = min(len(pillar_dict), self.max_pillars)
        pillars = np.zeros(
            (num_pillars, self.max_points_per_pillar, 9), dtype=np.float32
        )
        pillar_indices = np.zeros((num_pillars, 2), dtype=np.int32)

        for pillar_idx, ((gx, gy), point_indices) in enumerate(pillar_dict.items()):
            if pillar_idx >= self.max_pillars:
                break

            # grab the points belonging to this pillar
            pillar_points = points[point_indices]

            # cap the number of points if the pillar is too dense
            if len(pillar_points) > self.max_points_per_pillar:
                sampled = np.random.choice(
                    len(pillar_points), self.max_points_per_pillar, replace=False
                )
                pillar_points = pillar_points[sampled]

            n = len(pillar_points)

            # calculate the centroid of this pillar
            centroid = pillar_points[:, :3].mean(axis=0)

            # calculate the geometric center of the cell in world coordinates
            center_x = self.x_range[0] + (gx + 0.5) * self.cell_size
            center_y = self.y_range[0] + (gy + 0.5) * self.cell_size

            # augment each point with distances to centroid and cell center
            x_c = pillar_points[:, 0] - centroid[0]
            y_c = pillar_points[:, 1] - centroid[1]
            z_c = pillar_points[:, 2] - centroid[2]
            x_p = pillar_points[:, 0] - center_x
            y_p = pillar_points[:, 1] - center_y

            # stack original features with the augmented ones
            augmented = np.column_stack([
                pillar_points,         # x y z reflectance
                x_c, y_c, z_c,        # distance to centroid
                x_p, y_p              # distance to cell center
            ])

            # place the augmented points into the output array
            pillars[pillar_idx, :n, :] = augmented
            pillar_indices[pillar_idx] = [gy, gx]

        return pillars, pillar_indices, (self.grid_h, self.grid_w)