import torch
import torch.nn as nn


class PillarEncoder(nn.Module):
    def __init__(self, num_point_features=9, num_output_features=64):
        super().__init__()

        # this mlp runs on each point independently before we collapse the pillar
        # input is 9 features per point, output is whatever channel size we want
        self.mlp = nn.Sequential(
            nn.Linear(num_point_features, num_output_features),
            nn.BatchNorm1d(num_output_features),
            nn.ReLU()
        )

        # saving this so we know how many channels the pseudo image will have
        self.num_output_features = num_output_features

    def forward(self, pillars, pillar_indices, spatial_shape):
        """
        pillars: (batch, num_pillars, num_points, 9) the augmented point features
        pillar_indices: (batch, num_pillars, 2) the (i,j) position of each pillar in the grid
        spatial_shape: (H, W) the size of the output pseudo image
        """
        batch_size, num_pillars, num_points, num_features = pillars.shape

        # flatten everything except features so BatchNorm1d is happy
        x = pillars.view(-1, num_features)

        # run the mlp on every point independently
        x = self.mlp(x)

        # recover the pillar structure so we can collapse points
        x = x.view(batch_size, num_pillars, num_points, self.num_output_features)

        # collapse all points in each pillar by taking the max per feature
        x, _ = torch.max(x, dim=2)

        # x is now (batch, num_pillars, 64), each pillar has one vector

        # now we scatter each pillar vector into its (i,j) position in the grid
        H, W = spatial_shape
        pseudo_image = torch.zeros(
            batch_size, self.num_output_features, H, W,
            device=pillars.device
        )

        for b in range(batch_size):
            indices = pillar_indices[b]  # (num_pillars, 2)
            pseudo_image[b, :, indices[:, 0], indices[:, 1]] = x[b].t()

        return pseudo_image