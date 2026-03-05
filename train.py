import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.pointpillars import PointPillars
from utils.kitti_dataset import KittiDataset


def collate_fn(batch):
    # the default collate breaks with variable length labels so we handle it manually
    pillars = torch.tensor(
        [sample['pillars'] for sample in batch], dtype=torch.float32
    )
    pillar_indices = torch.tensor(
        [sample['pillar_indices'] for sample in batch], dtype=torch.long
    )
    spatial_shape = batch[0]['spatial_shape']
    labels = [sample['labels'] for sample in batch]
    return pillars, pillar_indices, spatial_shape, labels


def compute_loss(class_preds, box_preds, dir_preds, labels):
    # placeholder loss until we wire up the full anchor matching logic
    # for now we just make sure the forward pass runs without crashing
    dummy_loss = class_preds.sum() * 0 + box_preds.sum() * 0 + dir_preds.sum() * 0
    return dummy_loss


def train(data_dir, num_epochs=10, batch_size=4, learning_rate=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"training on {device}")

    # set up the dataset and dataloader
    dataset = KittiDataset(data_dir=data_dir, split='train')
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # build the model and move it to the right device
    model = PointPillars(num_classes=3).to(device)

    # adam works well as a starting point for most detection models
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    cls_criterion = nn.CrossEntropyLoss()
    box_criterion = nn.SmoothL1Loss()
    dir_criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (pillars, pillar_indices, spatial_shape, labels) in enumerate(dataloader):
            # move tensors to gpu if available
            pillars = pillars.to(device)
            pillar_indices = pillar_indices.to(device)

            # clear gradients from the previous step
            optimizer.zero_grad()

            # forward pass through the full pipeline
            class_preds, box_preds, dir_preds = model(
                pillars, pillar_indices, spatial_shape
            )

            # calculate the loss against ground truth labels
            loss = compute_loss(class_preds, box_preds, dir_preds, labels)

            # backprop and update weights
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"epoch {epoch + 1}/{num_epochs} - avg loss: {avg_loss:.4f}")


if __name__ == '__main__':
    train(data_dir='data/kitti')