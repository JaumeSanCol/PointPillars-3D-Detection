import torch
from torch.utils.data import DataLoader
import numpy as np
import json
import os
from models.pointpillars import PointPillars
from utils.kitti_dataset import KittiDataset
from utils.loss import PointPillarsLoss


def collate_fn(batch):
    pillars = torch.tensor(
        np.array([sample['pillars'] for sample in batch]), dtype=torch.float32
    )
    pillar_indices = torch.tensor(
        np.array([sample['pillar_indices'] for sample in batch]), dtype=torch.long
    )
    spatial_shape = batch[0]['spatial_shape']
    labels = [sample['labels'] for sample in batch]
    return pillars, pillar_indices, spatial_shape, labels

def train(data_dir, num_epochs=10, batch_size=4, learning_rate=1e-3):
    loss_history = {
        'total': [], 'cls': [], 'box': [], 'dir': []
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"training on {device}")

    dataset = KittiDataset(data_dir=data_dir, split='train', max_samples=750)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = PointPillars(num_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # grid dimensions come from the pillarization parameters
    loss_fn = PointPillarsLoss(num_classes=3, grid_h=220, grid_w=250)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (pillars, pillar_indices, spatial_shape, labels) in enumerate(dataloader):
            pillars = pillars.to(device)
            pillar_indices = pillar_indices.to(device)

            optimizer.zero_grad()

            class_preds, box_preds, dir_preds = model(
                pillars, pillar_indices, spatial_shape
            )

            loss, cls_loss, box_loss, dir_loss = loss_fn.compute(
                class_preds, box_preds, dir_preds, labels
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"epoch {epoch + 1} batch {batch_idx} "
                    f"loss {loss.item():.4f} "
                    f"cls {cls_loss.item():.4f} "
                    f"box {box_loss.item():.4f} "
                    f"dir {dir_loss.item():.4f}"
                )
                loss_history['total'].append(loss.item())
                loss_history['cls'].append(cls_loss.item())
                loss_history['box'].append(box_loss.item())
                loss_history['dir'].append(dir_loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"epoch {epoch + 1}/{num_epochs} avg loss {avg_loss:.4f}")

        os.makedirs('assets', exist_ok=True)
        with open('assets/loss_history.json', 'w') as f:
            json.dump(loss_history, f)
        print("loss history saved to assets/loss_history.json")
        
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(
            model.state_dict(),
            f'checkpoints/model_epoch_{epoch + 1}.pth'
        )
        print(f"checkpoint saved for epoch {epoch + 1}")

if __name__ == '__main__':
    train(data_dir='data/kitti')