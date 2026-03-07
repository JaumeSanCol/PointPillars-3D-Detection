import torch
from models.pointpillars import PointPillars

def test_forward_pass():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"running on {device}")

    # these are the spatial dimensions we get from Pillarization
    # 440 rows and 500 columns matching our grid_h and grid_w
    spatial_shape = (440, 500)

    # cook up a fake batch with the shapes the model expects
    batch_size   = 2
    num_pillars  = 12000
    num_points   = 100
    num_features = 9

    pillars = torch.randn(
        batch_size, num_pillars, num_points, num_features
    ).to(device)

    pillar_indices = torch.randint(
        0, 440, (batch_size, num_pillars, 2)
    ).to(device)

    # build the model
    model = PointPillars(num_classes=3).to(device)
    model.eval()

    with torch.no_grad():
        class_preds, box_preds, dir_preds = model(
            pillars, pillar_indices, spatial_shape
        )

    print(f"class_preds shape:  {class_preds.shape}")
    print(f"box_preds shape:    {box_preds.shape}")
    print(f"dir_preds shape:    {dir_preds.shape}")
    print("forward pass completed without errors")

if __name__ == '__main__':
    test_forward_pass()