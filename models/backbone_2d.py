import torch
import torch.nn as nn

class Backbone2D(nn.Module):
    def __init__(self, num_input_channels=64):
        super().__init__()
        
        # setting up the first block to shrink the image and grab basic patterns
        self.block_one = nn.Sequential(
            nn.Conv2d(num_input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # moving to the second block to squeeze it more and find bigger shapes
        self.block_two = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # now we stretch things back up so they match in size
        self.stretch_one = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # stretching the second block up to match the first one
        self.stretch_two = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

    def forward(self, pseudo_image):
        # push the data through the first shrinking block
        out_one = self.block_one(pseudo_image)
        
        # push that result through the second shrinking block
        out_two = self.block_two(out_one)
        
        # stretch both outputs so they are the exact same size
        up_one = self.stretch_one(out_one)
        up_two = self.stretch_two(out_two)
        
        # smash them together into one big feature map
        final_features = torch.cat([up_one, up_two], dim=1)
        
        return final_features