import torch
import torch.nn as nn

class DetectionHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=3, num_anchors=2):
        super().__init__()
        
        # setting up the convolutional layer to guess what kind of object we are looking at
        self.class_head = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=1)
        
        # building the layer that figures out the exact dimensions and position of the box
        # the seven outputs are x y z width length height and the rotation angle
        self.box_head = nn.Conv2d(in_channels, num_anchors * 7, kernel_size=1)
        
        # adding a small layer to fix the issue where the model gets confused about front and back
        self.dir_head = nn.Conv2d(in_channels, num_anchors * 2, kernel_size=1)

    def forward(self, x):
        # run the feature map through our class guesser
        class_guesses = self.class_head(x)
        
        # push the map through the box predictor to get the 3d shapes
        box_guesses = self.box_head(x)
        
        # figure out which way the objects are facing to stop them from driving backwards
        dir_guesses = self.dir_head(x)
        
        # bundle all our predictions together and ship them out
        return class_guesses, box_guesses, dir_guesses  