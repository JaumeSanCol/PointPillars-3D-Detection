import os
import numpy as np
import torch
from torch.utils.data import Dataset


class KittiDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        """
        data_dir: path where the KITTI stuff lives like /data/kitti/
        split: picking train val or test
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # setting up where to find the LiDAR scans and labels
        self.lidar_dir = os.path.join(data_dir, split, 'velodyne')
        self.label_dir = os.path.join(data_dir, split, 'label_2')
        
        # fetching all the .bin files hanging out in the folder
        self.file_list = [f.split('.')[0] for f in os.listdir(self.lidar_dir) if f.endswith('.bin')]
        self.file_list.sort()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_id = self.file_list[idx]
        
        # pull the LiDAR point cloud data getting x y z and reflectance
        lidar_path = os.path.join(self.lidar_dir, f"{file_id}.bin")
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        
        # grab labels if we are actually training or validating
        labels = []
        if self.split != 'test':
            label_path = os.path.join(self.label_dir, f"{file_id}.txt")
            labels = self.parse_kitti_labels(label_path)
            
        # throw in some data augmentation or shape things up with voxelization
        if self.transform:
            points, labels = self.transform(points, labels)
            
        return {'points': points, 'labels': labels, 'id': file_id}

    def parse_kitti_labels(self, label_path):
        # open up the file and grab all the lines inside
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        # this will hold our valid bounding boxes
        good_boxes = []
        
        for line in lines:
            # chop the line into pieces so we can read the raw data
            chunks = line.strip().split(' ')
            
            # the very first piece tells us what kind of object it is
            stuff_type = chunks[0]
            
            # ignore anything that isn't a car pedestrian or cyclist
            if stuff_type not in ['Car', 'Pedestrian', 'Cyclist']:
                continue
                
            # grab the basic dimensions height width and length
            box_h = float(chunks[8])
            box_w = float(chunks[9])
            box_l = float(chunks[10])
            
            # figure out exactly where the thing is sitting in the 3d world
            pos_x = float(chunks[11])
            pos_y = float(chunks[12])
            pos_z = float(chunks[13])
            
            # grab the rotation angle to know where it is pointing
            rotation = float(chunks[14])
            
            # pack all this info together and throw it into our list
            good_boxes.append([pos_x, pos_y, pos_z, box_h, box_w, box_l, rotation])
            
        return good_boxes