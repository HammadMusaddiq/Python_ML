import torch
from torch.utils.data import Dataset
import numpy as np


class KittiDataset(Dataset):
    def __init__(self, lidar_data, camera_images, annotations, transform=None, lidar_length=50000, max_objects=10):
        self.lidar_data = lidar_data
        self.camera_images = camera_images
        self.annotations = annotations
        self.transform = transform
        self.lidar_length = lidar_length
        self.max_objects = max_objects

    def __len__(self):
        return len(self.camera_images)

    def __getitem__(self, idx):
        lidar = self.pad_lidar_data(self.lidar_data[idx], self.lidar_length)
        image = self.camera_images[idx]
        annot = self.annotations[idx]

        orig_size = image.shape[:2][::-1]  # (width, height)
        new_size = (224, 224)
        
        if self.transform:
            image = self.transform(image)

        resized_bboxes = self.resize_bboxes([obj['bbox'] for obj in annot], orig_size, new_size)
        for i, obj in enumerate(annot):
            obj['bbox'] = resized_bboxes[i]
        
        bbox_targets, class_targets = self.prepare_targets(annot, self.max_objects)

        # Adjust the shape for LiDAR data
        lidar = torch.tensor(lidar, dtype=torch.float32).unsqueeze(0)  # [1, 50000, 4]
        
        return lidar, torch.tensor(image, dtype=torch.float32), (bbox_targets, class_targets)

    
    def pad_lidar_data(self, lidar_data, target_length=2000):
        if len(lidar_data) < target_length:
            padding = np.zeros((target_length - len(lidar_data), lidar_data.shape[1]))
            return np.vstack((lidar_data, padding))
        else:
            return lidar_data[:target_length, :]
    
    def prepare_targets(self, annotations, max_objects):
        bboxes = []
        classes = []
        for obj in annotations:
            bboxes.append(obj['bbox'])
            if obj['type'] == 'Car':
                classes.append(1)
            elif obj['type'] == 'Truck':
                classes.append(2)
            else:
                classes.append(0)
        
        # Pad targets to have a consistent size
        while len(bboxes) < max_objects:
            bboxes.append([0, 0, 0, 0])
            classes.append(0)
        
        bboxes = bboxes[:max_objects]  # Truncate if necessary
        classes = classes[:max_objects]  # Truncate if necessary
        
        return torch.tensor(bboxes, dtype=torch.float32), torch.tensor(classes, dtype=torch.long)
    
    def resize_bboxes(self, bboxes, orig_size, new_size):
        orig_w, orig_h = orig_size
        new_w, new_h = new_size
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
        resized_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            x1 = x1 * scale_x
            y1 = y1 * scale_y
            x2 = x2 * scale_x
            y2 = y2 * scale_y
            resized_bboxes.append([x1, y1, x2, y2])
        return resized_bboxes