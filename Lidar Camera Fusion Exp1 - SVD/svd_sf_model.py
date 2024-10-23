import torch
import torch.nn as nn
import torch.nn.functional as F

class LiDARFeatureExtractorWithSVD(nn.Module):
    def __init__(self, input_dim=4, svd_dim=2):
        super(LiDARFeatureExtractorWithSVD, self).__init__()
        self.svd_dim = svd_dim
        self.fc1 = nn.Linear(svd_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        
    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(1)
        x = x.view(batch_size * num_points, -1)  # Flatten to [batch_size * num_points, features]
        
        # Apply SVD
        U, S, V = torch.svd(x)
        x_svd = torch.mm(U[:, :self.svd_dim], torch.diag(S[:self.svd_dim]))
        x_svd = x_svd.view(batch_size, num_points, -1)
        
        # Continue with the rest of the network
        x = F.relu(self.fc1(x_svd))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.mean(dim=1)  # Aggregate features
        return x

class CameraFeatureExtractor(nn.Module):
    def __init__(self):
        super(CameraFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256 * 28 * 28, 1024)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x

# Updated SensorFusionModel to use LiDARFeatureExtractorWithSVD
class SensorFusionModelwithSVD(nn.Module):
    def __init__(self, num_objects=10, num_classes=3):
        super(SensorFusionModelwithSVD, self).__init__()
        self.lidar_extractor = LiDARFeatureExtractorWithSVD()
        self.camera_extractor = CameraFeatureExtractor()
        self.fc1 = nn.Linear(1024 + 512, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.num_objects = num_objects
        self.num_classes = num_classes
        self.fc3 = nn.Linear(256, self.num_objects * (4 + self.num_classes))  # Predicts bounding box (4) and class scores (num_classes) for each object
        
    def forward(self, lidar, image):
        lidar_features = self.lidar_extractor(lidar)
        camera_features = self.camera_extractor(image)
        combined = torch.cat((lidar_features, camera_features), dim=1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, self.num_objects, 4 + self.num_classes)  # Reshape to [batch_size, num_objects, 4 + num_classes]
        return x

# model_with_svd = SensorFusionModelwithSVD()

