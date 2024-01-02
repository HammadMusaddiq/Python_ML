import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the novel equivariant convolution based on faces
class MeshConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=5, bias=True):
        super(MeshConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, k), bias=bias)

    def forward(self, x):
        """
        Forward pass given a feature tensor x with shape (N, C, F, 5):
        N - batch
        C - # features
        F - # faces in mesh
        5 - vertices in neighborhood (0 is central vertex)
        """
        x_1 = x[:, :, :, 1] + x[:, :, :, 3]
        x_2 = x[:, :, :, 2] + x[:, :, :, 4]
        x_3 = torch.abs(x[:, :, :, 1] - x[:, :, :, 3])
        x_4 = torch.abs(x[:, :, :, 2] - x[:, :, :, 4])
        # x = torch.stack([x[:, :, :, 0], x_1, x_2, x_3, x_4], dim=4)
        x = torch.stack([x[:, :, :, 0], x_1, x_2, x_3, x_4], dim=2)
        x = self.conv(x)
        return x


# Define the novel vertex-based mesh pooling
class MeshPool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MeshPool, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        """
        Forward pass given a feature tensor x with shape (N, C, V):
        N - batch
        C - # features
        V - # vertices in mesh
        """
        x = self.pool(x)
        x = x.transpose(1, 2)  # Reshape to (N, V, C)
        x = self.linear(x)
        x = x.transpose(1, 2)  # Reshape back to (N, C, V)
        return x


# # Define the complete CNN architecture for 3D object recognition
class MeshCNN(nn.Module):
    def __init__(self, num_classes):
        super(MeshCNN, self).__init__()
        self.conv1 = MeshConv(in_channels=3, out_channels=64)
        self.conv2 = MeshConv(in_channels=64, out_channels=128)
        self.pool1 = MeshPool(in_channels=128, out_channels=64)
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.pool1(x))
        x = x.max(dim=2)[0]  # Max-pooling over the faces dimension
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# # Example usage
# model = MeshCNN(num_classes=10)  # Assuming 10 classes in the SHREC16 dataset
# input_data = torch.randn(32, 3, 100, 5)  # Example input data with 32 samples, 100 faces, and 5 vertices per face
# output = model(input_data)
# print(output.shape)  # Print the output shape for verification

# Example usage
model = MeshCNN(num_classes=10)  # Assuming 10 classes in the SHREC16 dataset
input_data = torch.randn(32, 3, 100, 5)  # Example input data with 32 samples, 100 faces, and 5 vertices per face
print(input_data.shape)
output = model(input_data)
print(output.shape)  #