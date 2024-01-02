import torch
import torch.nn as nn
import torch.nn.functional as F

class MeshConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=5, mesh_norm=True):
        super(MeshConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.mesh_norm = mesh_norm

    def forward(self, x, mesh):
        x = self.conv(x)

        if self.mesh_norm:
            laplacian = mesh.laplacian.clone().detach()
            laplacian.requires_grad = False
            laplacian = laplacian.to(x.device)
            x = torch.matmul(laplacian, x)

        return x