import torch
import torch.nn as nn
import torch.nn.functional as F

class MeshConv(nn.Module):
    """Computes convolution between faces and 4 incident (1-ring) face neighbors.
    In the forward pass, it takes:
    x: face features (Batch x Features x Faces)
    mesh: list of mesh data structures (len(mesh) == Batch)
    and applies convolution.
    """
    def __init__(self, in_channels, out_channels, k=5, bias=True):
        super(MeshConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, k), bias=bias)
        self.k = k

    def __call__(self, face_f, mesh):
        return self.forward(face_f, mesh)

    def forward(self, x, mesh):
        x = x.squeeze(-1)
        G = torch.cat([self.pad_gemm(i, x.shape[2], x.device) for i in mesh], 0)
        # Build 'neighborhood image' and apply convolution
        G = self.create_GeMM(x, G)
        x = self.conv(G)

        ### new code
        x_1 = x[:, :, :, 1] + x[:, :, :, 3]
        x_2 = x[:, :, :, 2] + x[:, :, :, 4]
        x_3 = torch.abs(x[:, :, :, 1] - x[:, :, :, 3])
        x_4 = torch.abs(x[:, :, :, 2] - x[:, :, :, 4])
        # x = torch.stack([x[:, :, :, 0], x_1, x_2, x_3, x_4], dim=4)
        x = torch.stack([x[:, :, :, 0], x_1, x_2, x_3, x_4], dim=2)
        x = self.conv(x)

        return x

    def flatten_gemm_inds(self, Gi):
        (b, nf, nn) = Gi.shape
        nf += 1
        batch_n = torch.floor(torch.arange(b * nf, device=Gi.device).float() / nf).view(b, nf)
        add_fac = batch_n * nf
        add_fac = add_fac.view(b, nf, 1)
        add_fac = add_fac.repeat(1, 1, nn)
        # Flatten Gi
        Gi = Gi.float() + add_fac[:, 1:, :]
        return Gi

    def create_GeMM(self, x, Gi):
        """Gathers the face features (x) with the 1-ring indices (Gi),
        applies symmetric functions to handle order invariance,
        and returns a 'fake image' which can be used for 2D convolution.
        Output dimensions: Batch x Channels x Faces x 5.
        """
        Gishape = Gi.shape
        # Pad the first row of every sample in the batch with zeros
        padding = torch.zeros((x.shape[0], x.shape[1], 1), requires_grad=True, device=x.device)
        x = torch.cat((padding, x), dim=2)
        Gi = Gi + 1  # Shift

        # First flatten indices
        Gi_flat = self.flatten_gemm_inds(Gi)
        Gi_flat = Gi_flat.view(-1).long()

        odim = x.shape
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(odim[0] * odim[2], odim[1])

        f = torch.index_select(x, dim=0, index=Gi_flat)
        f = f.view(Gishape[0], Gishape[1], Gishape[2], -1)
        f = f.permute(0, 3, 1, 2)

        # Apply the symmetric functions for an equivariant convolution
        x_1 = f[:, :, :, 1] + f[:, :, :, 3]
        x_2 = f[:, :, :, 2] + f[:, :, :, 4]
        x_3 = torch.abs(f[:, :, :, 1] - f[:, :, :, 3])
        x_4 = torch.abs(f[:, :, :, 2] - f[:, :, :, 4])
        f = torch.stack([f[:, :, :, 0], x_1, x_2, x_3, x_4], dim=3)
        return f

    def pad_gemm(self, m, xsz, device):
        """Extracts one-ring neighbors (4x) -> m.gemm_faces
        which is of size #faces x 4
        Add the face_id itself to make #faces x 5
        Then pad to the desired size e.g., xsz x 5
        """
        padded_gemm = torch.tensor(m.gemm_faces, device=device).float()
        padded_gemm = padded_gemm.requires_grad_()
        padded_gemm = torch.cat((torch.arange(m.faces_count, device=device).float().unsqueeze(1), padded_gemm), dim=1)
        # Pad using F
        padded_gemm = F.pad(padded_gemm, (0, 0, 0, xsz - m.faces_count), "constant", 0)
        padded_gemm = padded_gemm.unsqueeze(0)
        return padded_gemm

