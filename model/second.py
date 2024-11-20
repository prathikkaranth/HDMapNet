import torch
import torch.nn as nn
import MinkowskiEngine as ME

class SparseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dimension):
        super(SparseConvBlock, self).__init__()
        self.block = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride, dimension=dimension
            ),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU()
        )

    def forward(self, x):
        return self.block(x)


class SECONDBackbone(nn.Module):
    def __init__(self, in_channels=4, out_channels=128):
        
        super(SECONDBackbone, self).__init__()
        dimension = 3

        # Input Sparse Convolution
        self.input_conv = SparseConvBlock(in_channels, 16, kernel_size=3, stride=1, dimension=dimension)

        # Sparse Convolution Layers
        self.conv1 = SparseConvBlock(16, 32, kernel_size=3, stride=2, dimension=dimension)
        self.conv2 = SparseConvBlock(32, 64, kernel_size=3, stride=2, dimension=dimension)
        self.conv3 = SparseConvBlock(64, out_channels, kernel_size=3, stride=2, dimension=dimension)

        # Optional: Pooling layer to aggregate features
        self.global_pool = ME.MinkowskiGlobalPooling()

    def forward(self, x):
        # Apply sparse convolution layers
        x = self.input_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Global pooling to aggregate features (if needed)
        global_features = self.global_pool(x)

        return x, global_features

# Example usage
if __name__ == "__main__":
    # return result from voxel.py
    voxel_coords = torch.load("voxel_coords.pt")
    voxel_features = torch.load("voxel_features.pt")
    voxel_paddings = torch.load("voxel_paddings.pt")
    voxel_centers = torch.load("voxel_centers.pt")
    voxel_indices = torch.load("voxel_indices.pt")

    # Convert to Minkowski sparse tensor
    input_tensor = ME.SparseTensor(features=features, coordinates=coords)

    # Initialize backbone
    backbone = SECONDBackbone(in_channels=4, out_channels=128)

    # Forward pass
    voxel_features, global_features = backbone(input_tensor)

    print("Voxel features shape:", voxel_features.F.shape)  # Sparse features
    print("Global features shape:", global_features.F.shape)  # Aggregated features
