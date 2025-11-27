"""
U-Net Architecture for Semantic Segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double Convolution Block: (Conv -> BN -> ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        
        # Use bilinear upsampling or transposed convolutions
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 
                                         kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        Args:
            x1: Feature map from decoder path
            x2: Feature map from encoder path (skip connection)
        """
        x1 = self.up(x1)
        
        # Handle size mismatch between x1 and x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution for output"""
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net Architecture for Semantic Segmentation
    
    Paper: https://arxiv.org/abs/1505.04597
    """
    
    def __init__(self, in_channels=3, num_classes=2, bilinear=True, 
                 base_channels=64):
        """
        Args:
            in_channels (int): Number of input channels (3 for RGB)
            num_classes (int): Number of segmentation classes
            bilinear (bool): Use bilinear upsampling or transposed conv
            base_channels (int): Number of channels in first layer
        """
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        
        # Encoder path
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        
        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        
        # Decoder path
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        
        # Output layer
        self.outc = OutConv(base_channels, num_classes)
    
    def forward(self, x):
        """
        Args:
            x (Tensor): [B, C, H, W] input image
        
        Returns:
            Tensor: [B, num_classes, H, W] segmentation logits
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits
    
    def get_feature_maps(self, x):
        """
        Extract intermediate feature maps (useful for visualization).
        
        Returns:
            dict: Dictionary of feature maps at different scales
        """
        features = {}
        
        # Encoder
        x1 = self.inc(x)
        features['enc1'] = x1
        
        x2 = self.down1(x1)
        features['enc2'] = x2
        
        x3 = self.down2(x2)
        features['enc3'] = x3
        
        x4 = self.down3(x3)
        features['enc4'] = x4
        
        x5 = self.down4(x4)
        features['bottleneck'] = x5
        
        # Decoder
        x = self.up1(x5, x4)
        features['dec1'] = x
        
        x = self.up2(x, x3)
        features['dec2'] = x
        
        x = self.up3(x, x2)
        features['dec3'] = x
        
        x = self.up4(x, x1)
        features['dec4'] = x
        
        return features


class LightUNet(nn.Module):
    """
    Lightweight U-Net for faster training and inference.
    Uses fewer channels and depth.
    """
    
    def __init__(self, in_channels=3, num_classes=2, bilinear=True):
        super(LightUNet, self).__init__()
        
        # Smaller base channels
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        
        factor = 2 if bilinear else 1
        self.down3 = Down(128, 256 // factor)
        
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32, bilinear)
        
        self.outc = OutConv(32, num_classes)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        logits = self.outc(x)
        return logits


def get_unet(variant='standard', **kwargs):
    """
    Factory function to create U-Net models.
    
    Args:
        variant (str): 'standard', 'light'
        **kwargs: Additional arguments for the model
    
    Returns:
        nn.Module: U-Net model
    """
    if variant == 'standard':
        return UNet(**kwargs)
    elif variant == 'light':
        return LightUNet(**kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}")


# Example usage
if __name__ == "__main__":
    # Test the model
    model = UNet(in_channels=3, num_classes=5)
    x = torch.randn(2, 3, 256, 256)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")