import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import config

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for timestep encoding
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DownBlock(nn.Module):
    """
    A basic convolutional block with residual connection
    """
    def __init__(self, in_ch, out_ch, time_emb_dim=None, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch) if time_emb_dim else None
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
            
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t=None):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        
        # Time embedding
        if self.time_mlp and t is not None:
            time_emb = self.relu(self.time_mlp(t))
            h = h + time_emb.unsqueeze(-1).unsqueeze(-1)
            
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        
        # Down or Upsample
        return self.transform(h)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch) if time_emb_dim else None
        
        self.transform = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        self.conv = nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1)
        
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t=None, residual=None):
        # Apply the ConvTranspose2d layer for upsampling
        x = self.relu(self.transform(x))
        
        # Concatenate residual if it exists
        if residual is not None:
            #TODO: dim doesn't match
            if x.shape[2] != residual.shape[2] or x.shape[3] != residual.shape[3]:
                x = F.interpolate(x, size=(residual.shape[2], residual.shape[3]), mode='bilinear', align_corners=False)
            x = torch.cat((x, residual), dim=1)
        
        # Apply convolution and batch normalization
        x = self.bnorm1(self.relu(self.conv(x)))
        
        # Time embedding if provided
        if self.time_mlp and t is not None:
            time_emb = self.relu(self.time_mlp(t))
            x = x + time_emb.unsqueeze(-1).unsqueeze(-1)

        x = self.bnorm2(x)
        return x


class SimpleUnet(nn.Module):
    """
    A simplified U-Net architecture for diffusion models
    """
    def __init__(self):
        super().__init__()
        image_channels = config['channels']
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        time_emb_dim = 32
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)
        
        # Downsample
        self.downs = nn.ModuleList([
            DownBlock(down_channels[i], down_channels[i+1], time_emb_dim)
            for i in range(len(down_channels)-1)
        ])
        
        self.mid = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1)
        )
        
        # Upsample
        self.ups = nn.ModuleList([
            UpBlock(up_channels[i], up_channels[i+1], time_emb_dim) 
            for i in range(len(up_channels)-1)
        ])
        
        # Final output
        self.output = nn.Conv2d(up_channels[-1], image_channels, 1)
        
    def forward(self, x, timestep):
        # Embed time
        t = self.time_mlp(timestep)
        
        # Initial conv
        x = self.conv0(x) # x -> [batch_size, 64, 28, 28]
        
        # Downsample
        residuals = [x]
        for down in self.downs:  # x -> [batch_size, down_channels[-1], 1, 1]
            x = down(x, t)
            residuals.append(x)
        residuals = residuals[:-1]

        x = self.mid(x)

        # Upsample
        for up, residual in zip(self.ups, reversed(residuals)):
            x = up(x, t, residual)
            
        return self.output(x)
