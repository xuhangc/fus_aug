import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResNet(nn.Module):
    """
    A ResNet block with two convolutional layers and an optional time embedding.
    """
    def __init__(self, in_channels, out_channels, *, time_emb_dim=None, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        if time_emb_dim is not None:
            self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x, t=None):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        if t is not None:
            t = F.silu(self.time_mlp(t))
            h = h + t[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)

class Up(nn.Module):
    """
    Upsampling block.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class Down(nn.Module):
    """
    Downsampling block.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Attention(nn.Module):
    """
    Attention block.
    """
    def __init__(self, in_channels, n_heads=8, head_dim=32):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        
        self.norm = nn.GroupNorm(32, in_channels)
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, stride=1, padding=0)
        self.to_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        b, c, h, w = x.shape
        
        qkv = self.to_qkv(self.norm(x)).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(b, self.n_heads, c // self.n_heads, h * w), qkv)
        
        q = q * self.scale
        
        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        
        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = out.reshape(b, c, h, w)
        
        return self.to_out(out) + x

class TimeEmbedding(nn.Module):
    """
    Time embedding module.
    """
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        half = self.dim // 2
        self.freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        )

    def forward(self, t):
        self.freqs = self.freqs.to(t.device)
        args = t[:, None].float() * self.freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

class UNet(nn.Module):
    """
    A UNet model for noise prediction in Stable Diffusion.
    """
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        dim=128,
        dim_mults=(1, 2, 4),
        n_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.dim_mults = dim_mults
        
        # Time embedding
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            TimeEmbedding(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1)
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        dims = [dim]
        current_dim = dim
        
        for i, mult in enumerate(dim_mults):
            out_dim = dim * mult
            for _ in range(n_res_blocks):
                self.down_blocks.append(ResNet(current_dim, out_dim, time_emb_dim=time_dim, dropout=dropout))
                current_dim = out_dim
                dims.append(current_dim)
            
            if i != len(dim_mults) - 1:
                self.down_blocks.append(Down(current_dim, current_dim))
                dims.append(current_dim)

        # Middle block
        self.mid_block1 = ResNet(current_dim, current_dim, time_emb_dim=time_dim, dropout=dropout)
        self.mid_attn = Attention(current_dim)
        self.mid_block2 = ResNet(current_dim, current_dim, time_emb_dim=time_dim, dropout=dropout)

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(dim_mults))):
            out_dim = dim * mult
            for _ in range(n_res_blocks + 1):
                self.up_blocks.append(ResNet(dims.pop() + current_dim, out_dim, time_emb_dim=time_dim, dropout=dropout))
                current_dim = out_dim

            if i != 0:
                self.up_blocks.append(Up(current_dim, current_dim))
        
        # Final layers
        self.final_conv = nn.Sequential(
            nn.GroupNorm(32, current_dim),
            nn.SiLU(),
            nn.Conv2d(current_dim, out_channels, 3, padding=1),
        )

    def forward(self, x, time):
        # Time embedding
        t = self.time_mlp(time)

        # Initial convolution
        x = self.init_conv(x)

        # Downsampling
        h = [x]  # Start with the initial feature map as the first skip connection
        for block in self.down_blocks:
            # Apply the block (ResNet or Downsampling)
            x = block(x, t) if isinstance(block, ResNet) else block(x)
            h.append(x)
        
        # Middle block
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # Upsampling
        for block in self.up_blocks:
            if isinstance(block, ResNet):
                # Concatenate with the corresponding feature map from the down path
                x = torch.cat([x, h.pop()], dim=1)
                x = block(x, t)
            else:
                # Upsample
                x = block(x)

        return self.final_conv(x)