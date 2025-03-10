import math
import torch
import torch.nn as nn
from einops import repeat


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(
            half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=128, patch_size=8, in_chans=1, embed_dim=1024):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == W == self.img_size, f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})"

        # BCHW -> BNC
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    """MLP with GeLU activation"""

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm"""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim,
                       hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, cond=None):
        # Apply conditioning via adaptive layer norm if provided
        if cond is not None:
            x = x + self.attn(self.norm1(x + cond))
            x = x + self.mlp(self.norm2(x + cond))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class DiT(nn.Module):
    """
    Diffusion Transformer (DiT) for ultrasound image generation
    """

    def __init__(
        self,
        img_size=128,
        patch_size=8,
        in_channels=1,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        num_classes=2,
    ):
        super().__init__()

        # Image Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches

        # Positional Embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim))

        # Time Embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        # Class Embedding for binary labels
        self.class_embed = nn.Embedding(num_classes, embed_dim)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            )
            for _ in range(depth)
        ])

        # Final Layer Norm
        self.norm = nn.LayerNorm(embed_dim)

        # Output projection to reconstruct image
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, patch_size * patch_size * in_channels)
        )

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def unpatchify(self, x):
        """
        x: (B, L, patch_size**2 * in_channels)
        """
        p = self.patch_embed.patch_size
        h = w = int(x.shape[1] ** 0.5)
        in_chans = self.proj[0].out_features // (p ** 2)

        # (B, L, patch_size**2 * in_channels) -> (B, h, w, patch_size, patch_size, in_channels)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, in_chans))

        # (B, h, w, patch_size, patch_size, in_channels) -> (B, in_channels, h*patch_size, w*patch_size)
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(shape=(x.shape[0], in_chans, h * p, w * p))

        return x

    def forward(self, x, timesteps, class_labels=None):
        # Get batch dimensions
        B = x.shape[0]

        # Tokenize image
        x = self.patch_embed(x)

        # Add positional encoding
        x = x + self.pos_embed

        # Time conditioning
        time_embed = self.time_embed(timesteps)
        time_embed = repeat(time_embed, 'b d -> b n d', n=x.shape[1])

        # Class conditioning
        if class_labels is not None:
            class_labels = class_labels.long().squeeze()
            class_embed = self.class_embed(class_labels)
            class_embed = repeat(class_embed, 'b d -> b n d', n=x.shape[1])
            cond = time_embed + class_embed
        else:
            cond = time_embed

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, cond)

        # Final normalization and output projection
        x = self.norm(x)
        x = self.proj(x)

        # Unpatchify to reconstruct image
        output = self.unpatchify(x)

        return output
