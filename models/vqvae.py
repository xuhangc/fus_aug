import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
import numpy as np
from typing import List
    

class Phi(nn.Module):
    def __init__(self, dim, residual_ratio: float = 0.5):
        super().__init__()
        self.residual_ratio = residual_ratio
        self.conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)

    def forward(self, h_BChw):
        return (1 - self.residual_ratio) * h_BChw + self.residual_ratio * self.conv(h_BChw)


class PhiPartiallyShared(nn.Module):
    def __init__(self, dim, residual_ratio: float = 0.5, num_phi: int = 4):
        super().__init__()
        self.phis = nn.ModuleList([Phi(dim, residual_ratio) for _ in range(num_phi)])
        self.num_phi = num_phi
        if self.num_phi == 4:
            self.ticks = np.linspace(1 / 3 / self.num_phi, 1 - 1 / 3 / self.num_phi, self.num_phi)
        else:
            self.ticks = np.linspace(1 / 2 / self.num_phi, 1 - 1 / 2 / self.num_phi, self.num_phi)

    def forward(self, x: torch.Tensor, idx_ratio: float) -> Phi:
        return self.phis[np.argmin(np.abs(self.ticks - idx_ratio)).item()](x)


class VectorQuantizer(nn.Module):
    def __init__(self, vocab_size: int, dim: int, patch_sizes: List[int], residual_ratio: float = 0.5, num_phi: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.resolutions = patch_sizes
        self.phis = PhiPartiallyShared(dim, residual_ratio, num_phi)
        self.codebook = nn.Embedding(self.vocab_size, dim)
        self.codebook.weight.data.uniform_(-1 / self.vocab_size, 1 / self.vocab_size)

    def forward(self, f_BCHW: torch.Tensor):
        r_R_BChw, idx_R_BL, zqs_post_conv_R_BCHW = self.encode(f_BCHW)
        f_hat_BCHW, scales_BLC, loss = self.decode(f_BCHW, zqs_post_conv_R_BCHW)
        return f_hat_BCHW, r_R_BChw, idx_R_BL, scales_BLC, loss

    def encode(self, f_BCHW: torch.Tensor):
        B, C, H, W = f_BCHW.shape
        r_R_BChw = []
        idx_R_BL = []
        zqs_post_conv_R_BCHW = []
        for resolution_idx, resolution_k in enumerate(self.resolutions):
            r_BChw = F.interpolate(f_BCHW, (resolution_k, resolution_k), mode="area")
            r_flattened_NC = r_BChw.permute(0, 2, 3, 1).reshape(-1, self.dim).contiguous()
            dist = r_flattened_NC.pow(2).sum(1, keepdim=True) + self.codebook.weight.data.pow(2).sum(1) - 2 * r_flattened_NC @ self.codebook.weight.data.T

            idx_Bhw = torch.argmin(dist, dim=1).view(B, resolution_k, resolution_k)
            idx_R_BL.append(idx_Bhw.reshape(B, -1))
            r_R_BChw.append(r_BChw)

            zq_BChw = self.codebook(idx_Bhw).permute(0, 3, 1, 2).contiguous()
            zq_BCHW = F.interpolate(zq_BChw, size=(H, W), mode="bicubic")
            phi_idx = resolution_idx / (len(self.resolutions) - 1)
            zq_BCHW = self.phis(zq_BCHW, phi_idx)
            zqs_post_conv_R_BCHW.append(zq_BCHW)

            f_BCHW = f_BCHW - zq_BCHW

        return r_R_BChw, idx_R_BL, zqs_post_conv_R_BCHW

    def decode(self, f_BCHW: torch.Tensor, zqs_post_conv_R_BCHW: torch.Tensor):
        f_hat_BCHW = torch.zeros_like(f_BCHW)
        loss = 0
        scales = []  # this is for the teacher forcing input so doesnt include the first scale
        for resolution_idx, resolution_k in enumerate(self.resolutions):
            zq_BCHW = zqs_post_conv_R_BCHW[resolution_idx]
            f_hat_BCHW = f_hat_BCHW + zq_BCHW
            if resolution_idx < len(self.resolutions) - 1:
                next_size = self.resolutions[resolution_idx + 1]
                scales.append(F.interpolate(f_hat_BCHW, (next_size, next_size), mode="area").flatten(-2).transpose(1, 2).contiguous())

            commitment_loss = torch.mean((f_hat_BCHW.detach() - f_BCHW) ** 2)
            codebook_loss = torch.mean((f_hat_BCHW - f_BCHW.detach()) ** 2)
            loss += codebook_loss + 0.25 * commitment_loss

        loss /= len(self.resolutions)
        f_hat_BCHW = f_BCHW + (f_hat_BCHW - f_BCHW).detach()
        return f_hat_BCHW, torch.cat(scales, dim=1), loss

    def get_next_autoregressive_input(self, idx: int, f_hat_BCHW: torch.Tensor, h_BChw: torch.Tensor):
        final_patch_size = self.resolutions[-1]
        h_BCHW = F.interpolate(h_BChw, (final_patch_size, final_patch_size), mode="bicubic")
        h_BCHW = self.phis(h_BCHW, idx / (len(self.resolutions) - 1))
        f_hat_BCHW = f_hat_BCHW + h_BCHW
        return f_hat_BCHW
    


class VQVAEConfig:
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        dim: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
        out_ch: int,
        vocab_size: int,
        patch_sizes: list[int]
    ):
        self.resolution = resolution
        self.in_channels = in_channels
        self.dim = dim
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.z_channels = z_channels
        self.out_ch = out_ch
        self.vocab_size = vocab_size
        self.patch_sizes = patch_sizes


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x_BCHW: Tensor) -> Tensor:
        x_BCHW = self.norm(x_BCHW)
        q_BCHW = self.q(x_BCHW)
        k_BCHW = self.k(x_BCHW)
        v_BCHW = self.v(x_BCHW)

        B, C, H, W = x_BCHW.shape
        q_B1HWC = rearrange(q_BCHW, "b c h w -> b 1 (h w) c").contiguous()
        k_B1HWC = rearrange(k_BCHW, "b c h w -> b 1 (h w) c").contiguous()
        v_B1HWC = rearrange(v_BCHW, "b c h w -> b 1 (h w) c").contiguous()
        h_B1HWC = F.scaled_dot_product_attention(q_B1HWC, k_B1HWC, v_B1HWC)
        h_BCHW = rearrange(h_B1HWC, "b 1 (h w) c -> b c h w", h=H, w=W, c=C, b=B).contiguous()
        return x_BCHW + self.proj_out(h_BCHW)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class VQVAE(nn.Module):
    def __init__(self, config: VQVAEConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(resolution=config.resolution, in_channels=config.in_channels, ch=config.dim, ch_mult=config.ch_mult, num_res_blocks=config.num_res_blocks, z_channels=config.z_channels)
        self.decoder = Decoder(
            ch=config.dim,
            out_ch=config.out_ch,
            ch_mult=config.ch_mult,
            num_res_blocks=config.num_res_blocks,
            in_channels=config.in_channels,
            resolution=config.resolution,
            z_channels=config.z_channels,
        )
        self.quantizer = VectorQuantizer(vocab_size=config.vocab_size, dim=config.z_channels, patch_sizes=config.patch_sizes)

    def forward(self, x):
        f = self.encoder(x)
        fhat, r_maps, idxs, scales, loss = self.quantizer(f)
        x_hat = self.decoder(fhat)
        return x_hat, r_maps, idxs, scales, loss

    def get_nearest_embedding(self, idxs):
        return self.quantizer.codebook(idxs)

    def get_next_autoregressive_input(self, idx, f_hat_BCHW, h_BChw):
        return self.quantizer.get_next_autoregressive_input(idx, f_hat_BCHW, h_BChw)

    def to_img(self, f_hat_BCHW):
        return self.decoder(f_hat_BCHW).clamp(-1, 1)

    def img_to_indices(self, x):
        f = self.encoder(x)
        fhat, r_maps, idxs, scales, loss = self.quantizer(f)
        return idxs