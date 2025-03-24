import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
import numpy as np
from typing import List


class ConditionNet(nn.Module):
    def __init__(self, in_ch=3, nf=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, nf, 7, 2, 1)
        self.conv2 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.conv3 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        cond = self.avg_pool(out)
        return cond


class GFM(nn.Module):
    def __init__(self, cond_nf, in_nf, base_nf):
        super().__init__()
        self.mlp_scale = nn.Conv2d(cond_nf, base_nf, 1, 1, 0)
        self.mlp_shift = nn.Conv2d(cond_nf, base_nf, 1, 1, 0)
        self.conv = nn.Conv2d(in_nf, base_nf, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, cond):
        feat = self.conv(x)
        scale = self.mlp_scale(cond)
        shift = self.mlp_shift(cond)
        out = feat * scale + shift + feat
        out = self.relu(out)
        return out


class EnhanceNet(nn.Module):
    def __init__(self, in_ch=1,
                 out_ch=1,
                 base_nf=64,
                 cond_nf=32):
        super().__init__()
        self.condnet = ConditionNet(in_ch, cond_nf)
        self.gfm1 = GFM(cond_nf, in_ch, base_nf)
        self.gfm2 = GFM(cond_nf, base_nf, base_nf)
        self.gfm3 = GFM(cond_nf, base_nf, out_ch)
    def forward(self, x):
        cond = self.condnet(x)
        out = self.gfm1(x, cond)
        out = self.gfm2(out, cond)
        out = self.gfm3(out, cond)
        return out
    

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

        self.final = EnhanceNet()

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
        h = self.final(h)
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


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    _freqs_cis = freqs_cis[: x.shape[1]]
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return _freqs_cis.view(*shape)


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis_xq = reshape_for_broadcast(freqs_cis, xq_)
    freqs_cis_xk = reshape_for_broadcast(freqs_cis, xk_)

    xq_out = torch.view_as_real(xq_ * freqs_cis_xq).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis_xk).flatten(3)
    return xq_out, xk_out


def sample(logits: torch.Tensor, temperature: float, top_p: float):
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1)

    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)

    probs = torch.softmax(logits_flat / temperature, dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > top_p
    probs_sort[mask] = 0.0
    probs_sort /= probs_sort.sum(dim=-1, keepdim=True)

    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    next_token = next_token.reshape(batch_size, seq_len)
    return next_token


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 4
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x_BLD)) * self.w3(x_BLD))


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.q_norm = nn.LayerNorm(dim)
        self.k_norm = nn.LayerNorm(dim)

    def forward(self, x_BLD: torch.Tensor, attn_mask: torch.Tensor, freq_cis: torch.Tensor) -> torch.Tensor:
        B, L, _ = x_BLD.shape
        dtype = x_BLD.dtype

        xq_BLD = self.wq(x_BLD)
        xk_BLD = self.wk(x_BLD)
        xv_BLD = self.wv(x_BLD)

        xq_BLD = self.q_norm(xq_BLD)
        xk_BLD = self.k_norm(xk_BLD)

        xq_BLD, xk_BLD = apply_rotary_emb(xq_BLD, xk_BLD, freq_cis)
        xq_BLD = xq_BLD.to(dtype)
        xk_BLD = xk_BLD.to(dtype)

        xq_BHLK = xq_BLD.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # (bs, num_heads, L, head_dim)
        xk_BHLK = xk_BLD.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        xv_BHLK = xv_BLD.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        out_BHLK = F.scaled_dot_product_attention(xq_BHLK, xk_BHLK, xv_BHLK, attn_mask=attn_mask).transpose(1, 2).reshape(B, L, self.dim)
        return self.wo(out_BHLK)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = Attention(dim, n_heads)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dim * 4)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6, bias=True),
        )

    def forward(self, x_BLD: torch.Tensor, cond_BD: torch.Tensor, attn_mask: torch.Tensor, freq_cis: torch.Tensor) -> torch.Tensor:
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaLN(cond_BD).chunk(6, dim=1)

        attn_input_BLD = modulate(self.attention_norm(x_BLD), beta1, gamma1)
        attn_output_BLD = self.attention(attn_input_BLD, attn_mask, freq_cis) * alpha1.unsqueeze(1)
        x_BLD = x_BLD + attn_output_BLD

        ffn_input_BLD = modulate(self.ffn_norm(x_BLD), beta2, gamma2)
        ffn_output_BLD = self.ffn(ffn_input_BLD) * alpha2.unsqueeze(1)
        x_BLD = x_BLD + ffn_output_BLD

        return x_BLD


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.embedding = nn.Embedding(num_classes + int(dropout_prob > 0), hidden_size)

    def forward(self, labels, train=True):
        if self.dropout_prob > 0 and train:
            drop_mask = torch.rand_like(labels.float()) < self.dropout_prob
            drop_mask = drop_mask.to(labels.device)
            labels = torch.where(drop_mask, self.num_classes, labels)

        return self.embedding(labels)


class FinalLayer(nn.Module):
    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 2, bias=True),
        )
        self.fc = nn.Linear(dim, vocab_size)

    def forward(self, x_BLC: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.adaLN(x_BLC).chunk(2, dim=2)
        x_BLC = self.layer_norm(x_BLC)
        x_BLC = x_BLC * (1 + gamma) + beta
        return self.fc(x_BLC)


class SmoothLayer(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, window_size=8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        self.layer_norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        B, L, C = x.shape
        shortcut = x
        
        # Layer Norm
        x = self.layer_norm(x)
        
        # Handle the case where sequence length is not divisible by window_size
        # We'll process full windows and then handle the remainder separately
        
        # Process full windows
        num_full_windows = L // self.window_size
        if num_full_windows > 0:
            # Process the portion that fits into full windows
            full_len = num_full_windows * self.window_size
            x_full = x[:, :full_len, :]
            
            # Reshape to windows
            x_windows = x_full.view(B, num_full_windows, self.window_size, C)
            x_windows = x_windows.reshape(B * num_full_windows, self.window_size, C)
            
            # Apply MLP
            x_windows = self.mlp(x_windows)
            
            # Reshape back
            x_full = x_windows.view(B, num_full_windows, self.window_size, C)
            x_full = x_full.reshape(B, full_len, C)
            
            # If there's a remainder, process it separately
            if full_len < L:
                x_remainder = x[:, full_len:, :]
                x_remainder = self.mlp(x_remainder)
                x = torch.cat([x_full, x_remainder], dim=1)
            else:
                x = x_full
        else:
            # If sequence is shorter than window_size, just apply MLP directly
            x = self.mlp(x)
        
        return shortcut + x
    

class VAR(nn.Module):
    def __init__(self, vqvae: VQVAE, dim: int, n_heads: int, n_layers: int, patch_sizes: tuple, n_classes: int = 2, cls_dropout: float = 0.1):
        super().__init__()
        self.vqvae = vqvae
        self.dim = dim
        self.max_len = sum(p**2 for p in patch_sizes)
        self.patch_sizes = patch_sizes
        self.final_patch_size = patch_sizes[-1]
        self.latent_dim = vqvae.config.z_channels
        self.idxs_L = torch.cat([torch.full((patch * patch,), i) for i, patch in enumerate(patch_sizes)]).view(1, self.max_len)
        self.attn_mask = torch.where(self.idxs_L.unsqueeze(-1) >= self.idxs_L.unsqueeze(-2), 0.0, -torch.inf)

        self.class_embedding = LabelEmbedder(n_classes, dim, cls_dropout)
        self.stage_idx_embedding = nn.Embedding(len(patch_sizes), dim)

        self.in_proj = nn.Linear(self.latent_dim, dim)
        self.freqs_cis = precompute_freqs_cis(self.dim, self.max_len)

        self.layers = nn.ModuleList([TransformerBlock(dim, n_heads) for _ in range(n_layers)])
        self.vocab_size = vqvae.config.vocab_size
        self.final_layer = FinalLayer(dim, self.vocab_size)
        self.smooth = SmoothLayer(self.vocab_size, mlp_ratio=4.0, drop=0.0)
        self.num_classes = n_classes

    def predict_logits(self, x_BlD, cond_BD: torch.Tensor) -> torch.Tensor:
        attn_mask = self.attn_mask.to(x_BlD.device)[:, : x_BlD.shape[1], : x_BlD.shape[1]]
        for layer in self.layers:
            x_BlD = layer(x_BlD, cond_BD, attn_mask, self.freqs_cis.to(x_BlD.device))
        x_BlD = self.final_layer(x_BlD)
        x_BlD = self.smooth(x_BlD)
        return x_BlD

    def forward(self, x_BlC: torch.Tensor, cond: torch.LongTensor) -> torch.Tensor:
        B, _, _ = x_BlC.shape  # for training, l = L - (patch_size[0]) = L - 1
        sos = cond_BD = self.class_embedding(cond)
        sos = sos.unsqueeze(1).expand(B, 1, self.dim).to(x_BlC.dtype)
        x_BLD = torch.cat([sos, self.in_proj(x_BlC)], dim=1) + self.stage_idx_embedding(self.idxs_L.expand(B, -1).to(x_BlC.device))
        logits_BLC = self.predict_logits(x_BLD, cond_BD)
        return logits_BLC

    @torch.no_grad()
    def generate(self, cond: torch.LongTensor, cfg_scale: float, temperature: float = 0.1, top_p: float = 0.35) -> torch.Tensor:
        bs = cond.shape[0]
        B = bs * 2  # for classifier free guidance
        out_bCHW = torch.zeros(bs, self.latent_dim, self.final_patch_size, self.final_patch_size).to(cond.device)
        sos = cond_bD = self.class_embedding(cond, train=False)
        cond_BD = torch.cat([cond_bD, torch.full_like(cond_bD, fill_value=self.num_classes)], dim=0)
        sos_B1D = sos.unsqueeze(1).repeat(2, 1, 1).to(cond.device)
        stage_embedding = self.stage_idx_embedding(self.idxs_L.expand(B, -1).to(cond.device))

        all_scales = [sos_B1D]
        curr_start = 0
        for idx, patch_size in enumerate(self.patch_sizes):
            curr_end = curr_start + patch_size**2
            stage_ratio = idx / (len(self.patch_sizes) - 1)

            x_BlD = torch.cat(all_scales, dim=1)
            x_BlD = x_BlD + stage_embedding[:, : x_BlD.shape[1]]
            logits_BlV = self.predict_logits(x_BlD, cond_BD)[:, curr_start:curr_end]

            cfg = cfg_scale * stage_ratio
            # original paper uses logits_BlV = (1 + cfg) * logits_BlV[:bs] - cfg * logits_BlV[bs:]
            # cond_out_blV = logits_BlV[:bs]
            # uncond_out_blV = logits_BlV[bs:]
            # logits_blV = uncond_out_blV + cfg * (cond_out_blV - uncond_out_blV)
            # logits_blV = cond_out_blV
            logits_blV = (1 + cfg) * logits_BlV[:bs] - cfg * logits_BlV[bs:]

            # idx_bl = torch.argmax(logits_blV, dim=-1)
            idx_bl = sample(logits_blV, temperature, top_p)
            idx_bhw = idx_bl.view(bs, patch_size, patch_size)

            zq_bChw = self.vqvae.get_nearest_embedding(idx_bhw).permute(0, 3, 1, 2)
            zq_bCHW = F.interpolate(zq_bChw, (self.final_patch_size, self.final_patch_size), mode="bicubic")
            h_bCHW = self.vqvae.quantizer.phis(zq_bCHW, stage_ratio)
            out_bCHW = out_bCHW + h_bCHW

            if idx != len(self.patch_sizes) - 1:
                next_bCHW = F.interpolate(out_bCHW, (self.patch_sizes[idx + 1], self.patch_sizes[idx + 1]), mode="area")
                next_blC = next_bCHW.flatten(-2).transpose(1, 2)
                next_BlD = self.in_proj(next_blC).repeat(2, 1, 1)
                all_scales.append(next_BlD)

            curr_start = curr_end

        return self.vqvae.to_img(out_bCHW)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        patch_sizes = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 32]
        max_len = sum(p**2 for p in patch_sizes)
        config = VQVAEConfig(
            resolution=128,
            in_channels=1,
            dim=128,
            ch_mult=[1, 2, 4],
            num_res_blocks=2,
            z_channels=128,
            out_ch=1,
            vocab_size=8192,
            patch_sizes=patch_sizes,
        )
        model = VQVAE(config).to(device)
        var = VAR(model, 128, 8, 3, patch_sizes).to(device)
        x = torch.randn(1, max_len - 1, var.latent_dim).to(device)
        cond = torch.randint(0, 2, (1,)).to(device)
        out = var(x, cond)
        assert out.shape == (1, max_len, var.vocab_size)
        out = var.generate(cond, 0)
        print(out.shape)
        print("Success")