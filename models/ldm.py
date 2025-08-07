import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dropout):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.activation = Swish()

        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = self.activation(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return self.nin_shortcut(x) + h

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)
        return x + h_

class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x

# ======================================================================================
# 3. First Stage Model: Variational Autoencoder (VAE)
# ======================================================================================

class VAEEncoder(nn.Module):
    def __init__(self, *, in_channels, ch, ch_mult=(1, 2, 4), num_res_blocks, dropout, z_channels):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = self.ch
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = curr_res
            block_out = self.ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)
            curr_res = block_out

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=curr_res, out_channels=curr_res, dropout=dropout)
        self.mid.attn_1 = AttentionBlock(curr_res)
        self.mid.block_2 = ResnetBlock(in_channels=curr_res, out_channels=curr_res, dropout=dropout)

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=curr_res, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(curr_res, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)
        
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        h = self.norm_out(h)
        h = Swish()(h)
        h = self.conv_out(h)
        return h

class VAEDecoder(nn.Module):
    def __init__(self, *, out_channels, ch, ch_mult=(1, 2, 4), num_res_blocks, dropout, z_channels):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = block_in

        self.conv_in = nn.Conv2d(z_channels, curr_res, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=curr_res, out_channels=curr_res, dropout=dropout)
        self.mid.attn_1 = AttentionBlock(curr_res)
        self.mid.block_2 = ResnetBlock(in_channels=curr_res, out_channels=curr_res, dropout=dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in)
            self.up.insert(0, up)

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        
        h = self.norm_out(h)
        h = Swish()(h)
        h = self.conv_out(h)
        return h

class VAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Create separate configs for encoder and decoder to avoid TypeError
        encoder_config = {k: v for k, v in config.items() if k != 'out_channels'}
        decoder_config = {k: v for k, v in config.items() if k != 'in_channels'}
        self.encoder = VAEEncoder(**encoder_config)
        self.decoder = VAEDecoder(**decoder_config)

    def encode(self, x):
        h = self.encoder(x)
        moments = torch.chunk(h, 2, dim=1)
        return moments

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# ======================================================================================
# 4. Second Stage Model: U-Net for Diffusion
# ======================================================================================

class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings

class UNetModel(nn.Module):
    def __init__(self, *, in_channels, model_channels, out_channels, num_res_blocks,
                 attention_resolutions, dropout=0, channel_mult=(1, 2, 4, 8),
                 num_classes=None):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_classes = num_classes

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            TimestepEmbedding(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            Swish(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [nn.Conv2d(in_channels, model_channels, 3, padding=1)]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResnetBlock(in_channels=ch, out_channels=mult * model_channels, dropout=dropout)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch))
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(Downsample(ch))
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = nn.Sequential(
            ResnetBlock(in_channels=ch, dropout=dropout),
            AttentionBlock(ch),
            ResnetBlock(in_channels=ch, dropout=dropout),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResnetBlock(
                        in_channels=ch + ich,
                        out_channels=model_channels * mult,
                        dropout=dropout,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch))
                if level != 0 and i == num_res_blocks:
                    layers.append(Upsample(ch))
                    ds //= 2
                self.output_blocks.append(nn.Sequential(*layers))

        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=ch, eps=1e-6, affine=True),
            Swish(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps, y=None):
        hs = []
        t_emb = self.time_embed(timesteps)
        if y is not None:
            t_emb = t_emb + self.label_emb(y)

        h = x
        for module in self.input_blocks:
            h = module(h)
            hs.append(h)
        
        h = self.middle_block(h)
        
        for module in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in)
            
        return self.out(h)

# ======================================================================================
# 5. Latent Diffusion Model (LDM) Wrapper
# ======================================================================================

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

class LatentDiffusionModel(nn.Module):
    def __init__(self, vae, unet, timesteps=1000, latent_scale_factor=0.18215):
        super().__init__()
        self.vae = vae
        self.unet = unet
        self.timesteps = timesteps
        self.latent_scale_factor = latent_scale_factor

        # Freeze VAE parameters
        for param in self.vae.parameters():
            param.requires_grad = False

        # Create beta schedule
        betas = linear_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    def get_latents(self, x):
        with torch.no_grad():
            mu, logvar = self.vae.encode(x)
            # Reparameterization trick is only for training VAE, here we just use the mean
            latents = mu * self.latent_scale_factor
        return latents

    def q_sample(self, x_start, t, noise=None):
        """Forward process: add noise to latents"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, x, y=None):
        """Training step for the U-Net"""
        b, _, _, _ = x.shape
        t = torch.randint(0, self.timesteps, (b,), device=x.device).long()
        
        latents = self.get_latents(x)
        noise = torch.randn_like(latents)
        
        x_noisy = self.q_sample(x_start=latents, t=t, noise=noise)
        predicted_noise = self.unet(x_noisy, t, y)
        
        loss = F.mse_loss(noise, predicted_noise)
        return loss

    @torch.no_grad()
    def sample(self, num_samples, class_labels, device):
        """Sampling from the model (DDPM reverse process)"""
        shape = (num_samples, self.unet.in_channels, 32, 32) # Latent shape
        img = torch.randn(shape, device=device)
        
        for i in tqdm(reversed(range(0, self.timesteps)), desc="Sampling", total=self.timesteps):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            
            predicted_noise = self.unet(img, t, class_labels)
            
            alpha_t = 1. - self.betas[i]
            alpha_t_cumprod = self.alphas_cumprod[i]
            
            sqrt_recip_alpha_t = torch.sqrt(1.0 / alpha_t)
            
            # Denoise step
            model_mean = sqrt_recip_alpha_t * (img - self.betas[i] * predicted_noise / self.sqrt_one_minus_alphas_cumprod[i])
            
            if i == 0:
                img = model_mean
            else:
                posterior_variance = (1. - self.alphas_cumprod_prev[i]) / (1. - alpha_t_cumprod) * self.betas[i]
                noise = torch.randn_like(img)
                img = model_mean + torch.sqrt(posterior_variance) * noise
        
        # Decode the latents back to image space
        img = img / self.latent_scale_factor
        img = self.vae.decode(img)
        return img.clamp(-1, 1)