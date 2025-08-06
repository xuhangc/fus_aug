import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
import numpy as np
from tqdm import tqdm


# Helper modules for UNet architecture
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if time_emb_dim is not None else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = time_emb.view(time_emb.shape[0], -1, 1, 1)
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(b, self.heads, -1, h * w), qkv)

        q = q * self.scale
        sim = torch.einsum("bhdn,bhdm->bhnm", q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v)
        out = out.reshape(b, -1, h, w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            nn.GroupNorm(1, dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(b, self.heads, -1, h * w), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("bhdn,bhdm->bhnm", k, v)
        out = torch.einsum("bhdn,bhnm->bhdm", q, context)
        out = out.view(b, -1, h, w)
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        context_dim = context_dim or query_dim
        inner_dim = dim_head * heads
        
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        h = self.heads
        context = context if context is not None else x
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = map(lambda t: t.reshape(*t.shape[:-1], h, -1).transpose(-3, -2), (q, k, v))
        
        sim = torch.einsum('bhnd,bhmd->bhnm', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('bhnm,bhmd->bhnd', attn, v)
        
        out = out.transpose(-3, -2).reshape(*out.shape[:-3], -1)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class Downsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim
        self.conv = nn.Conv2d(dim, dim_out, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim
        self.conv = nn.Conv2d(dim, dim_out, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class ContextEmbedding(nn.Module):
    def __init__(self, num_classes, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        
    def forward(self, context):
        # context is expected to be of shape [batch_size] with integer class labels
        return self.embedding(context)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        model_channels=128,
        out_channels=1,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        channel_mult=(1, 2, 4, 8),
        num_heads=8,
        n_classes=None  # Changed from context_dim to n_classes
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        
        # Time embedding
        time_dim = model_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Context embedding for conditional generation
        self.has_context = n_classes is not None
        if self.has_context:
            self.context_embedding = ContextEmbedding(n_classes, time_dim)
            
        # Initial projection
        self.init_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Downsampling
        self.downs = nn.ModuleList([])
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResnetBlock(ch, mult * model_channels, time_emb_dim=time_dim),
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(PreNorm(ch, Attention(ch, heads=num_heads)))
                self.downs.append(nn.ModuleList(layers))
                input_block_chans.append(ch)
            
            if level != len(channel_mult) - 1:
                self.downs.append(nn.ModuleList([Downsample(ch)]))
                input_block_chans.append(ch)
                ds *= 2
        
        # Middle blocks
        self.middle_block = nn.ModuleList([
            ResnetBlock(ch, ch, time_emb_dim=time_dim),
            PreNorm(ch, Attention(ch, heads=num_heads)),
            ResnetBlock(ch, ch, time_emb_dim=time_dim)
        ])
        
        # Upsampling
        self.ups = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResnetBlock(
                        ch + input_block_chans.pop(),
                        model_channels * mult,
                        time_emb_dim=time_dim
                    ),
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(PreNorm(ch, Attention(ch, heads=num_heads)))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch))
                    ds //= 2
                self.ups.append(nn.ModuleList(layers))
        
        # Final layers
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )
    
    def forward(self, x, time, context=None):
        """
        x: [B, C, H, W] image tensor
        time: [B] timestep tensor
        context: [B] class label tensor (optional)
        """
        # Time embedding
        t_emb = self.time_mlp(time)
        
        # Context embedding for conditional generation
        if self.has_context and context is not None:
            c_emb = self.context_embedding(context)
            t_emb = t_emb + c_emb
        
        # Initial convolution - x is just the image, NOT concatenated with time
        h = self.init_conv(x)
        hs = [h]
        
        # Downsampling - pass time embedding to ResnetBlocks
        for module in self.downs:
            for layer in module:
                if isinstance(layer, ResnetBlock):
                    h = layer(h, t_emb)
                else:
                    h = layer(h)
            hs.append(h)
        
        # Middle
        for layer in self.middle_block:
            if isinstance(layer, ResnetBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)
        
        # Upsampling
        for module in self.ups:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:
                if isinstance(layer, ResnetBlock):
                    h = layer(h, t_emb)
                else:
                    h = layer(h)
        
        # Final
        return self.final_conv(h)


class DiffusionModel(nn.Module):
    def __init__(
        self,
        model,
        image_size,
        timesteps=1000,
        sampling_timesteps=None,
        loss_type='l2',
        objective='pred_noise',
        beta_schedule='cosine',
        min_beta=0.0001,
        max_beta=0.02,
    ):
        super().__init__()
        self.model = model
        self.channels = model.in_channels  # Subtract time channel
        self.image_size = image_size
        self.objective = objective
        self.loss_type = loss_type
        
        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps if sampling_timesteps else timesteps
        
        if beta_schedule == 'linear':
            betas = torch.linspace(min_beta, max_beta, timesteps)
        elif beta_schedule == 'cosine':
            s = 0.008
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0, 0.999)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        
        # Register buffer for pytorch to automatically move to device
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        
        # Log calculation clipped for numerical stability
        self.register_buffer('posterior_log_variance_clipped', 
                            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1', 
                            betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', 
                            (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
    
    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict x_0 from x_t and the predicted noise.
        """
        # Reshape t for proper broadcasting
        t_shape = t.shape
        t = t.view(*t_shape, 1, 1, 1)
        
        return (
            self.sqrt_recip_alphas_cumprod[t] * x_t -
            self.sqrt_recipm1_alphas_cumprod[t] * noise
        )
    
    def q_posterior(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0).
        """
        # Reshape t for proper broadcasting
        t_shape = t.shape
        t = t.view(*t_shape, 1, 1, 1)
        
        posterior_mean = (
            self.posterior_mean_coef1[t] * x_start +
            self.posterior_mean_coef2[t] * x_t
        )
        posterior_variance = self.posterior_variance[t]
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def q_sample(self, x_start, t, noise=None):
        """
        Sample from q(x_t | x_0) using the forward diffusion process.
        """
        noise = torch.randn_like(x_start) if noise is None else noise
        
        # Reshape t for proper broadcasting
        t_shape = t.shape
        t = t.view(*t_shape, 1, 1, 1)
        
        return (
            self.sqrt_alphas_cumprod[t] * x_start +
            self.sqrt_one_minus_alphas_cumprod[t] * noise
        )
    
    def p_mean_variance(self, x, t, cond=None, clip_denoised=True):
        # Predict noise or x_start
        model_output = self.model(x, t, cond)
        
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
        elif self.objective == 'pred_x0':
            x_start = model_output
        else:
            raise ValueError(f'unknown objective {self.objective}')
        
        if clip_denoised:
            x_start = torch.clamp(x_start, -1., 1.)
            
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start, x, t)
        return model_mean, posterior_variance, posterior_log_variance, x_start
    
    @torch.no_grad()
    def p_sample(self, x, t, cond=None, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=t, cond=cond, clip_denoised=clip_denoised
        )
        noise = torch.randn_like(x)
        # No noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    
    @torch.no_grad()
    def p_sample_loop(self, shape, cond=None):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        
        if self.sampling_timesteps < self.timesteps:
            sampling_indices = torch.linspace(0, self.timesteps - 1, 
                                              self.sampling_timesteps, dtype=torch.long)
        else:
            sampling_indices = torch.arange(self.timesteps - 1, -1, -1)
            
        for i in tqdm(sampling_indices, desc='Sampling'):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, cond)
        
        return img
    
    @torch.no_grad()
    def sample(self, batch_size=16, cond=None):
        channels = self.channels
        image_size = self.image_size
        sample_shape = (batch_size, channels, image_size, image_size)
        return self.p_sample_loop(sample_shape, cond)
    
    @torch.no_grad()
    def reconstruct(self, x):
        batch_size, channels, image_size, image_size = x.shape
        t = torch.ones(batch_size, device=x.device).long() * (self.timesteps - 1)
        
        # Add noise to the input image according to the timestep
        noised_x = self.q_sample(x, t)
        
        # Reconstruct by denoising
        return self.p_sample_loop(noised_x.shape, None)
    
    # def q_sample(self, x_start, t, noise=None):
    #     noise = torch.randn_like(x_start) if noise is None else noise
    #     return (
    #         self.sqrt_alphas_cumprod[t, None, None, None] * x_start +
    #         self.sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
    #     )
    
    def p_losses(self, x_start, t, cond=None, noise=None):
        b, c, h, w = x_start.shape
        noise = torch.randn_like(x_start) if noise is None else noise
        
        # Noisy images
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Predict noise or x_start
        model_out = self.model(x_noisy, t, cond)
        
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')
            
        if self.loss_type == 'l1':
            loss = F.l1_loss(model_out, target)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(model_out, target)
        else:
            raise ValueError(f'unknown loss type {self.loss_type}')
            
        return loss
    
    def forward(self, x, cond=None):
        b, c, h, w, device = *x.shape, x.device
        t = torch.randint(0, self.timesteps, (b,), device=device).long()
        return self.p_losses(x, t, cond)