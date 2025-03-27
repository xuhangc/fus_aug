import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from data import NPZDataLoader
from einops import rearrange, repeat
import torch.nn as nn
import torch.nn.functional as F


# Set random seeds for reproducibility
def set_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- Model Components from dit_train.py ---

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
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
        ) if time_emb_dim else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        batch, height, width, channels = x.shape
        x = self.norm(x)
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b h w (heads d) -> b heads (h w) d', heads=self.heads), qkv)
        
        q = q * self.scale
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b heads (h w) d -> b h w (heads d)', heads=self.heads, h=height, w=width)
        
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, mlp_dim=None):
        super().__init__()
        mlp_dim = mlp_dim or (dim * 4)
        
        self.attn = Attention(dim, heads=heads, dim_head=dim_head)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
        x = x + self.attn(x)
        x = x + self.mlp(x)
        x = x.permute(0, 3, 1, 2)  # BHWC -> BCHW
        return x


class Upsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim
        self.conv = nn.Conv2d(dim, dim_out, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))


class Downsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim
        self.conv = nn.Conv2d(dim, dim_out, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        in_channels=1,
        model_channels=64,
        out_channels=1,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        channel_mult=(1, 2, 4, 8),
        num_heads=4,
        use_scale_shift_norm=True,
        dropout=0.0,
        label_dim=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.use_scale_shift_norm = use_scale_shift_norm
        self.label_dim = label_dim

        # Time embedding
        time_dim = model_channels * 4
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Label embedding
        if label_dim > 0:
            self.label_embedding = nn.Embedding(label_dim, time_dim)
        
        # Initial convolution
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        ])
        
        # Downsampling
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResnetBlock(
                        ch,
                        mult * model_channels,
                        time_emb_dim=time_dim,
                    )
                ]
                ch = mult * model_channels
                
                if ds in attention_resolutions:
                    layers.append(TransformerBlock(ch, heads=num_heads))
                    
                self.input_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)
                
            if level != len(channel_mult) - 1:
                self.input_blocks.append(nn.ModuleList([Downsample(ch, ch)]))
                input_block_chans.append(ch)
                ds *= 2
                
        # Middle block
        self.middle_block = nn.ModuleList([
            ResnetBlock(ch, ch, time_emb_dim=time_dim),
            TransformerBlock(ch, heads=num_heads),
            ResnetBlock(ch, ch, time_emb_dim=time_dim)
        ])
        
        # Upsampling
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResnetBlock(
                        ch + input_block_chans.pop(),
                        model_channels * mult,
                        time_emb_dim=time_dim,
                    )
                ]
                ch = model_channels * mult
                
                if ds in attention_resolutions:
                    layers.append(TransformerBlock(ch, heads=num_heads))
                    
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, ch))
                    ds //= 2
                    
                self.output_blocks.append(nn.ModuleList(layers))
                
        # Final layers
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )

    def forward(self, x, time, labels=None):
        # Time embedding
        time_emb = self.time_embedding(time)
        
        # Label conditioning
        if labels is not None and self.label_dim > 0:
            label_emb = self.label_embedding(labels.squeeze(-1).long())
            time_emb = time_emb + label_emb
            
        # Initial conv
        h = self.input_blocks[0](x)
        hs = [h]
        
        # Downsampling
        for module in self.input_blocks[1:]:
            if isinstance(module, nn.ModuleList):
                for layer in module:
                    if isinstance(layer, ResnetBlock):
                        h = layer(h, time_emb)
                    else:
                        h = layer(h)
            else:
                h = module(h)
                
            hs.append(h)
            
        # Middle
        for layer in self.middle_block:
            if isinstance(layer, ResnetBlock):
                h = layer(h, time_emb)
            else:
                h = layer(h)
                
        # Upsampling
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:
                if isinstance(layer, ResnetBlock):
                    h = layer(h, time_emb)
                else:
                    h = layer(h)
                    
        # Final layer
        return self.out(h)


def extract(a, t, x_shape):
    """Extract coefficients at specified timesteps t and reshape to match x_shape."""
    b, *_ = t.shape
    out = a.gather(-1, t).float()
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DiffusionModel:
    def __init__(self, model, timesteps=1000, beta_schedule='cosine', device='cuda'):
        self.model = model
        self.timesteps = timesteps
        self.device = device
        
        if beta_schedule == 'linear':
            betas = torch.linspace(0.0001, 0.02, timesteps).to(device)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps).to(device)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
            
        self.betas = betas
        
        # Pre-calculate different terms for closed form
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def p_sample(self, x_t, t, labels=None, clip_denoised=True):
        """Sample from p(x_{t-1} | x_t)"""
        model_output = self.model(x_t, t, labels)
        
        # Get the parameters for the posterior distribution
        posterior_mean_coef1 = extract(self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod), t, x_t.shape)
        posterior_mean_coef2 = extract(torch.sqrt(self.alphas) * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod), t, x_t.shape)
        
        # Predict original sample from noise
        pred_x0 = (x_t - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * model_output) / extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        
        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1., 1.)
            
        posterior_mean = posterior_mean_coef1 * pred_x0 + posterior_mean_coef2 * x_t
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        
        # Sample from the posterior
        noise = torch.randn_like(x_t) if t[0] > 0 else torch.zeros_like(x_t)
        return posterior_mean + torch.sqrt(posterior_variance) * noise


if __name__ == "__main__":
    # Set random seed for reproducibility
    set_seed(42)

    session = 'S2'
    model_name = "DiT"  # Diffusion Transformer

    val_dataset = NPZDataLoader(f'{session}_test.npz')
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    generator = DiffusionTransformer(
        in_channels=1,
        out_channels=1,
        model_channels=64,
        num_res_blocks=8,
        attention_resolutions=(8, 16),
        channel_mult=(1, 2, 4, 8),
        num_heads=4,
        label_dim=2,  # Binary classification (0 or 1)
    ).to(device)
    
    # Load the trained model
    checkpoint = torch.load(f'{model_name}/{session}_dit_model_checkpoint.pt', weights_only=True, map_location=device)
    generator.load_state_dict(checkpoint['model_state_dict'])
    generator.eval()

    # Setup diffusion process
    diffusion = DiffusionModel(
        model=generator,
        timesteps=1000,
        beta_schedule='cosine',
        device=device
    )

    # Create directory for saving results
    os.makedirs(f"{model_name}", exist_ok=True)

    # Set the number of inference steps
    inference_steps = 100
    diffusion.timesteps = inference_steps

    data_list = []
    label_list = []
    
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(tqdm(val_dataloader, desc="Generating samples")):
            # Move data to device
            real_imgs = imgs.to(device)
            labels = labels.to(device)
            
            # Generate random noise
            x_t = torch.randn((imgs.size(0), 1, 128, 128), device=device)
            
            # Perform the denoising diffusion process
            for t in tqdm(reversed(range(inference_steps)), desc=f"Sample {i+1}", leave=False):
                time_tensor = torch.full((imgs.size(0),), t, device=device, dtype=torch.long)
                x_t = diffusion.p_sample(x_t, time_tensor, labels)
            
            # Process the generated image to match format in test_biggan.py
            fake_imgs = x_t.clone()
            fake_imgs = fake_imgs.squeeze(0).permute(1, 2, 0)
            labels = labels.permute(1, 0)
            
            data_list.append(fake_imgs)
            label_list.append(labels)
    
    data_list = torch.cat(data_list, dim=2).cpu().numpy()
    label_list = torch.cat(label_list, dim=1).cpu().numpy()

    np.savez(f"{model_name}/{model_name}_{session}.npz", fus=data_list, label=label_list)
    print(f"Generated samples saved to {model_name}/{model_name}_{session}.npz")