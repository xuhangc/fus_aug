# models/stable_diffusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class StableDiffusionConfig:
    def __init__(
        self,
        resolution,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout,
        channel_mult,
        num_heads,
        use_scale_shift_norm,
        num_classes,
        timesteps,
        beta_schedule,
        beta_start,
        beta_end,
    ):
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.use_scale_shift_norm = use_scale_shift_norm
        self.num_classes = num_classes
        self.timesteps = timesteps
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end


class SinusoidalPositionEmbeddings(nn.Module):
    """时间步的正弦位置编码"""
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


class ResidualBlock(nn.Module):
    """UNet中的残差块"""
    def __init__(self, in_channels, out_channels, time_dim, dropout):
        super().__init__()
        self.time_mlp = nn.Linear(time_dim, out_channels)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.act1 = nn.SiLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()
        
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb):
        h = self.act1(self.norm1(self.conv1(x)))
        
        # 添加时间嵌入
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        h = self.act2(self.norm2(self.conv2(self.dropout(h))))
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """自注意力块"""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        # 重塑为多头注意力格式
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)
        
        # 计算注意力
        scale = (c // self.num_heads) ** -0.5
        attn = torch.einsum("bhci,bhcj->bhij", q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力
        out = torch.einsum("bhij,bhcj->bhci", attn, v)
        out = out.reshape(b, c, h, w)
        
        return x + self.proj_out(out)


class DownSample(nn.Module):
    """下采样层"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class UpSample(nn.Module):
    """上采样层"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """UNet模型，用于预测噪声"""
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout,
        channel_mult,
        num_heads,
        use_scale_shift_norm,
    ):
        super().__init__()
        self.time_dim = model_channels * 4
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        
        # 初始卷积
        self.init_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # 下采样部分
        self.downs = nn.ModuleList([])
        
        curr_res = 1
        in_ch = model_channels
        down_block_chans = []
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            
            for _ in range(num_res_blocks):
                self.downs.append(ResidualBlock(in_ch, out_ch, self.time_dim, dropout))
                in_ch = out_ch
                down_block_chans.append(in_ch)
                
                # 添加注意力层
                if curr_res in attention_resolutions:
                    self.downs.append(AttentionBlock(in_ch, num_heads=num_heads))
            
            # 下采样，除了最后一层
            if level != len(channel_mult) - 1:
                self.downs.append(DownSample(in_ch))
                down_block_chans.append(in_ch)
                curr_res *= 2
        
        # 中间块
        self.mid_block1 = ResidualBlock(in_ch, in_ch, self.time_dim, dropout)
        self.mid_attn = AttentionBlock(in_ch, num_heads=num_heads)
        self.mid_block2 = ResidualBlock(in_ch, in_ch, self.time_dim, dropout)
        
        # 上采样部分
        self.ups = nn.ModuleList([])
        
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            
            for _ in range(num_res_blocks + 1):
                self.ups.append(ResidualBlock(
                    in_ch + down_block_chans.pop(), out_ch, self.time_dim, dropout
                ))
                in_ch = out_ch
                
                # 添加注意力层
                if curr_res in attention_resolutions:
                    self.ups.append(AttentionBlock(in_ch, num_heads=num_heads))
            
            # 上采样，除了第一层
            if level != 0:
                self.ups.append(UpSample(in_ch))
                curr_res //= 2
        
        # 输出层
        self.final_conv = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_channels, 3, padding=1),
        )
    
    def forward(self, x, timestep, context=None):
        # 时间嵌入
        t_emb = self.time_embed(timestep)
        
        # 初始特征
        h = self.init_conv(x)
        
        # 保存下采样特征用于跳跃连接
        hs = [h]
        
        # 下采样
        for module in self.downs:
            if isinstance(module, ResidualBlock):
                h = module(h, t_emb)
            else:
                h = module(h)
            hs.append(h)
        
        # 中间块
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        # 上采样
        for module in self.ups:
            if isinstance(module, ResidualBlock):
                # 添加跳跃连接
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, t_emb)
            else:
                h = module(h)
        
        # 最终输出
        return self.final_conv(h)


class AutoencoderKL(nn.Module):
    """VAE模型，用于将图像编码到潜在空间"""
    def __init__(self, in_channels, out_channels, latent_channels=4, hidden_dims=[32, 64, 128, 256]):
        super().__init__()
        
        # 编码器
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(8, h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # 均值和方差预测
        self.fc_mu = nn.Conv2d(hidden_dims[-1], latent_channels, kernel_size=1)
        self.fc_var = nn.Conv2d(hidden_dims[-1], latent_channels, kernel_size=1)
        
        # 解码器
        modules = []
        
        self.decoder_input = nn.Conv2d(latent_channels, hidden_dims[-1], kernel_size=1)
        
        hidden_dims.reverse()
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.GroupNorm(8, hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
        
        self.decoder = nn.Sequential(*modules)
        
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )
    
    def encode(self, x):
        """编码图像到潜在空间"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """重参数化技巧"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """从潜在表示解码图像"""
        h = self.decoder_input(z)
        h = self.decoder(h)
        return self.final_layer(h)
    
    def forward(self, x):
        """前向传播"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mu, log_var


class CLIP(nn.Module):
    """简化版的CLIP文本编码器"""
    def __init__(self, embed_dim, vocab_size, num_heads):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, 77, embed_dim))
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True
            ),
            num_layers=6
        )
        
        self.ln_final = nn.LayerNorm(embed_dim)
        self.text_projection = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, text_tokens):
        x = self.token_embedding(text_tokens)
        x = x + self.position_embedding
        x = self.transformer(x)
        x = self.ln_final(x)
        x = x[:, 0]  # 取CLS token
        x = self.text_projection(x)
        return x


class DiffusionModel(nn.Module):
    """扩散模型"""
    def __init__(self, unet, timesteps=1000, beta_schedule="linear", beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.unet = unet
        self.timesteps = timesteps
        
        # 设置噪声调度
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        elif beta_schedule == "cosine":
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0, 0.999)
        
        # 预计算扩散过程中的各种常数
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 计算扩散q(x_t | x_{t-1})和其他常数的系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        
        # 计算后验方差
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
    
    def q_sample(self, x_start, t, noise=None):
        """前向扩散过程"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start, t, condition=None, noise=None):
        """计算损失"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # 添加噪声
        x_noisy = self.q_sample(x_start, t, noise=noise)
        
        # 预测噪声
        predicted_noise = self.unet(x_noisy, t, context=condition)
        
        # 计算简单的MSE损失
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    def training_step(self, x_start, condition=None):
        """训练步骤"""
        b, c, h, w = x_start.shape
        device = x_start.device
        
        # 随机采样时间步
        t = torch.randint(0, self.timesteps, (b,), device=device).long()
        
        return self.p_losses(x_start, t, condition)
    
    @torch.no_grad()
    def p_sample(self, x, t, t_index, condition=None):
        """单步去噪采样"""
        betas_t = self.betas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        # 预测噪声
        predicted_noise = self.unet(x, t, context=condition)
        
        # 计算均值
        mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        
        # 无噪声的情况
        if t_index == 0:
            return mean
        
        # 添加噪声
        noise = torch.randn_like(x)
        variance = self.posterior_variance[t].reshape(-1, 1, 1, 1)
        
        return mean + torch.sqrt(variance) * noise
    
    @torch.no_grad()
    def sample(self, batch_size, image_size, channels, condition=None, guidance_scale=3.0):
        """从噪声生成图像"""
        device = next(self.unet.parameters()).device
        
        # 初始化为纯噪声
        img = torch.randn(batch_size, channels, image_size, image_size, device=device)
        
        # 逐步去噪
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            # 条件引导
            if condition is not None and guidance_scale > 1.0:
                # 无条件预测
                self.unet.eval()
                with torch.no_grad():
                    uncond_predicted_noise = self.unet(img, t, context=None)
                    
                    # 有条件预测
                    cond_predicted_noise = self.unet(img, t, context=condition)
                    
                    # 执行引导
                    predicted_noise = uncond_predicted_noise + guidance_scale * (cond_predicted_noise - uncond_predicted_noise)
                
                # 手动计算去噪步骤
                betas_t = self.betas[t].reshape(-1, 1, 1, 1)
                sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
                sqrt_recip_alphas_t = self.sqrt_recip_alphas_cumprod[t].reshape(-1, 1, 1, 1)
                
                # 计算均值
                mean = sqrt_recip_alphas_t * (img - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
                
                # 无噪声的情况
                if i == 0:
                    img = mean
                else:
                    # 添加噪声
                    noise = torch.randn_like(img)
                    variance = self.posterior_variance[t].reshape(-1, 1, 1, 1)
                    img = mean + torch.sqrt(variance) * noise
            else:
                # 标准采样
                img = self.p_sample(img, t, i, condition)
        
        return img
