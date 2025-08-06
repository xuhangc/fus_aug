# stable_diffusion_train.py
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import os
import matplotlib.pyplot as plt
from data import NPZDataLoader
from tqdm import tqdm

from models.stable_diffusion import UNet, AutoencoderKL, CLIP, DiffusionModel, StableDiffusionConfig


def plot_images(pred, original=None):
    n = pred.size(0)
    pred = pred * 0.5 + 0.5
    pred = pred.clamp(0, 1)
    img = pred.cpu().detach()

    if original is not None:
        original = original * 0.5 + 0.5
        original = original.clamp(0, 1)
        original = original.cpu().detach()
        img = torch.cat([original, img], dim=0)

    img_grid = make_grid(img, nrow=n)
    img_grid = img_grid.permute(1, 2, 0).numpy()
    img_grid = (img_grid * 255).astype("uint8")
    plt.imshow(img_grid)
    plt.axis("off")


if __name__ == "__main__":
    session = "S1"

    model_name = "StableDiffusion"
    os.makedirs(model_name, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 配置模型参数
    config = StableDiffusionConfig(
        resolution=128,
        in_channels=1,
        model_channels=128,
        out_channels=1,
        num_res_blocks=2,
        attention_resolutions=(8, 16, 32),
        dropout=0.1,
        channel_mult=(1, 2, 4, 8),
        num_heads=8,
        use_scale_shift_norm=True,
        num_classes=2,  # 根据您的数据集类别数
        timesteps=1000,  # 扩散步骤数
        beta_schedule="linear",  # 噪声调度
        beta_start=0.0001,
        beta_end=0.02,
    )

    # 初始化模型组件
    vae = AutoencoderKL(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        latent_channels=4,  # 潜在空间通道数
        hidden_dims=[32, 64, 128, 256],
    )
    
    # 简化版本，不使用CLIP文本编码器，而是直接使用类别标签
    # text_encoder = CLIP(
    #     embed_dim=512,
    #     vocab_size=8192,
    #     num_heads=8,
    # )
    
    # UNet模型需要接收条件输入
    unet = UNet(
        in_channels=4 + config.num_classes,  # 潜在通道 + 条件嵌入
        model_channels=config.model_channels,
        out_channels=4,
        num_res_blocks=config.num_res_blocks,
        attention_resolutions=config.attention_resolutions,
        dropout=config.dropout,
        channel_mult=config.channel_mult,
        num_heads=config.num_heads,
        use_scale_shift_norm=config.use_scale_shift_norm,
    )
    
    diffusion = DiffusionModel(
        unet=unet,
        timesteps=config.timesteps,
        beta_schedule=config.beta_schedule,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
    )

    # 数据加载
    train_dataset = NPZDataLoader(f'{session}_train.npz')
    test_dataset = NPZDataLoader(f'{session}_test.npz')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=16)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=16)

    # 第一阶段：训练VAE
    print("=" * 10 + "Training VAE" + "=" * 10)
    vae = vae.to(device)
    vae_optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-4)

    temp_loss = float('inf')
    best_vae_checkpoint = ""
    
    for epoch in range(100):  # VAE训练轮数
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        
        for i, (x, c) in enumerate(tqdm(train_loader)):
            x, c = x.to(device), c.to(device).flatten()
            vae_optimizer.zero_grad()
            
            # VAE前向传播
            reconstructed, mu, log_var = vae(x)
            
            # 计算重建损失和KL散度
            recon_loss = F.mse_loss(reconstructed, x)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            kl_loss = kl_loss / torch.numel(x)
            
            # 总损失
            loss = recon_loss + 0.1 * kl_loss
            
            loss.backward()
            vae_optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
        
        epoch_loss /= len(train_loader)
        epoch_recon_loss /= len(train_loader)
        epoch_kl_loss /= len(train_loader)
        
        print(f"Epoch: {epoch}, Loss: {epoch_loss}, Recon Loss: {epoch_recon_loss}, KL Loss: {epoch_kl_loss}")
        
        # 保存模型
        torch.save(vae.state_dict(), f"{model_name}/{session}_{epoch}_vae.pth")
        
        # 每个epoch评估模型
        if epoch % 1 == 0:
            with torch.no_grad():
                total_loss = 0
                for i, (x, c) in enumerate(tqdm(test_loader)):
                    x, c = x.to(device), c.to(device)
                    reconstructed, mu, log_var = vae(x)
                    recon_loss = F.mse_loss(reconstructed, x)
                    total_loss += recon_loss.item()
                
                total_loss /= len(test_loader)
                
                # stable_diffusion_train.py (续)
                if total_loss < temp_loss:
                    temp_loss = total_loss
                    best_vae_checkpoint = f"{model_name}/{session}_{epoch}_vae.pth"
                    
                    # 保存重建图像
                    x = x[:10, :].to(device)
                    reconstructed, _, _ = vae(x)
                    
                    plot_images(pred=reconstructed, original=x)
                    plt.savefig(f"{model_name}/{session}_vae_{epoch}.png")
                    plt.close()
                
                print(f"Epoch: {epoch}, Test Loss: {total_loss}")

    # 第二阶段：训练扩散模型
    print("=" * 10 + "Training Diffusion Model" + "=" * 10)
    
    # 加载训练好的VAE
    vae.load_state_dict(torch.load(best_vae_checkpoint))
    vae.eval()
    
    # 冻结VAE参数
    for param in vae.parameters():
        param.requires_grad = False
    
    # 将模型移至设备
    unet = unet.to(device)
    diffusion = diffusion.to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5)
    
    # 打印模型参数量
    print(f"VAE Parameters: {sum(p.numel() for p in vae.parameters())/1e6:.2f}M")
    print(f"UNet Parameters: {sum(p.numel() for p in unet.parameters())/1e6:.2f}M")
    
    temp_loss = float('inf')
    
    for epoch in range(200):  # 扩散模型训练轮数
        epoch_loss = 0
        
        for i, (x, c) in enumerate(tqdm(train_loader)):
            x, c = x.to(device), c.to(device).flatten()
            optimizer.zero_grad()
            
            # 使用VAE编码图像到潜在空间
            with torch.no_grad():
                mu, log_var = vae.encode(x)
                z = vae.reparameterize(mu, log_var)
            
            # 将类别标签转换为条件嵌入
            c_embed = torch.nn.functional.one_hot(c, num_classes=config.num_classes).float()
            
            # 扩展条件嵌入到与潜在表示相同的空间维度
            c_embed = c_embed.unsqueeze(-1).unsqueeze(-1)
            c_embed = c_embed.expand(-1, -1, z.shape[2], z.shape[3])
            
            # 连接潜在表示和条件嵌入
            z_cond = torch.cat([z, c_embed], dim=1)
            
            # 训练扩散模型
            loss = diffusion.training_step(z, condition=c_embed)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_loader)
        print(f"Epoch: {epoch}, Loss: {epoch_loss}")
        
        # 每10个epoch保存模型
        if epoch % 10 == 0:
            torch.save({
                'unet': unet.state_dict(),
                'diffusion': diffusion.state_dict(),
            }, f"{model_name}/{session}_{epoch}_diffusion.pth")
        
        # 每5个epoch生成样本
        if epoch % 5 == 0:
            with torch.no_grad():
                # 为每个类别生成样本
                for class_idx in range(config.num_classes):
                    # 创建条件嵌入
                    c_embed = torch.zeros(1, config.num_classes).to(device)
                    c_embed[0, class_idx] = 1.0
                    
                    # 扩展条件嵌入
                    c_embed_expanded = c_embed.unsqueeze(-1).unsqueeze(-1)
                    c_embed_expanded = c_embed_expanded.expand(-1, -1, config.resolution//8, config.resolution//8)
                    
                    # 生成潜在表示
                    z_sample = diffusion.sample(
                        batch_size=1,
                        image_size=config.resolution//8,  # VAE通常会将图像尺寸缩小8倍
                        channels=4,  # 潜在空间通道数
                        condition=c_embed_expanded,
                        guidance_scale=3.0  # 条件引导尺度
                    )
                    
                    # 解码为图像
                    x_sample = vae.decode(z_sample)
                    
                    # 保存生成的图像
                    plot_images(pred=x_sample)
                    plt.savefig(f"{model_name}/{session}_diffusion_{epoch}_class{class_idx}.png")
                    plt.close()
                
                # 计算测试集损失
                total_loss = 0
                for i, (x, c) in enumerate(tqdm(test_loader)):
                    x, c = x.to(device), c.to(device).flatten()
                    
                    # 编码到潜在空间
                    with torch.no_grad():
                        mu, log_var = vae.encode(x)
                        z = vae.reparameterize(mu, log_var)
                    
                    # 条件嵌入
                    c_embed = torch.nn.functional.one_hot(c, num_classes=config.num_classes).float()
                    c_embed = c_embed.unsqueeze(-1).unsqueeze(-1)
                    c_embed = c_embed.expand(-1, -1, z.shape[2], z.shape[3])
                    
                    # 计算扩散损失
                    loss = diffusion.training_step(z, condition=c_embed)
                    total_loss += loss.item()
                
                total_loss /= len(test_loader)
                
                if total_loss < temp_loss:
                    temp_loss = total_loss
                    torch.save({
                        'unet': unet.state_dict(),
                        'diffusion': diffusion.state_dict(),
                    }, f"{model_name}/{session}_best_diffusion.pth")
                
                print(f"Epoch: {epoch}, Test Loss: {total_loss}")
