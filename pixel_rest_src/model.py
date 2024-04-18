import math

import lightning as L
import torch
from omegaconf import OmegaConf
from torch import nn

from pixel_rest_src.loss import GeneratorLoss


class Generator(nn.Module):
    def __init__(
        self,
        out_channels=64,
        scale_factor=2,
        block1_kernel_size=9,
        block1_padding=4,
        block7_kernel_size=3,
        block7_padding=1,
        block8_kernel_size=9,
        block8_padding=4,
    ):
        upsample_block_num = int(math.log(scale_factor, 2))

        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(
                3, out_channels, kernel_size=block1_kernel_size, padding=block1_padding
            ),
            nn.PReLU(),
        )
        self.block2 = ResidualBlock(out_channels)
        self.block3 = ResidualBlock(out_channels)
        self.block4 = ResidualBlock(out_channels)
        self.block5 = ResidualBlock(out_channels)
        self.block6 = ResidualBlock(out_channels)
        self.block7 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=block7_kernel_size,
                padding=block7_padding,
            ),
            nn.BatchNorm2d(out_channels),
        )
        block8 = [UpsampleBLock(out_channels, 2) for _ in range(upsample_block_num)]
        block8.append(
            nn.Conv2d(
                out_channels, 3, kernel_size=block8_kernel_size, padding=block8_padding
            )
        )
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        # just a tricky way to get higher derivative at x = 0 (x2 than sigmoid)
        return (torch.tanh(block8) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self, out_channels=64, kernel_size=3, padding=1, leaky_coef=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(leaky_coef),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(leaky_coef),
            nn.Conv2d(
                out_channels, 2 * out_channels, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm2d(2 * out_channels),
            nn.LeakyReLU(leaky_coef),
            nn.Conv2d(
                2 * out_channels,
                2 * out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
            ),
            nn.BatchNorm2d(2 * out_channels),
            nn.LeakyReLU(leaky_coef),
            nn.Conv2d(
                2 * out_channels,
                4 * out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm2d(4 * out_channels),
            nn.LeakyReLU(leaky_coef),
            nn.Conv2d(
                4 * out_channels,
                4 * out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
            ),
            nn.BatchNorm2d(4 * out_channels),
            nn.LeakyReLU(leaky_coef),
            nn.Conv2d(
                4 * out_channels,
                8 * out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm2d(8 * out_channels),
            nn.LeakyReLU(leaky_coef),
            nn.Conv2d(
                8 * out_channels,
                8 * out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
            ),
            nn.BatchNorm2d(8 * out_channels),
            nn.LeakyReLU(leaky_coef),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(8 * out_channels, 16 * out_channels, kernel_size=1),
            nn.LeakyReLU(leaky_coef),
            nn.Conv2d(16 * out_channels, 1, kernel_size=1),
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels * up_scale**2,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class SRGANModel(L.LightningModule):
    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.automatic_optimization = False  # to control optimizers manually
        self.save_hyperparameters()
        self.cfg = cfg
        self.netG = Generator(
            cfg.model.generator.out_channels,
            cfg.train.upscale_factor,
            cfg.model.generator.block1.kernel_size,
            cfg.model.generator.block1.padding,
            cfg.model.generator.block7.kernel_size,
            cfg.model.generator.block7.padding,
            cfg.model.generator.block8.kernel_size,
            cfg.model.generator.block8.padding,
        )
        self.netD = Discriminator(
            cfg.model.generator.out_channels,
            cfg.model.discriminator.kernel_size,
            cfg.model.discriminator.padding,
            cfg.model.discriminator.leaky_coef,
        )
        self.generator_criterion = GeneratorLoss(
            cfg.model.loss.generator.adversarial_weight,
            cfg.model.loss.generator.perception_weight,
            cfg.model.loss.generator.tv_weight,
        )

    def forward(self, batch):
        return self.netG(batch)

    def training_step(self, batch, batch_idx):
        optimizerG, optimizerD = self.optimizers()
        data, real_img = batch
        fake_img = self.netG(data)

        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ###########################
        self.netD.zero_grad()
        real_out = self.netD(real_img).mean()
        fake_out = self.netD(fake_img).mean()
        d_loss = 1 - real_out + fake_out
        d_loss.backward()
        optimizerD.step()

        ############################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ###########################
        self.netG.zero_grad()
        fake_img = self.netG(data)
        fake_out = self.netD(fake_img).mean()
        g_loss = self.generator_criterion(fake_out, fake_img, real_img)
        g_loss.backward()
        optimizerG.step()

        self.log("g_loss", g_loss.item(), on_step=True, on_epoch=False, prog_bar=True)
        self.log("d_loss", d_loss.item(), on_step=True, on_epoch=False, prog_bar=True)

        return {"g_loss": g_loss, "d_loss": d_loss}
