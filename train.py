import os
from math import log10

import hydra
import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from omegaconf import OmegaConf
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from pixel_rest_src import pytorch_ssim
from pixel_rest_src.data_utils import (
    TrainDatasetFromFolder,
    ValDatasetFromFolder,
    display_transform,
)
from pixel_rest_src.loss import GeneratorLoss
from pixel_rest_src.model import Discriminator, Generator


@hydra.main(config_path="./", config_name="config", version_base="1.3")
def main(cfg: OmegaConf) -> None:
    text = (
        "==========================================\n"
        f"Training model with the following config:\n{cfg}\n"
        "=========================================="
    )
    print(text)

    train_set = TrainDatasetFromFolder(
        cfg.train.data_path,
        crop_size=cfg.train.crop_size,
        upscale_factor=cfg.train.upscale_factor,
    )
    val_set = ValDatasetFromFolder(
        cfg.val.data_path, upscale_factor=cfg.model.upscale_factor
    )
    train_loader = DataLoader(
        dataset=train_set,
        num_workers=cfg.train.num_workers,
        batch_size=cfg.train.batch_size,
        shuffle=cfg.train.shuffle,
    )
    val_loader = DataLoader(
        dataset=val_set,
        num_workers=cfg.train.num_workers,
        batch_size=cfg.val.batch_size,
        shuffle=cfg.val.shuffle,
    )

    netG = Generator(
        cfg.model.generator.out_channels,
        cfg.train.upscale_factor,
        cfg.model.generator.block1.kernel_size,
        cfg.model.generator.block1.padding,
        cfg.model.generator.block7.kernel_size,
        cfg.model.generator.block7.padding,
        cfg.model.generator.block8.kernel_size,
        cfg.model.generator.block8.padding,
    )
    print("# generator parameters:", sum(param.numel() for param in netG.parameters()))
    netD = Discriminator(
        cfg.model.generator.out_channels,
        cfg.model.discriminator.kernel_size,
        cfg.model.discriminator.padding,
        cfg.model.discriminator.leaky_coef,
    )
    print(
        "# discriminator parameters:", sum(param.numel() for param in netD.parameters())
    )

    generator_criterion = GeneratorLoss(
        cfg.model.loss.generator.adversarial_weight,
        cfg.model.loss.generator.perception_weight,
        cfg.model.loss.generator.tv_weight,
    )

    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()

    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())

    results = {
        "d_loss": [],
        "g_loss": [],
        "d_score": [],
        "g_score": [],
        "psnr": [],
        "ssim": [],
    }

    for epoch in range(1, cfg.train.num_epochs + 1):
        train_bar = tqdm(train_loader)
        running_results = {
            "batch_sizes": 0,
            "d_loss": 0,
            "g_loss": 0,
            "d_score": 0,
            "g_score": 0,
        }

        netG.train()
        netD.train()
        for data, target in train_bar:
            batch_size = data.size(0)
            running_results["batch_sizes"] += batch_size

            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()
            fake_img = netG(z)

            netD.zero_grad()
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward()
            optimizerD.step()

            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()
            # The two lines below are added to prevent runtime error in Google Colab ##
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            g_loss = generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()
            optimizerG.step()

            # loss for current batch before optimization
            running_results["g_loss"] += g_loss.item() * batch_size
            running_results["d_loss"] += d_loss.item() * batch_size
            running_results["d_score"] += real_out.item() * batch_size
            running_results["g_score"] += fake_out.item() * batch_size

            train_bar.set_description(
                desc="[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f"
                % (
                    epoch,
                    cfg.train.num_epochs,
                    running_results["d_loss"] / running_results["batch_sizes"],
                    running_results["g_loss"] / running_results["batch_sizes"],
                    running_results["d_score"] / running_results["batch_sizes"],
                    running_results["g_score"] / running_results["batch_sizes"],
                )
            )

        netG.eval()
        out_path = "training_results/SRF_" + str(cfg.train_upscale_factor) + "/"
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {
                "mse": 0,
                "ssims": 0,
                "psnr": 0,
                "ssim": 0,
                "batch_sizes": 0,
            }
            val_images = []
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results["batch_sizes"] += batch_size
                lr = val_lr
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                sr = netG(lr)

                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results["mse"] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results["ssims"] += batch_ssim * batch_size
                valing_results["psnr"] = 10 * log10(
                    (hr.max() ** 2)
                    / (valing_results["mse"] / valing_results["batch_sizes"])
                )
                valing_results["ssim"] = (
                    valing_results["ssims"] / valing_results["batch_sizes"]
                )
                val_bar.set_description(
                    desc="[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f"
                    % (valing_results["psnr"], valing_results["ssim"])
                )

                val_images.extend(
                    [
                        display_transform()(val_hr_restore.squeeze(0)),
                        display_transform()(hr.data.cpu().squeeze(0)),
                        display_transform()(sr.data.cpu().squeeze(0)),
                    ]
                )
            val_images = torch.stack(val_images)
            val_images = torch.chunk(
                val_images, val_images.size(0) // cfg.val.chunk_size
            )
            val_save_bar = tqdm(val_images, desc="[saving training results]")
            index = 1
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(
                    image,
                    out_path + "epoch_%d_index_%d.png" % (epoch, index),
                    padding=5,
                )
                index += 1

        # save model parameters
        torch.save(
            netG.state_dict(),
            "epochs/netG_epoch_%d_%d.pth" % (cfg.train_upscale_factor, epoch),
        )
        torch.save(
            netD.state_dict(),
            "epochs/netD_epoch_%d_%d.pth" % (cfg.train_upscale_factor, epoch),
        )
        # save loss\scores\psnr\ssim
        results["d_loss"].append(
            running_results["d_loss"] / running_results["batch_sizes"]
        )
        results["g_loss"].append(
            running_results["g_loss"] / running_results["batch_sizes"]
        )
        results["d_score"].append(
            running_results["d_score"] / running_results["batch_sizes"]
        )
        results["g_score"].append(
            running_results["g_score"] / running_results["batch_sizes"]
        )
        results["psnr"].append(valing_results["psnr"])
        results["ssim"].append(valing_results["ssim"])

        if epoch % 10 == 0 and epoch != 0:
            out_path = "statistics/"
            data_frame = pd.DataFrame(
                data={
                    "Loss_D": results["d_loss"],
                    "Loss_G": results["g_loss"],
                    "Score_D": results["d_score"],
                    "Score_G": results["g_score"],
                    "PSNR": results["psnr"],
                    "SSIM": results["ssim"],
                },
                index=range(1, epoch + 1),
            )
            data_frame.to_csv(
                out_path
                + "srf_"
                + str(cfg.train_upscale_factor)
                + "_train_results.csv",
                index_label="Epoch",
            )


if __name__ == "__main__":
    main()
