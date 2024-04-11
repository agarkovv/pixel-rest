import os
from math import log10

import hydra
import torch.utils.data
import torchvision.utils as utils
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from pixel_rest_src import pytorch_ssim
from pixel_rest_src.data_utils import ValDatasetFromFolder, display_transform


@hydra.main(config_path="./", config_name="config", version_base="1.3")
def main(cfg: OmegaConf) -> None:
    inf_set = ValDatasetFromFolder("data/inf", upscale_factor=cfg.infer.upscale_factor)
    inf_loader = DataLoader(dataset=inf_set, num_workers=4, batch_size=1, shuffle=False)

    netG = torch.load(
        "epochs/netG_epoch_%d_%d.pth" % (cfg.infer.upscale_factor, cfg.infer.num_epochs)
    )
    print("# generator parameters:", sum(param.numel() for param in netG.parameters()))

    if torch.cuda.is_available():
        netG.cuda()

    netG.eval()
    out_path = "training_results/SRF_" + str(cfg.infer.upscale_factor) + "/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with torch.no_grad():
        inf_bar = tqdm(inf_loader)
        valing_results = {
            "mse": 0,
            "ssims": 0,
            "psnr": 0,
            "ssim": 0,
            "batch_sizes": 0,
        }
        inf_images = []
        for inf_lr, inf_hr_restore, inf_hr in inf_bar:
            batch_size = inf_lr.size(0)
            valing_results["batch_sizes"] += batch_size
            lr = inf_lr
            hr = inf_hr
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
            inf_bar.set_description(
                desc="[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f"
                % (valing_results["psnr"], valing_results["ssim"])
            )

            inf_images.extend(
                [
                    display_transform()(inf_hr_restore.squeeze(0)),
                    display_transform()(hr.data.cpu().squeeze(0)),
                    display_transform()(sr.data.cpu().squeeze(0)),
                ]
            )
        inf_images = torch.stack(inf_images)
        inf_images = torch.chunk(inf_images, inf_images.size(0) // cfg.infer.chunk_size)
        inf_save_bar = tqdm(inf_images, desc="[saving training results]")
        index = 1
        for image in inf_save_bar:
            image = utils.make_grid(image, nrow=3, padding=5)
            utils.save_image(
                image,
                out_path + "epoch_%d_index_%d.png" % (cfg.infer.num_epochs, index),
                padding=5,
            )
            index += 1


if __name__ == "__main__":
    main()
