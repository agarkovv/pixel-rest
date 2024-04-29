from os import listdir
from os.path import join

import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    RandomCrop,
    Resize,
    ToPILImage,
    ToTensor,
)


def is_image_file(filename):
    return any(
        filename.endswith(extension)
        for extension in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]
    )


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose(
        [
            RandomCrop(crop_size),
            ToTensor(),
        ]
    )


def train_lr_transform(crop_size, upscale_factor):
    return Compose(
        [
            ToPILImage(),
            Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
            ToTensor(),
        ]
    )


def display_transform():
    return Compose([ToPILImage(), Resize(400), CenterCrop(400), ToTensor()])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super().__init__()
        self.image_filenames = [
            join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)
        ]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [
            join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)
        ]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + "/SRF_" + str(upscale_factor) + "/data/"
        self.hr_path = dataset_dir + "/SRF_" + str(upscale_factor) + "/target/"
        self.upscale_factor = upscale_factor
        self.lr_filenames = [
            join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)
        ]
        self.hr_filenames = [
            join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)
        ]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split("/")[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize(
            (self.upscale_factor * h, self.upscale_factor * w),
            interpolation=Image.BICUBIC,
        )
        hr_restore_img = hr_scale(lr_image)
        return (
            image_name,
            ToTensor()(lr_image),
            ToTensor()(hr_restore_img),
            ToTensor()(hr_image),
        )

    def __len__(self):
        return len(self.lr_filenames)


class SRGANDataModule(L.LightningDataModule):
    def __init__(
        self,
        # dir paths
        train_dataset_dir,
        val_dataset_dir,
        test_dataset_dir,
        # bool shuffle data or not
        train_shuffle,
        val_shuffle,
        test_shuffle,
        # batch sizes
        train_batch_size,
        val_batch_size,
        test_batch_size,
        num_workers,
        crop_size,
        upscale_factor,
    ):
        super().__init__()
        self.save_hyperparameters()
        # dir paths
        self.train_dataset_dir = train_dataset_dir
        self.val_dataset_dir = val_dataset_dir
        self.test_dataset_dir = test_dataset_dir

        # bool shuffle data or not
        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle
        self.test_shuffle = test_shuffle

        # batch sizes
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

        self.num_workers = num_workers

        self.crop_size = crop_size
        self.upscale_factor = upscale_factor

    def prepare_data(self) -> None:
        # TODO: add downloading data from gdrive / s3
        pass

    def setup(self, stage: str) -> None:
        self.train_dataset = TrainDatasetFromFolder(
            dataset_dir=self.train_dataset_dir,
            crop_size=self.crop_size,
            upscale_factor=self.upscale_factor,
        )
        self.val_dataset = ValDatasetFromFolder(
            dataset_dir=self.val_dataset_dir, upscale_factor=self.upscale_factor
        )
        # TODO: add test_dataset
        # self.test_dataset = ...

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.train_batch_size,
            shuffle=self.train_shuffle,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.val_batch_size,
            shuffle=self.val_shuffle,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        # TODO: implement test dataloader
        return super().test_dataloader()
