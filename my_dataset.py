import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()
        # self.flag = "training" if train else "test"
        self.flag = "train" if train else "val"
        # data_root = os.path.join(root, "DRIVE", self.flag)
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        self.transforms = transforms
        # img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".tif")]
        # self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        # self.manual = [os.path.join(data_root, "1st_manual", i.split("_")[0] + "_manual1.gif")
        img_names = [i for i in os.listdir(os.path.join(root, "img_dir", self.flag)) if i.endswith(".jpg")]
        self.img_list = [os.path.join(root, "img_dir", self.flag, i) for i in img_names]
        # self.manual = [os.path.join(data_root, "1st_manual", i.split("_")[0] + "_manual1.gif")
        #                for i in img_names]
        # # check files
        # for i in self.manual:
        #     if os.path.exists(i) is False:
        #         raise FileNotFoundError(f"file {i} does not exists.")

        self.masks = [os.path.join(root, "ann_dir", self.flag, i.split(".")[0] + ".png")
                         for i in img_names]
        # check files
        for i in self.masks:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        target = Image.open(self.masks[idx]).convert('L')

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


# dataset = DriveDataset(root="./data/", train=True)
# d1 = dataset[0]
# print(d1)

