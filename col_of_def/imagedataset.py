# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
from random import randint
from torchvision.transforms import functional as F

class CustomRandomCrop:
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, img):
        width, height = img.size
        crop_h, crop_w = self.output_size
        if width < crop_w or height < crop_h:
            return img # you can add padding here if the image is smaller than the crop size
        i = randint(0, height - crop_h)
        j = randint(0, width - crop_w)
        return F.crop(img, i, j, crop_h, crop_w)

class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:
    .. code-block::
        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png
    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root="./openimages", split="train",transform=True, height=256, width=256):
        splitdir = Path(root) / split / "data"

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]

        self.transform = transform
        self.transform_img = transforms.Compose(
            [CustomRandomCrop(height), transforms.ToTensor()]
        )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")

        if self.transform:
            img = self.transform_img(img)
            
        mask = torch.ones_like(img[0:1,:,:])
        masked_image = img.clone()
        image_with_alpha = torch.cat([img, mask], dim=0)

        return masked_image, mask, img, mask, image_with_alpha

    def __len__(self):
        return len(self.samples)