"""
This file contains classes relevant to custom torch datasets.
"""

import builtins
import logging
import os

import numpy as np
import pandas as pd

import torch
from skimage import io, transform, color


logging.basicConfig(
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class SampleShapeError(builtins.Exception):
    """Custom Error for samples of the Dataset"""

    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)


class Rescale:  # pylint: disable=too-few-public-methods
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, the image height is matched
            to output_size, while keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape
        if isinstance(self.output_size, int):
            new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        return transform.resize(image, (new_h, new_w))


class CropVerticalStripe:  # pylint: disable=too-few-public-methods
    """Extract a vertical stripe of an image in a sample.
    Args:
        center (int): the column where the middle
            of the stripe is located.
        width (int): Desired width of the stripe.
            If the width is an even number, the center will be slightly
            to the left of the stripe.

    """

    def __init__(self, width):
        assert isinstance(width, int)
        self.width = width

    def __call__(self, img, center):
        if self.width % 2 == 0:
            left = center - self.width // 2 + 1
            right = center + self.width // 2
        else:
            left = center - self.width // 2
            right = center + self.width // 2

        # Ensuring the bounds are within the image
        if (left < 0) or (right >= img.shape[1]):
            raise SampleShapeError(
                f"Cannot extract a crop of width {self.width} "
                f"from a sample of width {img.shape[1]}"
            )

        img = img[:, left : right + 1]

        return img


class ToTensor:  # pylint: disable=too-few-public-methods
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img, label = sample["image"], sample["label"]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        if len(img.shape) == 3:
            img = img.transpose((2, 0, 1))

        return {"image": torch.from_numpy(img), "label": torch.from_numpy(label)}


class MangaPagesDataset(torch.utils.data.Dataset):
    """Manga Pages dataset."""

    def __init__(self, csv_file, root_dir, seed=42, sample_shape=(1200, 300)):
        """
        Initiates the Dataset. The files considered should not contain
            double pages splitted in two distinct files
        Arguments:
            csv_file (string): Path to the csv file containing data o.
                The files name should be ordered inside a same tome/volume.
                Contains columns ['File_name', 'Manga', 'Volume'] before
                initialization.
            root_dir (string): Directory with all the images.
            seed (int): seed used for reproductibility when creating random samples.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir

        self.seed = seed
        self.generator = np.random.default_rng(seed=self.seed)
        self.__create_labels()

        self.sample_shape = sample_shape

        self.rescale = Rescale(self.sample_shape[0])
        self.crop = CropVerticalStripe(self.sample_shape[1])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        The following was in the tuto but I think it is an error
        as os.path.join cannot accept an iterator as an argument
        if torch.is_tensor(idx):
            idx = idx.tolist()
        """
        img_name = os.path.join(self.root_dir, self.df.loc[idx, "File_name"])

        if self.df.loc[idx, "is_double_page"]:
            img = self.imread(img_name)
            width = img.shape[1]
            min_left = (self.crop.width + 1) // 2
            max_left = width - 1 - self.crop.width // 2

            # Cropping at random in the image
            center = self.generator.integers(min_left, max_left + 1)
            img = self.crop(img, center)
        else:
            # We create an image [File_name, pairing]
            # So that if img_X is paired with img_X+1
            # and img_X+1 is paired with img_X
            # we get two different double images.

            img_1_name = os.path.join(self.root_dir, self.df.loc[idx, "pairing"])

            img_0 = self.imread(img_name)
            img_1 = self.imread(img_1_name)

            img = np.hstack((img_0, img_1))
            # Croping using the border border as center
            img = self.crop(img, img_0.shape[1] - 1)

        sample = {"image": img, "label": int(self.df.loc[idx, "is_double_page"])}

        return sample

    def imread(self, img_name):
        """
        Method for reading an image and converting it to grayscale with a given height
        Arguments:
            img_name (str): full path to image file
        Returns a 2D np.array
        """
        # Cannot do 'as_gray=True' as there is no way of knowing
        # if the image has been normalized or not with only skimage
        img = io.imread(img_name)

        if len(img.shape) > 2:
            return self.rescale(color.rgb2gray(img).astype(np.float32))
        return self.rescale(img.astype(np.float32) / 255.0)

    def __create_labels(self):
        """
        Generating labels for the dataset.
        Adds "is_double_page" and "pairing" columns to self DataFrame
        Updated DataFrame is something like
                File_name 	     Manga 	  Volume   is_double_page 	    pairing
        0 	Example_T1_001.jpg 	Example 	T1 	         0 	       Example_T1_002.jpg
        1 	Example_T1_002.jpg 	Example 	T1 	         1
        2 	Example_T1_003.jpg 	Example 	T1 	         1
        3 	Example_T1_004.jpg 	Example 	T1 	         0 	       Example_T1_005.jpg
        4 	Example_T1_005.jpg 	Example 	T1 	         0 	       Example_T1_004.jpg
        """
        # Generating labels, if 1 then it is a double page
        # if 0 then the file is not a double page and should be
        # associated with another page
        self.df["is_double_page"] = self.generator.integers(
            2, size=len(self.df)
        ).astype(dtype=np.bool_)

        # Generating other pair value when sample is not a double page
        self.df["pairing"] = ""
        for manga in pd.unique(self.df["Manga"]):
            sub_df = self.df[self.df["Manga"] == manga]
            for volume in pd.unique(sub_df["Volume"]):
                sub_sub_df = sub_df[sub_df["Volume"] == volume]
                for idx in sub_sub_df[~sub_sub_df["is_double_page"]].index:
                    consecutive = []
                    if idx - 1 in sub_sub_df.index:
                        consecutive += [idx - 1]
                    if idx + 1 in sub_sub_df.index:
                        consecutive += [idx + 1]
                    elif not consecutive:
                        consecutive = [idx]
                    self.df.loc[idx, "pairing"] = self.df.loc[
                        self.generator.choice(consecutive, 1)[0], "File_name"
                    ]
