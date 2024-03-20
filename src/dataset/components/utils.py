# utils.py content

import os

import requests
from tqdm import tqdm
from zenodo_get import zenodo_get

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

import numpy as np
from spectral import open_image

from src.utils.utils import open_file

from .datasets_config import DATASETS_CONFIG


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def download_from_zenodo(doi, target_folder):
    """
    Downloads files from a given Zenodo DOI to a specified target
    folder using the zenodo_get package.

    Args:
    - doi (list): The DOI(s) of the Zenodo record(s) to download, as a list.
    - target_folder (str): The directory path where files should be saved.
    """

    # # Ensure the target directory exists
    # if not os.path.exists(target_folder):
    #     os.makedirs(target_folder)

    # Prepare the arguments as if they were command-line arguments
    args = ['--output-dir', target_folder] + [doi]
    # print(args)

    # Use the zenodo_get package to download files, passing the arguments as a single list
    zenodo_get(args)


def download_dataset(dataset_name, target_folder="./",
                     datasets=DATASETS_CONFIG):
    if dataset_name not in datasets.keys():
        raise ValueError(f"{dataset_name} dataset is unknown.")

    dataset_info = datasets[dataset_name]
    folder = os.path.join(target_folder, dataset_info.get(
        "folder", dataset_name + "/"))

    if dataset_info.get("download", True):
        if not os.path.isdir(folder):
            os.makedirs(folder)
        for url in dataset_info["urls"]:
            filename = url.split("/")[-1]
            filepath = os.path.join(folder, filename)
            if not os.path.exists(filepath):
                with TqdmUpTo(unit="B", unit_scale=True, miniters=1, desc=f"Downloading {filename}") as t:
                    urlretrieve(url, filename=filepath, reporthook=t.update_to)
    else:
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"The directory for \
                {dataset_name} does not exist, and the dataset is not set to be downloadable.")


def load_dataset(dataset_name, target_folder="./", datasets=DATASETS_CONFIG):
    palette = None
    if dataset_name not in datasets.keys():
        raise ValueError(f"{dataset_name} dataset is unknown.")

    dataset_info = datasets[dataset_name]
    folder = os.path.join(target_folder, dataset_info.get(
        "folder", dataset_name + "/"))

    if dataset_name == "PaviaC":
        # Load the image
        img = open_file(folder + "Pavia.mat")["pavia"]

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + "Pavia_gt.mat")["pavia_gt"]

        label_values = [
            "Undefined",
            "Water",
            "Trees",
            "Asphalt",
            "Self-Blocking Bricks",
            "Bitumen",
            "Tiles",
            "Shadows",
            "Meadows",
            "Bare Soil",
        ]

        ignored_labels = [0]

    elif dataset_name == "PaviaU":
        # Load the image
        img = open_file(folder + "PaviaU.mat")["paviaU"]

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + "PaviaU_gt.mat")["paviaU_gt"]

        label_values = [
            "Undefined",
            "Asphalt",
            "Meadows",
            "Gravel",
            "Trees",
            "Painted metal sheets",
            "Bare Soil",
            "Bitumen",
            "Self-Blocking Bricks",
            "Shadows",
        ]

        ignored_labels = [0]

    elif dataset_name == "Salinas":
        img = open_file(folder + "Salinas_corrected.mat")["salinas_corrected"]

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + "Salinas_gt.mat")["salinas_gt"]

        label_values = [
            "Undefined",
            "Brocoli_green_weeds_1",
            "Brocoli_green_weeds_2",
            "Fallow",
            "Fallow_rough_plow",
            "Fallow_smooth",
            "Stubble",
            "Celery",
            "Grapes_untrained",
            "Soil_vinyard_develop",
            "Corn_senesced_green_weeds",
            "Lettuce_romaine_4wk",
            "Lettuce_romaine_5wk",
            "Lettuce_romaine_6wk",
            "Lettuce_romaine_7wk",
            "Vinyard_untrained",
            "Vinyard_vertical_trellis",
        ]

        ignored_labels = [0]

    elif dataset_name == "IndianPines":
        # Load the image
        img = open_file(folder + "Indian_pines_corrected.mat")
        img = img["indian_pines_corrected"]

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + "Indian_pines_gt.mat")["indian_pines_gt"]
        label_values = [
            "Undefined",
            "Alfalfa",
            "Corn-notill",
            "Corn-mintill",
            "Corn",
            "Grass-pasture",
            "Grass-trees",
            "Grass-pasture-mowed",
            "Hay-windrowed",
            "Oats",
            "Soybean-notill",
            "Soybean-mintill",
            "Soybean-clean",
            "Wheat",
            "Woods",
            "Buildings-Grass-Trees-Drives",
            "Stone-Steel-Towers",
        ]

        ignored_labels = [0]

    elif dataset_name == "Botswana":
        # Load the image
        img = open_file(folder + "Botswana.mat")["Botswana"]

        rgb_bands = (75, 33, 15)

        gt = open_file(folder + "Botswana_gt.mat")["Botswana_gt"]
        label_values = [
            "Undefined",
            "Water",
            "Hippo grass",
            "Floodplain grasses 1",
            "Floodplain grasses 2",
            "Reeds",
            "Riparian",
            "Firescar",
            "Island interior",
            "Acacia woodlands",
            "Acacia shrublands",
            "Acacia grasslands",
            "Short mopane",
            "Mixed mopane",
            "Exposed soils",
        ]

        ignored_labels = [0]

    elif dataset_name == "KSC":
        # Load the image
        img = open_file(folder + "KSC.mat")["KSC"]

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + "KSC_gt.mat")["KSC_gt"]
        label_values = [
            "Undefined",
            "Scrub",
            "Willow swamp",
            "Cabbage palm hammock",
            "Cabbage palm/oak hammock",
            "Slash pine",
            "Oak/broadleaf hammock",
            "Hardwood swamp",
            "Graminoid marsh",
            "Spartina marsh",
            "Cattail marsh",
            "Salt marsh",
            "Mud flats",
            "Wate",
        ]

        ignored_labels = [0]
    else:
        # Custom dataset
        (
            img,
            gt,
            rgb_bands,
            ignored_labels,
            label_values,
            palette,
        ) = CUSTOM_DATASETS_CONFIG[
            dataset_name
        ]["loader"](folder)

    # Filter NaN out
    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        print(
            "Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled."
        )
    img[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)

    ignored_labels = list(set(ignored_labels))
    # Normalization
    img = np.asarray(img, dtype="float32")
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img, gt, label_values, ignored_labels, rgb_bands, palette


# def build_dataset(mat, gt, ignored_labels=None):
#     """
#     Create a list of training samples based on an image and a ground truth mask.

#     This function extracts the spectrums from a 3D hyperspectral image matrix and associates them with labels
#     based on a corresponding 2D ground truth mask. It allows for certain labels to be ignored, effectively
#     filtering out unwanted training samples.

#     Args:
#         mat (np.ndarray): A 3D hyperspectral image matrix from which to extract the spectrums.
#         gt (np.ndarray): A 2D ground truth mask where each pixel value corresponds to the label of the spectrum at that location.
#         ignored_labels (list, optional): A list of label values to ignore. For example, 0 could be used to ignore unlabeled pixels.
#             Defaults to None, which means no labels are ignored.

#     Returns:
#         tuple: A tuple containing two elements. The first element is a numpy array of the extracted samples (spectrums),
#         and the second element is a numpy array of the corresponding labels for each sample.
#     """
#     if ignored_labels is None:
#         ignored_labels = []  # Initialize as an empty list if None

#     samples = []
#     labels = []
#     # Check that image and ground truth have the same 2D dimensions
#     assert mat.shape[:2] == gt.shape[:2], "Image and ground truth dimensions do not match"

#     for label in np.unique(gt):
#         if label not in ignored_labels:
#             indices = np.nonzero(gt == label)
#             samples.extend(mat[indices])
#             labels.extend(len(indices[0]) * [label])                                                                                                              `

#     return np.asarray(samples), np.asarray(labels)

# def get_dataset(dataset_name, target_folder="./", datasets=DATASETS_CONFIG):
#     """Gets the dataset specified by name and return the related components.

#     Args:
#         dataset_name: string with the name of the dataset
#         target_folder (optional): folder to store the datasets, defaults to ./
#         datasets (optional): dataset configuration dictionary, defaults to prebuilt one
#     Returns:
#         img: 3D hyperspectral image (WxHxB)
#         gt: 2D int array of labels
#         label_values: list of class names
#         ignored_labels: list of int classes to ignore
#         rgb_bands: int tuple that correspond to red, green and blue bands
#     """
#     palette = None

#     if dataset_name not in datasets.keys():
#         raise ValueError(f"{dataset_name} dataset is unknown.")

#     dataset = datasets[dataset_name]

#     folder = target_folder + \
#         datasets[dataset_name].get("folder", dataset_name + "/")
#     if dataset.get("download", True):
#         # Download the dataset if is not present
#         if not os.path.isdir(folder):
#             os.makedirs(folder)
#         for url in datasets[dataset_name]["urls"]:
#             # download the files
#             filename = url.split("/")[-1]
#             if not os.path.exists(folder + filename):
#                 with TqdmUpTo(
#                     unit="B",
#                     unit_scale=True,
#                     miniters=1,
#                     desc=f"Downloading {filename}",
#                 ) as t:
#                     urlretrieve(url, filename=folder + filename,
#                                 reporthook=t.update_to)
#     elif not os.path.isdir(folder):
#         print(f"WARNING: {dataset_name} is not downloadable.")
