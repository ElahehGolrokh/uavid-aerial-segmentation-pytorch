import cv2
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from albumentations import (HorizontalFlip, ShiftScaleRotate, Resize, Compose, ToTensorV2)
# from omegaconf import Omegaconf

from pathlib import Path
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from .preprocessing import Preprocessor


def get_transforms(phase, height, width):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                HorizontalFlip(),
                ShiftScaleRotate(
                    shift_limit=0,  # no resizing
                    scale_limit=0.1,
                    rotate_limit=10, # rotate
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT
                ),
#                 GaussNoise(),
            ]
        )
    list_transforms.extend(
        [
            Resize(height, width),
            ToTensorV2()
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms


class SemanticSegmentationDataset(Dataset):
    CLASSES = [
        "Background",
        "Tree",
        "Moving car",
        "Human",
        "Building",
        "Road",
        "Low vegetation",
        "Static car",
    ]
    def __init__(self,
                 data_paths: Path,
                 phase: str,
                 height=256,
                 width=256,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        self.data_paths = data_paths
        self.mean = mean
        self.std = std
        self.transforms = get_transforms(phase, height, width)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = self.data_paths[idx]
        preprocessor = Preprocessor(image_path=image_path,
                                    normalize_flag=True,
                                    mean=self.mean,
                                    std=self.std)
        image = preprocessor.preprocess_image()
        mask = preprocessor.preprocess_mask()
        
        # image = cv2.imread(img_path)  # Loads as BGR, numpy array
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB
        # image = image.astype(float)
        # tensor_image = transformer(image)

        # label_path = img_path.replace('Images', 'Labels')
        # if not os.path.exists(label_path):
        #     raise FileNotFoundError(f"Label file not found for image: {img_path}")
        # bgr_label_array = cv2.imread(label_path)
        # rgb_label_array = cv2.cvtColor(bgr_label_array, cv2.COLOR_BGR2RGB)
        # mask = rgb_to_mask(rgb_label_array)
        # Removed: mask = mask.astype(float) to keep mask as integer type

        augmented = self.transforms(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask'].long()  # Explicitly convert mask to LongTensor for CrossEntropyLoss

        return image, mask


class DataGenerator:
    def __init__(self,
                 dir: Path,
                 phase: str,
                 batch_size: int,
                 shuffle: bool):
        self.dir = dir
        self.phase = phase
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def load_data(self):
        paths = self._create_data_paths()
        dataset = SemanticSegmentationDataset(paths, phase=self.phase)
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=self.shuffle,
                                num_workers=os.cpu_count())
        return dataloader

    def _create_data_paths(self):
        data_paths = []
        for seq in os.listdir(self.dir):
            seq_dir_images = os.path.join(self.dir, seq, 'Images')
            for files in os.listdir(seq_dir_images):
                if files.endswith('.jpg') or files.endswith('.png'):
                    file_path = os.path.join(seq_dir_images, files)
                    data_paths.append(file_path)
        return data_paths
