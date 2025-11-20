import cv2
import numpy as np
import os

from pathlib import Path


class Preprocessor():
    COLOR_MAP = {(np.uint8(0), np.uint8(0), np.uint8(0)): 0,  # Background
                 (np.uint8(0), np.uint8(128), np.uint8(0)): 1,  # Tree
                 (np.uint8(64), np.uint8(0), np.uint8(128)): 2,  # Moving car
                 (np.uint8(64), np.uint8(64), np.uint8(0)): 3,  # Human
                 (np.uint8(128), np.uint8(0), np.uint8(0)): 4,  # Building
                 (np.uint8(128), np.uint8(64), np.uint8(128)): 5,  # Road
                 (np.uint8(128), np.uint8(128), np.uint8(0)): 6,  # Low vegetation
                 (np.uint8(192), np.uint8(0), np.uint8(192)): 7}  # Static car

    def __init__(self,
                 image_path: Path,
                 normalize_flag: bool = False,
                 mean: tuple = None,
                 std: tuple = None):
        self.image_path = image_path
        self.normalize_flag = normalize_flag
        self.mean = mean
        self.std = std

    def preprocess_image(self):
        image = self._read_image(self.image_path)
        if self.normalize_flag:
            image = self._normalize(image)
        return image

    def preprocess_mask(self):
        mask_path = self.image_path.replace('Images', 'Labels')
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Label file not found for: {mask_path}")
        mask = self._read_image(mask_path)
        mask = self._rgb_to_gray(mask)
        return mask
    
    @staticmethod
    def _read_image(file_path):
        image = cv2.imread(file_path)  # Loads as BGR, numpy array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = image.astype(float)
        return image

    def _normalize(self, image):
        """
        Standard normalization is applied using the formula:
        img = (img - mean * max_pixel_value) / (std * max_pixel_value).
        """
        image = image/255.
        mean = np.array(self.mean, dtype=np.float32)
        std = np.array(self.std, dtype=np.float32)
        image = (image - mean) / std
        return image

    def _rgb_to_gray(cls, rgb_label_array):
        """
        Converts a 3-channel RGB label image to a single-channel class ID mask.

        Args:
            rgb_label_image (np.ndarray): The 3-channel (H, W, 3) RGB label image.

        Returns:
            np.ndarray: A single-channel (H, W) mask with integer class IDs.
        """
        rgb_pixels_tuples = [tuple(p) for p in rgb_label_array.reshape(-1, 3)]
        gray_mask = [cls.COLOR_MAP[rgb_pixels_tuples[i]] for i in range(len(rgb_pixels_tuples))]
        gray_mask = np.array(gray_mask, dtype=np.uint8).reshape(rgb_label_array.shape[0],
                                                                rgb_label_array.shape[1])
        return gray_mask
