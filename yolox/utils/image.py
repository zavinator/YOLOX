import cv2
import numpy as np

def read(path):
    """Read an image; keep grayscale if possible, otherwise return BGR. Enforce uint8/uint16 dtype.

    Args:
        path (str): The path to the image file.

    Raises:
        ValueError: If the image cannot be read, dtype is not uint8/uint16, or channel layout is unsupported.

    Returns:
        np.ndarray: Grayscale (H,W) or BGR (H,W,3) image with dtype uint8 or uint16.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # preserve depth and channels
    if img is None:
        raise ValueError(f"Image at {path} could not be read.")

    if img.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"Unsupported image dtype: {img.dtype}. Expected uint8 or uint16.")

    # Grayscale
    if img.ndim == 2:
        return img

    if img.ndim == 3:
        c = img.shape[2]
        if c == 3:
            # BGR
            return img
        if c == 4:
            # BGRA -> BGR
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if c == 1:
            # Sometimes grayscale may come as (H,W,1); squeeze to (H,W)
            return img.squeeze(axis=2)

    raise ValueError(f"Unsupported image shape: {img.shape}. Expected grayscale or BGR/BGRA.")
