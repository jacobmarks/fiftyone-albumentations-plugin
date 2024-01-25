import hashlib
from PIL import Image
import numpy as np
import random
import re

import fiftyone as fo

## for camelCase to snake_case conversion
pattern = re.compile(r"(?<!^)(?=[A-Z])")

def _camel_to_snake(name):
    return pattern.sub("_", name).lower()


def _create_hash():
    """Get a hash of the given sample.

    Args:
        sample: a fiftyone.core.sample.Sample
        label_fields: a list of label field names to collect bounding boxes from

    Returns:
        a hash of the given sample

    """
    randint = random.randint(0, 100000000)
    hash = hashlib.sha256(str(randint).encode("utf-8")).hexdigest()[:10]
    return hash



def _get_image_size(sample):
    if sample.metadata is not None and sample.metadata.width is not None:
        return (sample.metadata.width, sample.metadata.height)
    else:
        return Image.open(sample.filepath).size[::-1]
    

def _enforce_mask_size(mask, width, height):
    """Enforce the given mask to be of the given size.

    Args:
        mask: a numpy array representing a mask
        width: the desired width of the mask
        height: the desired height of the mask

    Returns:
        a numpy array representing the mask of the given size
    """
    if mask.shape != (height, width):
        mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize((width, height)))
    return mask


def _convert_bbox_to_albumentations(bbox):
    """Convert a FiftyOne bounding box to an Albumentations bounding box."""
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


def _convert_bbox_from_albumentations(bbox):
    """Convert an Albumentations bounding box to a FiftyOne bounding box."""
    return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]


def _convert_keypoint_to_albumentations(keypoint, frame_size):
    """Convert FiftyOne keypoints to an Albumentations keypoints."""
    return [keypoint[0] * frame_size[0], keypoint[1] * frame_size[1]]


def _convert_keypoint_from_albumentations(keypoint, frame_size):
    """Convert Albumentations keypoints to a FiftyOne keypoints."""
    return [keypoint[0] / frame_size[0], keypoint[1] / frame_size[1]]


def _get_label_fields(sample):
    """Get the names of the fields containing labels for the given sample."""
    return [
        field_name
        for field_name in sample.field_names
        if isinstance(sample[field_name], fo.Label)
    ]


def _get_detections_fields(sample, label_fields):
    """Get the names of the fields containing detections for the given sample."""
    return [
        field_name
        for field_name in label_fields
        if isinstance(sample[field_name], fo.Detections)
    ]


def _get_keypoints_fields(sample, label_fields):
    """Get the names of the fields containing keypoints for the given sample."""
    return [
        field_name
        for field_name in label_fields
        if isinstance(sample[field_name], fo.Keypoints)
    ]


def _get_mask_fields(sample, label_fields):
    """Get the names of the fields containing segmentations or heatmaps for the given sample."""
    return [
        field_name
        for field_name in label_fields
        if field_name in sample
        and (
            isinstance(sample[field_name], fo.Segmentation)
            or isinstance(sample[field_name], fo.Heatmap)
        )
    ]