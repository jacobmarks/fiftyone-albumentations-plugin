import hashlib
from PIL import Image
import numpy as np
import random
import re
import torch

import fiftyone as fo
from fiftyone.operators import types

## for camelCase to snake_case conversion
pattern = re.compile(r"(?<!^)(?=[A-Z])")


def _camel_to_snake(name):
    return pattern.sub("_", name).lower()


def _count_leading_spaces(line):
    return len(line) - len(line.lstrip())


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


def _execution_mode(ctx, inputs):
    delegate = ctx.params.get("delegate", False)

    if delegate:
        description = "Uncheck this box to execute the operation immediately"
    else:
        description = "Check this box to delegate execution of this task"

    inputs.bool(
        "delegate",
        default=False,
        required=True,
        label="Delegate execution?",
        description=description,
        view=types.CheckboxView(),
    )

    if delegate:
        inputs.view(
            "notice",
            types.Notice(
                label=(
                    "You've chosen delegated execution. Note that you must "
                    "have a delegated operation service running in order for "
                    "this task to be processed. See "
                    "https://docs.voxel51.com/plugins/index.html#operators "
                    "for more information"
                )
            ),
        )


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
        tensor_mask_4d = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
        prediction = torch.nn.functional.interpolate(
            tensor_mask_4d,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        mask = torch.clamp(prediction, min=0, max=1).squeeze().detach().numpy()

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


def _get_target_view(ctx, target):
    if target == "SELECTED_SAMPLES":
        return ctx.view.select(ctx.selected)

    if target == "DATASET":
        return ctx.dataset

    return ctx.view


def _join_lines_with_indentation(lines):
    """
    Joins lines that are indented (relative to previous line)
    with the previous line.
    """
    joined_lines = []
    for line in lines:
        if not line.strip():
            continue  # Skip empty lines

        if joined_lines:
            if _count_leading_spaces(line) > _count_leading_spaces(joined_lines[-1]):
                joined_lines[-1] += " " + line.strip()
            else:
                joined_lines.append(line)
        else:
            joined_lines.append(line)

    return joined_lines


def _list_target_views(ctx, inputs):
    has_view = ctx.view != ctx.dataset.view()
    has_selected = bool(ctx.selected)
    default_target = "DATASET"
    if has_view or has_selected:
        target_choices = types.RadioGroup()
        target_choices.add_choice(
            "DATASET",
            label="Entire dataset",
            description="Run model on the entire dataset",
        )

        if has_view:
            target_choices.add_choice(
                "CURRENT_VIEW",
                label="Current view",
                description="Run model on the current view",
            )
            default_target = "CURRENT_VIEW"

        if has_selected:
            target_choices.add_choice(
                "SELECTED_SAMPLES",
                label="Selected samples",
                description="Run model on the selected samples",
            )
            default_target = "SELECTED_SAMPLES"

        inputs.enum(
            "target",
            target_choices.values(),
            default=default_target,
            view=target_choices,
        )
    else:
        ctx.params["target"] = "DATASET"
