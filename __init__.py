"""Albumentations image augmentation plugin.
"""

import hashlib
import random
import albumentations as A
import cv2
from PIL import Image
import numpy as np
import json
from bson import json_util
import os

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types

LAST_ALBUMENTATIONS_RUN_KEY = "_last_albumentations_run"

def serialize_view(view):
    return json.loads(json_util.dumps(view._serialize()))

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


def _collect_bboxes(sample, label_fields):
    """Collect all bounding boxes from the given sample.

    Args:
        sample: a fiftyone.core.sample.Sample
        label_fields: a list of label field names to collect bounding boxes from

    Returns:
        a list of [x, y, width, height, id] lists representing bounding boxes
        for the given sample, where id is the id of the detection in the sample
    """
    boxes_list = []
    for field in label_fields:
        detections = sample[field].detections
        for det in detections:
            bbox = _convert_bbox_to_albumentations(det.bounding_box)
            boxes_list.append(bbox + [str(det._id)])

    return boxes_list


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


def _collect_masks(sample, mask_fields, detections_fields):
    """Collect all masks from the given sample.

    Args:
        sample: a fiftyone.core.sample.Sample
        mask_fields: a list of label field names to collect full masks from
        detections_fields: a list of label field names to collect instance masks from

    Returns:
        a dict of {id: mask} representing masks
        for the given sample, where id is the id of the label in the sample
    """

    width, height = _get_image_size(sample)

    masks_dict = {}

    ## Add semantic segmentation masks and heatmaps to the masks dict
    for field in mask_fields:
        mask_label = sample[field]
        if isinstance(mask_label, fo.Segmentation):
            if mask_label.mask_path is not None:
                mask = np.array(Image.open(mask_label.mask_path))
            else:
                mask = mask_label.mask
        elif isinstance(mask_label, fo.Heatmap):
            if mask_label.map_path is not None:
                mask = np.array(Image.open(mask_label.map_path))
            else:
                mask = mask_label.map

        mask = _enforce_mask_size(mask, width, height)
        masks_dict[str(mask_label._id)] = mask

    ## Add instance masks to the masks dict
    for field in detections_fields:
        detections = sample[field].detections
        for det in detections:
            mask = det.mask
            if mask is None:
                continue
            mask = _enforce_mask_size(mask, width, height)
            masks_dict[str(det._id)] = mask

    return masks_dict


# def _crop_instance_mask(mask, bbox, frame_size):
#     det = fo.Detection(mask=mask, bounding_box=bbox)
#     seg = det.to_segmentation(frame_size=frame_size)
#     # seg = fo.Segmentation(mask=mask)
#     return seg.to_detections().detections[0].mask


def _collect_keypoints(sample, keypoints_fields):

    width, height = _get_image_size(sample)

    keypoints_dict = {}
    for field in keypoints_fields:
        keypoints = sample[field].keypoints
        for kp in keypoints:
            kp_id = str(kp._id)
            kp_points = kp.points
            for i, point in enumerate(kp_points):
                kpp_id = f"{kp_id}_{i}"
                keypoints_dict[kpp_id] = _convert_keypoint_to_albumentations(
                    point, [width, height]
                )

    return keypoints_dict


def _update_detection_field(
    original_sample, new_sample, detection_field, transformed_boxes
):
    detections = original_sample[detection_field].detections
    new_detections = []
    for det in detections:
        if str(det._id) in transformed_boxes:
            new_det = det.copy()
            new_det.bounding_box = transformed_boxes[str(det._id)]

            ## Remove instance masks from the new detections
            ##  Until we can figure out how to crop them properly
            if new_det.mask is not None:
                new_det.mask = None
            ## Add instance masks to the new detections
            # if str(det._id) in transformed_masks_dict:
            #     full_mask = transformed_masks_dict[str(det._id)]
            #     instance_mask = _crop_instance_mask(
            #         full_mask, new_det.bounding_box, (image.shape[1], image.shape[0])
            #         )
            #     new_det.mask = instance_mask

            new_detections.append(new_det)
    new_sample[detection_field] = fo.Detections(detections=new_detections)
    return new_sample


def _update_mask_field(original_sample, new_sample, mask_field, transformed_masks_dict):
    mid = str(original_sample[mask_field]._id)
    new_mask_label = original_sample[mask_field].copy()
    if isinstance(new_mask_label, fo.Segmentation):
        if new_mask_label.mask_path is not None:
            new_mask_label["mask_path"] = None
        else:
            new_mask_label["mask"] = transformed_masks_dict[mid]
    elif isinstance(new_mask_label, fo.Heatmap):
        if new_mask_label.map_path is not None:
            new_mask_label["map_path"] = None
        else:
            new_mask_label["map"] = transformed_masks_dict[mid]
    new_sample[mask_field] = new_mask_label
    return new_sample


def _update_keypoints_field(
    original_sample, new_sample, keypoints_field, transformed_keypoints_dict
):
    new_kps = []
    for kp in original_sample[keypoints_field].keypoints:
        kp_id = str(kp._id)
        new_kp = kp.copy()
        for i in range(len(kp.points)):
            kpp_id = f"{kp_id}_{i}"
            if kpp_id in transformed_keypoints_dict:
                new_kp.points[i] = transformed_keypoints_dict[kpp_id]
            else:
                new_kp.points[i] = [float("nan"), float("nan")]
        new_kps.append(new_kp)
    kps = fo.Keypoints(keypoints=new_kps)
    new_sample[keypoints_field] = kps
    return new_sample


def transform_sample(sample, transform, label_fields=False, new_filepath=None):
    """Apply an Albumentations transform to the image
    and all label fields listed for the sample.

    Args:
        sample: a fiftyone.core.sample.Sample
        transform: an Albumentations transform
        label_fields (False): a field or list of field names to transform.
            If label_fields=False, no label fields are transformed.
            If label_fields=True, all label fields are transformed.

    Currently does not handle:
    - instance segmentation masks.
    """
    if new_filepath is None:
        hash = _create_hash()
        new_filepath = f"/tmp/{hash}.jpg"

    if not label_fields:
        label_fields = []
    if label_fields is True:
        label_fields = _get_label_fields(sample)

    image = cv2.cvtColor(cv2.imread(sample.filepath), cv2.COLOR_BGR2RGB)

    detection_fields = _get_detections_fields(sample, label_fields)
    boxes_list = _collect_bboxes(sample, detection_fields)

    mask_fields = _get_mask_fields(sample, label_fields)

    masks_dict = _collect_masks(sample, mask_fields, detection_fields)
    masks = list(masks_dict.values())

    keypoint_fields = _get_keypoints_fields(sample, label_fields)
    keypoints_dict = _collect_keypoints(sample, keypoint_fields)
    keypoints = list(keypoints_dict.values())
    keypoint_labels = list(keypoints_dict.keys())

    has_boxes = len(boxes_list) > 0
    has_masks = len(masks) > 0
    has_keypoints = len(keypoints) > 0

    kwargs = {"image": image}
    if has_boxes:
        kwargs["bboxes"] = boxes_list
    if has_masks:
        kwargs["masks"] = masks
    if has_keypoints:
        kwargs["keypoints"] = keypoints
        kwargs["keypoint_labels"] = keypoint_labels
    else:
        kwargs["keypoints"] = []
        kwargs["keypoint_labels"] = []

    transformed = transform(**kwargs)

    transformed_image = transformed["image"]

    if has_boxes:
        transformed_boxes = transformed["bboxes"]
        transformed_boxes = {
            bbox[-1]: _convert_bbox_from_albumentations(bbox[:-1])
            for bbox in transformed_boxes
        }

    if has_masks:
        transformed_masks = transformed["masks"]
        transformed_masks_dict = {
            id: mask for id, mask in zip(masks_dict.keys(), transformed_masks)
        }

    if has_keypoints:
        transformed_keypoints = transformed["keypoints"]
        transformed_keypoint_labels = transformed["keypoint_labels"]
        transformed_keypoints_dict = {
            kp_id: _convert_keypoint_from_albumentations(
                kp, [image.shape[1], image.shape[0]]
            )
            for kp_id, kp in zip(transformed_keypoint_labels, transformed_keypoints)
        }

    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)  
    cv2.imwrite(new_filepath, transformed_image)
    new_sample = fo.Sample(filepath=new_filepath)

    if has_boxes:
        for detection_field in detection_fields:
            new_sample = _update_detection_field(
                sample, new_sample, detection_field, transformed_boxes
            )

    if has_masks:
        for mask_field in mask_fields:
            new_sample = _update_mask_field(
                sample, new_sample, mask_field, transformed_masks_dict
            )

    if has_keypoints:
        for keypoint_field in keypoint_fields:
            new_sample = _update_keypoints_field(
                sample, new_sample, keypoint_field, transformed_keypoints_dict
            )

    sample._dataset.add_sample(new_sample)
    return new_sample.id


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


def _get_target_view(ctx, target):
    if target == "SELECTED_SAMPLES":
        return ctx.view.select(ctx.selected)

    if target == "DATASET":
        return ctx.dataset

    return ctx.view


######## Transformations ########


### Affine


def _affine_input(ctx, inputs):
    ## interpolation not supported yet
    ## translate_px not supported yet
    ## mask_interpolation not supported yet
    ## cval not supported yet
    ## cval_mask not supported yet
    ## mode not supported yet
    ## fit_output not supported yet
    ## keep_ratio not supported yet
    ## rotate_method not supported yet

    inputs.float(
        "affine__scale_min",
        label="Scale min",
        description="Minimum scale factor",
        required=True,
        default=0.75,
    )
    inputs.float(
        "affine__scale_max",
        label="Scale max",
        description="Maximum scale factor",
        required=True,
        default=1.25,
    )
    inputs.float(
        "affine__translate_percent_min",
        label="Translate percent min",
        description="Minimum fraction of image size by which to translate the image",
        required=True,
        default=0.2,
    )
    inputs.float(
        "affine__translate_percent_max",
        label="Translate percent max",
        description="Maximum fraction of image size by which to translate the image",
        required=True,
        default=0.8,
    )
    inputs.float(
        "affine__rotate",
        label="Rotate",
        description="Angle in degrees. Positive values mean clockwise rotation.",
        required=True,
        default=0,
    )
    inputs.float(
        "affine__shear",
        label="Shear",
        description="Shear angle in degrees.",
        required=True,
        default=0,
    )
    inputs.float(
        "affine__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=1.0,
    )


def _affine_transform(ctx):
    scale_min = ctx.params.get("affine__scale_min", None)
    scale_max = ctx.params.get("affine__scale_max", None)
    translate_percent_min = ctx.params.get("affine__translate_percent_min", None)
    translate_percent_max = ctx.params.get("affine__translate_percent_max", None)
    rotate = ctx.params.get("affine__rotate", None)
    shear = ctx.params.get("affine__shear", None)
    p = ctx.params.get("affine__p", None)
    return A.Affine(
        scale=(scale_min, scale_max),
        translate_percent=(translate_percent_min, translate_percent_max),
        rotate=rotate,
        shear=shear,
        p=p,
    )


### CenterCrop


def _center_crop_input(ctx, inputs):
    inputs.int(
        "center_crop__height",
        label="Height",
        description="The height of the crop in pixels",
        required=True,
    )
    inputs.int(
        "center_crop__width",
        label="Width",
        description="The width of the crop in pixels",
        required=True,
    )

    inputs.float(
        "center_crop__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=1.0,
    )


def _center_crop_transform(ctx):
    height = ctx.params.get("center_crop__height", None)
    width = ctx.params.get("center_crop__width", None)
    p = ctx.params.get("center_crop__p", None)

    return A.CenterCrop(height=height, width=width, p=p)


### ChannelDropout


def _channel_dropout_input(ctx, inputs):
    inputs.float(
        "channel_dropout__channel_drop_range_min",
        label="Mininum for channel drop range",
        description="Minimum of range from which the number of channels to drop will be sampled",
        required=True,
        default=1,
    )
    inputs.float(
        "channel_dropout__channel_drop_range_max",
        label="Maximum for channel drop range",
        description="Maximum of range from which the number of channels to drop will be sampled",
        required=True,
        default=1,
    )
    inputs.float(
        "channel_dropout__fill_value",
        label="Fill value",
        description="Fill value used to drop channels",
        required=True,
        default=0,
    )
    inputs.float(
        "channel_dropout__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=0.5,
    )


def _channel_dropout_transform(ctx):
    channel_drop_range_min = ctx.params.get(
        "channel_dropout__channel_drop_range_min", None
    )
    channel_drop_range_max = ctx.params.get(
        "channel_dropout__channel_drop_range_max", None
    )
    fill_value = ctx.params.get("channel_dropout__fill_value", None)
    p = ctx.params.get("channel_dropout__p", None)

    return A.ChannelDropout(
        channel_drop_range=(channel_drop_range_min, channel_drop_range_max),
        fill_value=fill_value,
        p=p,
    )


### ChannelShuffle


def _channel_shuffle_input(ctx, inputs):
    inputs.float(
        "channel_shuffle__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=0.5,
    )


def _channel_shuffle_transform(ctx):
    p = ctx.params.get("channel_shuffle__p", None)
    return A.ChannelShuffle(p=p)



### CLAHE


def _clahe_input(ctx, inputs):
    ## [float, float] for clip_limit not supported yet
    ## [int, int] for tile_grid_size not supported yet
    inputs.int(
        "clahe__clip_limit",
        label="Clip limit",
        description="Threshold for contrast limiting",
        required=True,
        default=4,
    )
    inputs.int(
        "clahe__tile_grid_size",
        label="Tile grid size",
        description="Size of grid for histogram equalization. Input image will be divided into equally sized rectangular tiles",
        required=True,
        default=8,
    )
    inputs.float(
        "clahe__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=0.5,
    )


def _clahe_transform(ctx):
    clip_limit = ctx.params.get("clahe__clip_limit", None)
    tile_grid_size = ctx.params.get("clahe__tile_grid_size", None)
    tile_grid_size = (tile_grid_size, tile_grid_size)
    p = ctx.params.get("clahe__p", None)
    return A.CLAHE(clip_limit=clip_limit, tile_grid_size=tile_grid_size, p=p)


### ColorJitter


def _color_jitter_input(ctx, inputs):
    inputs.float(
        "color_jitter__brightness",
        label="Brightness",
        description="Factor range for changing brightness.",
        required=True,
        default=0.2,
    )
    inputs.float(
        "color_jitter__contrast",
        label="Contrast",
        description="Factor range for changing contrast.",
        required=True,
        default=0.2,
    )
    inputs.float(
        "color_jitter__saturation",
        label="Saturation",
        description="Factor range for changing saturation.",
        required=True,
        default=0.2,
    )
    inputs.float(
        "color_jitter__hue",
        label="Hue",
        description="Factor range for changing hue.",
        required=True,
        default=0.2,
    )
    inputs.float(
        "color_jitter__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=0.5,
    )


def _color_jitter_transform(ctx):
    brightness = ctx.params.get("color_jitter__brightness", None)
    contrast = ctx.params.get("color_jitter__contrast", None)
    saturation = ctx.params.get("color_jitter__saturation", None)
    hue = ctx.params.get("color_jitter__hue", None)
    p = ctx.params.get("color_jitter__p", None)
    return A.ColorJitter(
        brightness=brightness, contrast=contrast, saturation=saturation, hue=hue, p=p
    )


### Crop


def _crop_input(ctx, inputs):
    inputs.int(
        "crop__x_min",
        label="X min",
        description="The minimum upper left x coordinate of the crop in pixels",
        required=True,
    )
    inputs.int(
        "crop__y_min",
        label="Y min",
        description="The minimum upper left y coordinate of the crop in pixels",
        required=True,
    )
    inputs.int(
        "crop__x_max",
        label="X max",
        description="The maximum lower right x coordinate of the crop in pixels",
        required=True,
    )
    inputs.int(
        "crop__y_max",
        label="Y max",
        description="The maximum lower right y coordinate of the crop in pixels",
        required=True,
    )

    inputs.float(
        "crop__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=1.0,
    )


def _crop_transform(ctx):
    x_min = ctx.params.get("crop__x_min", None)
    y_min = ctx.params.get("crop__y_min", None)
    x_max = ctx.params.get("crop__x_max", None)
    y_max = ctx.params.get("crop__y_max", None)
    p = ctx.params.get("crop__p", None)

    return A.Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, p=p)


### Downscale


def _downscale_input(ctx, inputs):
    ## interpolation not supported yet
    inputs.float(
        "downscale__scale_min",
        label="Scale min",
        description="Minimum scale factor",
        required=True,
        default=0.25,
    )
    inputs.float(
        "downscale__scale_max",
        label="Scale max",
        description="Maximum scale factor",
        required=True,
        default=0.25,
    )
    inputs.float(
        "downscale__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=1.0,
    )


def _downscale_transform(ctx):
    scale_min = ctx.params.get("downscale__scale_min", None)
    scale_max = ctx.params.get("downscale__scale_max", None)
    p = ctx.params.get("downscale__p", None)
    return A.Downscale(scale_min=scale_min, scale_max=scale_max, p=p)


### Embose -- Not yet supported


### Equalize


def _equalize_input(ctx, inputs):
    ## mode not supported yet
    inputs.bool(
        "equalize__by_channels",
        label="By channels",
        description="Apply equalization by channels separately",
        required=True,
        default=True,
    )
    inputs.float(
        "equalize__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=0.5,
    )


def _equalize_transform(ctx):
    by_channels = ctx.params.get("equalize__by_channels", None)
    p = ctx.params.get("equalize__p", None)
    return A.Equalize(by_channels=by_channels, p=p)


### FancyPCA -- Not yet supported


### Flip


def _flip_input(ctx, inputs):
    inputs.float(
        "flip__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=0.5,
    )


def _flip_transform(ctx):
    p = ctx.params.get("flip__p", None)
    return A.Flip(p=p)


### GaussNoise


def _gauss_noise_input(ctx, inputs):
    inputs.float(
        "gauss_noise__var_limit_min",
        label="Minimum limit for Variance",
        required=True,
        default=10.0,
    )
    inputs.float(
        "gauss_noise__var_limit_max",
        label="Maximum limit for Variance",
        required=True,
        default=50.0,
    )
    inputs.float(
        "gauss_noise__mean",
        label="Mean",
        description="Mean of the noise",
        required=True,
        default=0,
    )
    inputs.bool(
        "gauss_noise__per_channel",
        label="Per channel",
        description="If True, noise will be sampled independently for each channel value",
        required=True,
        default=True,
    )
    inputs.float(
        "gauss_noise__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=0.5,
    )


def _gauss_noise_transform(ctx):
    var_limit_min = ctx.params.get("gauss_noise__var_limit_min", None)
    var_limit_max = ctx.params.get("gauss_noise__var_limit_max", None)
    mean = ctx.params.get("gauss_noise__mean", None)
    per_channel = ctx.params.get("gauss_noise__per_channel", None)
    p = ctx.params.get("gauss_noise__p", None)
    return A.GaussNoise(
        var_limit=(var_limit_min, var_limit_max),
        mean=mean,
        per_channel=per_channel,
        p=p,
    )


### HorizontalFlip


def _horizontal_flip_input(ctx, inputs):
    inputs.float(
        "horizontal_flip__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=0.5,
    )


def _horizontal_flip_transform(ctx):
    p = ctx.params.get("horizontal_flip__p", None)
    return A.HorizontalFlip(p=p)


### HueSaturationValue -- Not yet supported

### ImageCompression


def _image_compression_input(ctx, inputs):
    inputs.int(
        "image_compression__quality_lower",
        label="Lower quality limit",
        description="Lower bound on the image quality",
        required=True,
        default=99,
    )
    inputs.int(
        "image_compression__quality_upper",
        label="Upper quality limit",
        description="Upper bound on the image quality",
        required=True,
        default=100,
    )
    inputs.float(
        "image_compression__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=0.5,
    )


def _image_compression_transform(ctx):
    ## Compression type not supported yet
    quality_lower = ctx.params.get("image_compression__quality_lower", None)
    quality_upper = ctx.params.get("image_compression__quality_upper", None)
    p = ctx.params.get("image_compression__p", None)
    return A.ImageCompression(
        quality_lower=quality_lower, quality_upper=quality_upper, p=p
    )


### InvertImg


def _invert_img_input(ctx, inputs):
    inputs.float(
        "invert_img__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=0.5,
    )


def _invert_img_transform(ctx):
    p = ctx.params.get("invert_img__p", None)
    return A.InvertImg(p=p)


### ISONoise -- Not yet supported


### JpegCompression


def _jpeg_compression_input(ctx, inputs):
    inputs.int(
        "jpeg_compression__quality_lower",
        label="Lower quality limit",
        description="Lower bound on the image quality",
        required=True,
        default=99,
    )
    inputs.int(
        "jpeg_compression__quality_upper",
        label="Upper quality limit",
        description="Upper bound on the image quality",
        required=True,
        default=100,
    )
    inputs.float(
        "jpeg_compression__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=0.5,
    )


def _jpeg_compression_transform(ctx):
    quality_lower = ctx.params.get("jpeg_compression__quality_lower", None)
    quality_upper = ctx.params.get("jpeg_compression__quality_upper", None)
    p = ctx.params.get("jpeg_compression__p", None)
    return A.JpegCompression(
        quality_lower=quality_lower, quality_upper=quality_upper, p=p
    )


### LongestMaxSize


def _longest_max_size_input(ctx, inputs):
    inputs.int(
        "longest_max_size__max_size",
        label="Max size",
        description="The maximum size of the longest edge in pixels",
        required=True,
    )
    inputs.float(
        "longest_max_size__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=1.0,
    )


def _longest_max_size_transform(ctx):
    max_size = ctx.params.get("longest_max_size__max_size", None)
    p = ctx.params.get("longest_max_size__p", None)
    return A.LongestMaxSize(max_size=max_size, p=p)


### Normalize -- Not yet supported


### OpticalDistortion


def _optical_distortion_input(ctx, inputs):
    ## [float, float] for distort_limit not supported yet
    ## [float, float] for shift_limit not supported yet
    ## interpolation not supported yet
    ## border_mode not supported yet
    ## value not supported yet
    ## mask_value not supported yet
    inputs.float(
        "optical_distortion__distort_limit",
        label="Distort limit",
        description="The maximum absolute change in image brightness",
        required=True,
        default=0.05,
    )
    inputs.float(
        "optical_distortion__shift_limit",
        label="Shift limit",
        description="The maximum absolute change in image brightness",
        required=True,
        default=0.05,
    )
    inputs.float(
        "optical_distortion__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=0.5,
    )


def _optical_distortion_transform(ctx):
    distort_limit = ctx.params.get("optical_distortion__distort_limit", None)
    shift_limit = ctx.params.get("optical_distortion__shift_limit", None)
    p = ctx.params.get("optical_distortion__p", None)
    return A.OpticalDistortion(
        distort_limit=distort_limit, shift_limit=shift_limit, p=p
    )


### PadIfNeeded


def _pad_if_needed_input(ctx, inputs):
    ## position not supported yet
    ## border_mode not supported yet
    ## value not supported yet
    ## mask_value not supported yet
    inputs.int(
        "pad_if_needed__min_height",
        label="Minimum height",
        description="Minimum height of the output",
        required=True,
    )
    inputs.int(
        "pad_if_needed__min_width",
        label="Minimum width",
        description="Minimum width of the output",
        required=True,
    )
    inputs.float(
        "pad_if_needed__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=1.0,
    )


def _pad_if_needed_transform(ctx):
    min_height = ctx.params.get("pad_if_needed__min_height", None)
    min_width = ctx.params.get("pad_if_needed__min_width", None)
    p = ctx.params.get("pad_if_needed__p", None)
    return A.PadIfNeeded(min_height=min_height, min_width=min_width, p=p)


### Perspective


def _perspective_input(ctx, inputs):
    inputs.float(
        "perspective__scale",
        label="Scale",
        description=(
            "Standard deviation of the normal distributions used to sample the ",
            " random distances of the subimage's corners from the full image's corners.",
        ),
        required=True,
        default=0.1,
    )
    inputs.bool(
        "perspective__keep_size",
        label="Keep size",
        description="Whether to resize image’s back to their original size after applying the perspective transform",
        required=True,
        default=True,
    )
    inputs.float(
        "perspective__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=0.5,
    )


def _perspective_transform(ctx):
    scale = ctx.params.get("perspective__scale", None)
    keep_size = ctx.params.get("perspective__keep_size", None)
    p = ctx.params.get("perspective__p", None)
    return A.Perspective(scale=scale, keep_size=keep_size, p=p)


def _pixel_dropout_input(ctx, inputs):
    inputs.float(
        "pixel_dropout__dropout_prob",
        label="Pixel drop probability",
        required=True,
        default=0.01,
    )
    inputs.bool(
        "pixel_dropout__per_channel",
        label="Per channel",
        description=(
            "if set to True drop mask will be sampled fo each channel,",
            "otherwise the same mask will be sampled for all channels.",
        ),
        required=True,
        default=False,
    )
    inputs.float(
        "pixel_dropout__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=0.5,
    )


def _pixel_dropout_transform(ctx):
    dropout_prob = ctx.params.get("pixel_dropout__dropout_prob", None)
    per_channel = ctx.params.get("pixel_dropout__per_channel", None)
    p = ctx.params.get("pixel_dropout__p", None)
    return A.PixelDropout(dropout_prob=dropout_prob, per_channel=per_channel, p=p)


### Posterize -- Not yet supported


### RandomBrightness


def _random_brightness_input(ctx, inputs):
    ## [float, float] for limit not supported yet
    inputs.float(
        "random_brightness__limit",
        label="Brightness limit",
        description="Factor range for changing brightness.",
        required=True,
        default=0.2,
    )
    inputs.float(
        "random_brightness__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=0.5,
    )


def _random_brightness_transform(ctx):
    limit = ctx.params.get("random_brightness__limit", None)
    p = ctx.params.get("random_brightness__p", None)
    return A.RandomBrightness(limit=limit, p=p)


### RandomBrightnessContrast


def _random_brightness_contrast_input(ctx, inputs):
    ## [float, float] for brightness_limit not supported yet
    ## [float, float] for contrast_limit not supported yet
    inputs.float(
        "random_brightness_contrast__brightness_limit",
        label="Brightness limit",
        description="Factor range for changing brightness.",
        required=True,
        default=0.2,
    )
    inputs.float(
        "random_brightness_contrast__contrast_limit",
        label="Contrast limit",
        description="Factor range for changing contrast.",
        required=True,
        default=0.2,
    )
    inputs.bool(
        "random_brightness_contrast__brightness_by_max",
        label="Brightness by max",
        description="If True adjust contrast by image dtype maximum.",
        required=True,
        default=True,
    )
    inputs.float(
        "random_brightness_contrast__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=0.5,
    )


def _random_brightness_contrast_transform(ctx):
    brightness_limit = ctx.params.get(
        "random_brightness_contrast__brightness_limit", None
    )
    contrast_limit = ctx.params.get("random_brightness_contrast__contrast_limit", None)
    brightness_by_max = ctx.params.get(
        "random_brightness_contrast__brightness_by_max", None
    )
    p = ctx.params.get("random_brightness_contrast__p", None)
    return A.RandomContrast(
        brightness_limit=brightness_limit,
        contrast_limit=contrast_limit,
        brightness_by_max=brightness_by_max,
        p=p,
    )


### RandomContrast


def _random_contrast_input(ctx, inputs):
    ## [float, float] for limit not supported yet
    inputs.float(
        "random_contrast__limit",
        label="Contrast limit",
        description="Factor range for changing contrast.",
        required=True,
        default=0.2,
    )
    inputs.float(
        "random_contrast__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=0.5,
    )


def _random_contrast_transform(ctx):
    limit = ctx.params.get("random_contrast__limit", None)
    p = ctx.params.get("random_contrast__p", None)
    return A.RandomContrast(limit=limit, p=p)


### RandomCrop


def _random_crop_input(ctx, inputs):
    inputs.int(
        "random_crop__height",
        label="Height",
        description="The height of the crop in pixels",
        required=True,
    )
    inputs.int(
        "random_crop__width",
        label="Width",
        description="The width of the crop in pixels",
        required=True,
    )
    inputs.float(
        "random_crop__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=1.0,
    )


def _random_crop_transform(ctx):
    height = ctx.params.get("random_crop__height", None)
    width = ctx.params.get("random_crop__width", None)
    p = ctx.params.get("random_crop__p", None)
    return A.RandomCrop(height=height, width=width, p=p)


### RandomFog


def _random_fog_input(ctx, inputs):
    inputs.float(
        "random_fog__fog_coef_lower",
        label="Lower fog coefficient",
        description="Lower bound on the fog intensity coefficient. Should be in the range [0, fog_coef_upper]",
        required=True,
        default=0.3,
    )
    inputs.float(
        "random_fog__fog_coef_upper",
        label="Upper fog coefficient",
        description="Upper bound on the fog intensity coefficient. Should be in the range [fog_coef_lower, 1.0]",
        required=True,
        default=1,
    )
    inputs.float(
        "random_fog__alpha_coef",
        label="Alpha coefficient",
        description="Transparency of the fog circles. Must be in the range (0, 1]",
        required=True,
        default=0.08,
    )
    inputs.float(
        "random_fog__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=0.5,
    )


def _random_fog_transform(ctx):
    fog_coef_lower = ctx.params.get("random_fog__fog_coef_lower", None)
    fog_coef_upper = ctx.params.get("random_fog__fog_coef_upper", None)
    alpha_coef = ctx.params.get("random_fog__alpha_coef", None)
    p = ctx.params.get("random_fog__p", None)
    return A.RandomFog(fog_coef_lower=fog_coef_lower, fog_coef_upper=fog_coef_upper, alpha_coef=alpha_coef, p=p)


### RandomGamma


def _random_gamma_input(ctx, inputs):
    inputs.float(
        "random_gamma__gamma_limit_min",
        label="Minimum limit for gamma",
        required=True,
        default=80.0,
    )
    inputs.float(
        "random_gamma__gamma_limit_max",
        label="Maximum limit for gamma",
        required=True,
        default=120.0,
    )
    inputs.float(
        "random_gamma__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=0.5,
    )


def _random_gamma_transform(ctx):
    gamma_limit_min = ctx.params.get("random_gamma__gamma_limit_min", None)
    gamma_limit_max = ctx.params.get("random_gamma__gamma_limit_max", None)
    p = ctx.params.get("random_gamma__p", None)
    return A.RandomGamma(gamma_limit=(gamma_limit_min, gamma_limit_max), p=p)


### RandomGravel -- Not yet supported


### RandomGridShuffle


def _random_grid_shuffle_input(ctx, inputs):
    ## [int, int] for grid not supported yet
    inputs.int(
        "random_grid_shuffle__grid",
        label="Grid",
        description="Grid size for shuffling.",
        required=True,
        default=3,
    )
    inputs.int(
        "random_grid_shuffle__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=0.5,
    )


def _random_grid_shuffle_transform(ctx):
    grid = ctx.params.get("random_grid_shuffle__grid", None)
    p = ctx.params.get("random_grid_shuffle__p", None)
    return A.RandomGridShuffle(grid=grid, p=p)


### RandomRain


def _random_rain_input(ctx, inputs):
    ## drop color not supported yet
    ## rain type not supported yet
    inputs.float(
        "random_rain__slant_lower",
        label="Lower slant limit",
        description="Lower bound on the slant of the rain drops. Should be in the range [-20, slant_upper]",
        required=True,
        default=-10,
    )
    inputs.float(
        "random_rain__slant_upper",
        label="Upper slant limit",
        description="Upper bound on the slant of the rain drops. Should be in the range [slant_lower, 20]",
        required=True,
        default=10,
    )
    inputs.float(
        "random_rain__drop_length",
        label="Drop length",
        description="Length of the rain drops in pixels",
        required=True,
        default=20,
    )
    inputs.float(
        "random_rain__drop_width",
        label="Drop width",
        description="Width of the rain drops in pixels",
        required=True,
        default=1,
    )
    inputs.float(
        "random_rain__blur_value",
        label="Blur value",
        description="Blurring value to simulate rain drop depth",
        required=True,
        default=7,
    )
    inputs.float(
        "random_rain__brightness_coefficient",
        label="Brightness coefficient",
        description="Brightness coefficient. Should be in the range (0, 1)",
        required=True,
        default=0.7,
    )
    inputs.float(
        "random_rain__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=0.5,
    )


def _random_rain_transform(ctx):
    slant_lower = ctx.params.get("random_rain__slant_lower", None)
    slant_upper = ctx.params.get("random_rain__slant_upper", None)
    drop_length = ctx.params.get("random_rain__drop_length", None)
    drop_width = ctx.params.get("random_rain__drop_width", None)
    blur_value = ctx.params.get("random_rain__blur_value", None)
    brightness_coefficient = ctx.params.get(
        "random_rain__brightness_coefficient", None
    )
    p = ctx.params.get("random_rain__p", None)
    return A.RandomRain(
        slant_lower=slant_lower,
        slant_upper=slant_upper,
        drop_length=drop_length,
        drop_width=drop_width,
        blur_value=blur_value,
        brightness_coefficient=brightness_coefficient,
        p=p,
    )


### RandomResizedCrop


def _random_resized_crop_input(ctx, inputs):
    inputs.int(
        "random_resized_crop__height",
        label="Height",
        description="The height after crop and resize.",
        required=True,
        default=224,
    )
    inputs.int(
        "random_resized_crop__width",
        label="Width",
        description="The width after crop and resize.",
        required=True,
        default=224,
    )

    inputs.float(
        "random_resized_crop__scale_min",
        label="Scale min",
        description="Minimum scale of the crop.",
        required=True,
        default=0.08,
    )
    inputs.float(
        "random_resized_crop__scale_max",
        label="Scale max",
        description="Maximum scale of the crop.",
        required=True,
        default=1.0,
    )

    inputs.float(
        "random_resized_crop__ratio_min",
        label="Ratio min",
        description="Minimum aspect ratio of the crop.",
        required=True,
        default=0.75,
    )
    inputs.float(
        "random_resized_crop__ratio_max",
        label="Ratio max",
        description="Maximum aspect ratio of the crop.",
        required=True,
        default=1.33,
    )


def _random_resized_crop_transform(ctx):
    height = ctx.params.get("random_resized_crop__height", None)
    width = ctx.params.get("random_resized_crop__width", None)
    scale_min = ctx.params.get("random_resized_crop__scale_min", None)
    scale_max = ctx.params.get("random_resized_crop__scale_max", None)
    ratio_min = ctx.params.get("random_resized_crop__ratio_min", None)
    ratio_max = ctx.params.get("random_resized_crop__ratio_max", None)

    return A.RandomResizedCrop(
        height=height,
        width=width,
        scale=(scale_min, scale_max),
        ratio=(ratio_min, ratio_max),
        p=1.0,
    )


### RandomRotate90


def _random_rotate90_input(ctx, inputs):
    inputs.float(
        "random_rotate90__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=0.5,
    )


def _random_rotate90_transform(ctx):
    p = ctx.params.get("random_rotate90__p", None)
    return A.RandomRotate90(p=p)


### RandomScale


def _random_scale_input(ctx, inputs):
    inputs.float(
        "random_scale__scale_limit",
        label="Scale limit",
        description="Scaling factor interval.",
        required=True,
        default=0.2,
    )
    inputs.bool(
        "random_scale__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=0.5,
    )


def _random_scale_transform(ctx):
    scale_limit = ctx.params.get("random_scale__scale_limit", None)
    interpolation = cv2.INTER_NEAREST
    p = ctx.params.get("random_scale__p", None)
    return A.RandomScale(scale_limit=scale_limit, interpolation=interpolation, p=p)


### RandomShadow -- Not yet supported


### RandomSnow


def _random_snow_input(ctx, inputs):
    inputs.float(
        "random_snow__snow_point_lower",
        label="Lower snow point",
        description="Lower bound on the snow point. Should be in the range [0, snow_point_upper]",
        required=True,
        default=0.1,
    )
    inputs.float(
        "random_snow__snow_point_upper",
        label="Upper snow point",
        description="Upper bound on the snow point. Should be in the range [snow_point_lower, 1.0]",
        required=True,
        default=0.3,
    )
    inputs.float(
        "random_snow__brightness_coeff",
        label="Brightness coefficient",
        description="Brightness coefficient. Should be in the range (0, 1)",
        required=True,
        default=2.5,
    )
    inputs.float(
        "random_snow__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=0.5,
    )


def _random_snow_transform(ctx):
    snow_point_lower = ctx.params.get("random_snow__snow_point_lower", None)
    snow_point_upper = ctx.params.get("random_snow__snow_point_upper", None)
    brightness_coeff = ctx.params.get("random_snow__brightness_coeff", None)
    p = ctx.params.get("random_snow__p", None)
    return A.RandomSnow(
        snow_point_lower=snow_point_lower,
        snow_point_upper=snow_point_upper,
        brightness_coeff=brightness_coeff,
        p=p,
    )


### RandomSunFlare -- Not yet supported


### RandomToneCurve


def _random_tone_curve_input(ctx, inputs):
    inputs.float(
        "random_tone_curve__scale",
        label="Scale",
        description=(
            "standard deviation of the normal distribution. Used to sample "
            "random distances to move two control points that modify the image's"
            " curve. Values should be in range [0, 1]. "
        ),
        required=True,
        default=0.1,
    )
    inputs.float(
        "random_tone_curve__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=0.5,
    )


def _random_tone_curve_transform(ctx):
    scale = ctx.params.get("random_tone_curve__scale", None)
    p = ctx.params.get("random_tone_curve__p", None)
    return A.RandomToneCurve(scale=scale, p=p)


### Resize


def _resize_input(ctx, inputs):
    inputs.int(
        "resize__height",
        label="Height",
        description="The height after crop and resize.",
        required=True,
        default=224,
    )
    inputs.int(
        "resize__width",
        label="Width",
        description="The width after crop and resize.",
        required=True,
        default=224,
    )
    inputs.float(
        "resize__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=1.0,
    )


def _resize_transform(ctx):
    height = ctx.params.get("resize__height", None)
    width = ctx.params.get("resize__width", None)
    p = ctx.params.get("resize__p", None)
    return A.Resize(height=height, width=width, p=p)


### RGBShift


def _rgb_shift_input(ctx, inputs):
    ## [int, int] for r_shift_limit not supported yet
    ## [int, int] for g_shift_limit not supported yet
    ## [int, int] for b_shift_limit not supported yet
    inputs.int(
        "rgb_shift__r_shift_limit",
        label="R shift limit",
        description="Factor range for changing red color channel.",
        required=True,
        default=20,
    )
    inputs.int(
        "rgb_shift__g_shift_limit",
        label="G shift limit",
        description="Factor range for changing green color channel.",
        required=True,
        default=20,
    )
    inputs.int(
        "rgb_shift__b_shift_limit",
        label="B shift limit",
        description="Factor range for changing blue color channel.",
        required=True,
        default=20,
    )
    inputs.float(
        "rgb_shift__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=0.5,
    )


def _rgb_shift_transform(ctx):
    r_shift_limit = ctx.params.get("rgb_shift__r_shift_limit", None)
    g_shift_limit = ctx.params.get("rgb_shift__g_shift_limit", None)
    b_shift_limit = ctx.params.get("rgb_shift__b_shift_limit", None)
    p = ctx.params.get("rgb_shift__p", None)
    return A.RGBShift(
        r_shift_limit=r_shift_limit, g_shift_limit=g_shift_limit, b_shift_limit=b_shift_limit, p=p
    )


### RingingOvershoot -- Not yet supported
### Sharpen -- Not yet supported
### ShiftScaleRotate -- Not yet supported
### Solarize -- Not yet supported
### Spatter -- Not yet supported
### Superpixels -- Not yet supported


### Transpose


def _transpose_input(ctx, inputs):
    inputs.float(
        "transpose__p",
        label="Probability",
        description="The probability of applying the transform",
        required=True,
        default=0.5,
    )


def _transpose_transform(ctx):
    p = ctx.params.get("transpose__p", None)
    return A.Transpose(p=p)



### TODO: CropAndPad, PadIfNeeded, PiecewiseAffine, ...


transform_name_to_input_parser = {
    "affine": _affine_input,
    "center_crop": _center_crop_input,
    "channel_dropout": _channel_dropout_input,
    "channel_shuffle": _channel_shuffle_input,
    "clahe": _clahe_input,
    "color_jitter": _color_jitter_input,
    "crop": _crop_input,
    "downscale": _downscale_input,
    "equalize": _equalize_input,
    "flip": _flip_input,
    "gauss_noise": _gauss_noise_input,
    "horizontal_flip": _horizontal_flip_input,
    "image_compression": _image_compression_input,
    "invert_img": _invert_img_input,
    "jpeg_compression": _jpeg_compression_input,
    "longest_max_size": _longest_max_size_input,
    "optical_distortion": _optical_distortion_input,
    "pad_if_needed": _pad_if_needed_input,
    "perspective": _perspective_input,
    "pixel_dropout": _pixel_dropout_input,
    "random_brightness": _random_brightness_input,
    "random_brightness_contrast": _random_brightness_contrast_input,
    "random_contrast": _random_contrast_input,
    "random_crop": _random_crop_input,
    "random_fog": _random_fog_input,
    "random_gamma": _random_gamma_input,
    "random_grid_shuffle": _random_grid_shuffle_input,
    "random_rain": _random_rain_input,
    "random_resized_crop": _random_resized_crop_input,
    "random_rotate90": _random_rotate90_input,
    "random_scale": _random_scale_input,
    "random_snow": _random_snow_input,
    "random_tone_curve": _random_tone_curve_input,
    "resize": _resize_input,
    "rgb_shift": _rgb_shift_input,
    "transpose": _transpose_input,
}


transform_name_to_transform = {
    "affine": _affine_transform,
    "center_crop": _center_crop_transform,
    "channel_dropout": _channel_dropout_transform,
    "channel_shuffle": _channel_shuffle_transform,
    "clahe": _clahe_transform,
    "color_jitter": _color_jitter_transform,
    "crop": _crop_transform,
    "downscale": _downscale_transform,
    "equalize": _equalize_transform,
    "flip": _flip_transform,
    "gauss_noise": _gauss_noise_transform,
    "horizontal_flip": _horizontal_flip_transform,
    "image_compression": _image_compression_transform,
    "invert_img": _invert_img_transform,
    "jpeg_compression": _jpeg_compression_transform,
    "longest_max_size": _longest_max_size_transform,
    "optical_distortion": _optical_distortion_transform,
    "pad_if_needed": _pad_if_needed_transform,
    "perspective": _perspective_transform,
    "pixel_dropout": _pixel_dropout_transform,
    "random_brightness": _random_brightness_transform,
    "random_brightness_contrast": _random_brightness_contrast_transform,
    "random_contrast": _random_contrast_transform,
    "random_crop": _random_crop_transform,
    "random_fog": _random_fog_transform,
    "random_gamma": _random_gamma_transform,
    "random_grid_shuffle": _random_grid_shuffle_transform,
    "random_rain": _random_rain_transform,
    "random_resized_crop": _random_resized_crop_transform,
    "random_rotate90": _random_rotate90_transform,
    "random_scale": _random_scale_transform,
    "random_snow": _random_snow_transform,
    "random_tone_curve": _random_tone_curve_transform,
    "resize": _resize_transform,
    "rgb_shift": _rgb_shift_transform,
    "transpose": _transpose_transform,
}


transform_name_to_label = {
    "affine": "Affine",
    "center_crop": "CenterCrop",
    "channel_dropout": "ChannelDropout",
    "channel_shuffle": "ChannelShuffle",
    "clahe": "CLAHE",
    "color_jitter": "ColorJitter",
    "crop": "Crop",
    "downscale": "Downscale",
    "equalize": "Equalize",
    "flip": "Flip",
    "gauss_noise": "GaussNoise",
    "horizontal_flip": "HorizontalFlip",
    "image_compression": "ImageCompression",
    "invert_img": "InvertImg",
    "jpeg_compression": "JpegCompression",
    "longest_max_size": "LongestMaxSize",
    "perspective": "Perspective",
    "pixel_dropout": "PixelDropout",
    "random_brightness": "RandomBrightness",
    "random_brightness_contrast": "RandomBrightnessContrast",
    "random_contrast": "RandomContrast",
    "random_crop": "RandomCrop",
    "random_fog": "RandomFog",
    "random_gamma": "RandomGamma",
    "random_grid_shuffle": "RandomGridShuffle",
    "random_rain": "RandomRain",
    "random_resized_crop": "RandomResizedCrop",
    "random_rotate90": "RandomRotate90",
    "random_scale": "RandomScale",
    "random_snow": "RandomSnow",
    "random_tone_curve": "RandomToneCurve",
    "resize": "Resize",
    "rgb_shift": "RGBShift",
}


def _transforms_input(ctx, inputs):
    transform_choices = list(transform_name_to_label.keys())

    transforms_group = types.RadioGroup()

    for tc in transform_choices:
        transforms_group.add_choice(tc, label=transform_name_to_label[tc])

    inputs.enum(
        "transforms",
        transforms_group.values(),
        label="Transform to apply",
        description="The Albumentations transform to apply to your images",
        view=types.AutocompleteView(),
        required=True,
    )


def _cleanup_last_transform(dataset):
    run_key = LAST_ALBUMENTATIONS_RUN_KEY

    if run_key not in dataset.list_runs():
        return

    results = dataset.load_run_results(run_key)
    if results.save_augmentations:
        results.save_augmentations = False
        dataset.save_run_results(run_key, results, overwrite=True)
        return

    ids = results.new_sample_ids
    fps = dataset.select(ids).values("filepath")
    for fp in fps:
        os.remove(fp)
    
    dataset.delete_samples(ids)



def _store_last_transform(transform, dataset, target_view, label_fields, new_sample_ids):
    _cleanup_last_transform(dataset)
    run_key = LAST_ALBUMENTATIONS_RUN_KEY
    transform_dict = transform.to_dict()

    config = dataset.init_run()
    config.transform = transform_dict
    config.label_fields = label_fields

    if run_key not in dataset.list_runs():
        dataset.register_run(run_key, config)
        results = dataset.init_run_results(run_key)
    else:
        dataset.update_run_config(run_key, config)
        results = dataset.load_run_results(run_key)
    
    results.target_view = target_view._serialize()
    results.new_sample_ids = new_sample_ids
    results.save_augmentations = False
    dataset.save_run_results(run_key, results, overwrite=True)


def _save_transform(dataset, transform, name):
    hash = _create_hash()

    run_key = f"albumentations_transform_{hash}"

    config = dataset.init_run()
    config.name = name
    config.transform = transform
    dataset.register_run(run_key, config)


def _save_augmentations(ctx):
    """
    Save the samples generated by the last Albumentations run. If this is not
    done, the samples will be deleted when the next Albumentations run is
    executed.
    """
    dataset = ctx.dataset
    run_key = LAST_ALBUMENTATIONS_RUN_KEY

    if run_key not in dataset.list_runs():
        return
    
    results = dataset.load_run_results(run_key)
    results.save_augmentations = True
    dataset.save_run_results(run_key, results, overwrite=True)


class AugmentWithAlbumentations(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="augment_with_albumentations",
            label="Augment with Albumentations",
            description="Apply an Albumentations transform to the image and all label fields listed for the sample.",
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="Augment with Albumentations",
            description="Apply an Albumentations transform to the image and all label fields listed for the sample.",
        )

        _transforms_input(ctx, inputs)

        transform_name = ctx.params.get("transforms", None)

        if transform_name is not None and transform_name in transform_name_to_input_parser:
            transform_input_parser = transform_name_to_input_parser[transform_name]
            transform_input_parser(ctx, inputs)

        inputs.int(
            "num_augs",
            label="Number of augmentations per sample",
            description="The number of random augmentations to apply to each sample",
            default=1,
            view=types.FieldView(),
        )

        # inputs.bool(
        #     "label_fields",
        #     label="Label fields",
        #     description=(
        #         "A field or list of field names to transform. If label_fields=False, "
        #         "no label fields are transformed. If label_fields=True, all label "
        #         "fields are transformed.",
        #     ),
        #     default=False,
        #     view=types.CheckboxView(),
        # )

        _list_target_views(ctx, inputs)

        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        transform_name = ctx.params.get("transforms", None)
        transform_func = transform_name_to_transform[transform_name]
        transform = transform_func(ctx)

        transform = A.Compose(
            [
                transform,
            ],
            bbox_params=A.BboxParams(format="albumentations"),
            keypoint_params=A.KeypointParams(
                format="xy", label_fields=["keypoint_labels"], remove_invisible=True
            ),
        )

        

        num_augs = ctx.params.get("num_augs", 1)

        label_fields = True

        target = ctx.params.get("target", None)
        target_view = _get_target_view(ctx, target)

        new_sample_ids = []

        for sample in target_view:
            for _ in range(num_augs):
                new_sample_id = transform_sample(sample, transform, label_fields)
                new_sample_ids.append(new_sample_id)

        _store_last_transform(transform, ctx.dataset, target_view, label_fields, new_sample_ids)
        ctx.trigger("reload_dataset")


class GetLastAlbumentationsRunInfo(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="get_last_albumentations_run_info",
            label="Get info about the last Albumentations run",
            light_icon="/assets/icon.svg",
            dark_icon="/assets/icon.svg",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        view = types.View(label="Get last Albumentations run info")
        return types.Property(inputs, view=view)

    def execute(self, ctx):
        run_key = LAST_ALBUMENTATIONS_RUN_KEY

        if run_key not in ctx.dataset.list_runs():
            return {
                "message": "To create a transform, use the `Augment with Albumentations` operator",
            }

        info = ctx.dataset.get_run_info(run_key)

        timestamp = info.timestamp.strftime("%Y-%M-%d %H:%M:%S")
        config = info.config.serialize()
        config = {k: v for k, v in config.items() if v is not None}

        transform = config.get("transform", {})
        if "transform" in transform:
            transform = transform["transform"].copy()
            transform.pop("bbox_params", None)
            transform.pop("keypoint_params", None)
            transform.pop("additional_targets", None)
            transform.pop("is_check_shapes", None)

        label_fields = config.get("label_fields", None)

        return {
            "timestamp": timestamp,
            "version": info.version,
            "label_fields": label_fields,
            "transform": transform,
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        if LAST_ALBUMENTATIONS_RUN_KEY not in ctx.dataset.list_runs():
            outputs.str("message", label="No Albumentations runs yet!")
            view = types.View()
            return types.Property(outputs, view=view)
        
        outputs.str("timestamp", label="Creation time")
        outputs.str("version", label="FiftyOne version")
        outputs.str("label_fields", label="Label fields")
        outputs.obj("transform", label="Transform", view=types.JSONView())
        view = types.View(label="Last Albumentations run info")
        return types.Property(outputs, view=view)
    

class ViewLastAlbumentationsRun(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="view_last_albumentations_run",
            label="View last Albumentations run",
            light_icon="/assets/icon.svg",
            dark_icon="/assets/icon.svg",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        view = types.View(label="View last Albumentations run")
        return types.Property(inputs, view=view)

    def execute(self, ctx):
        run_key = LAST_ALBUMENTATIONS_RUN_KEY

        if run_key not in ctx.dataset.list_runs():
            return {
                "message": "To create a transform, use the `Augment with Albumentations` operator",
            }
        
        results = ctx.dataset.load_run_results(run_key)
        new_sample_ids = results.new_sample_ids
        view = ctx.dataset.select(new_sample_ids)

        ctx.trigger(
            "set_view",
            params=dict(view=serialize_view(view)),
        )


class SaveLastAlbumentationsTransform(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="save_albumentations_transform",
            label="Save transform",
            light_icon="/assets/icon.svg",
            dark_icon="/assets/icon.svg",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        run_key = LAST_ALBUMENTATIONS_RUN_KEY

        info = ctx.dataset.get_run_info(run_key)

        config = info.config.serialize()
        config = {k: v for k, v in config.items() if v is not None}

        transform = config.get("transform", {})
        if "transform" in transform:
            transform = transform["transform"].copy()
            transform.pop("bbox_params", None)
            transform.pop("keypoint_params", None)
            transform.pop("additional_targets", None)
            transform.pop("is_check_shapes", None)

        inputs.obj(
            "transform",
            label="Transform",
            description="Serialized description of the transform",
            default=transform,
            required=False,
            view=types.JSONView(readonly=True),
        )

        inputs.str(
            "name",
            label="Name",
            description="The name of the transform. Cannot contain spaces or special characters.",
            required=True,
        )
        view = types.View(label="Save transform")
        return types.Property(inputs, view=view)

    def execute(self, ctx):
        last_run_key = LAST_ALBUMENTATIONS_RUN_KEY

        if last_run_key not in ctx.dataset.list_runs():
            return {
                "message": "To create a transform, use the `Augment with Albumentations` operator",
            }
        
        results = ctx.dataset.load_run_results(last_run_key)
        transform = results.config.transform
        name = ctx.params.get("name", None)
        _save_transform(ctx.dataset, transform, name)
        ctx.trigger("reload_dataset")


class SaveLastAlbumentationsAugmentations(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="save_albumentations_augmentations",
            label="Save augmentations",
            light_icon="/assets/icon.svg",
            dark_icon="/assets/icon.svg",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        view = types.View(label="Save augmentations")
        return types.Property(inputs, view=view)

    def execute(self, ctx):
        _save_augmentations(ctx)
        ctx.trigger("reload_dataset")


def register(plugin):
    plugin.register(AugmentWithAlbumentations)
    plugin.register(GetLastAlbumentationsRunInfo)
    plugin.register(ViewLastAlbumentationsRun)
    plugin.register(SaveLastAlbumentationsTransform)
    plugin.register(SaveLastAlbumentationsAugmentations)



### To Do:
## Add a message for each transform
## Delete transform
## See saved transforms
## compose transforms
## Export transform
## descriptive descriptions
## Rerun last transform
## Add in more base transforms

## Document with README
## Blog post
