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
import inspect
import re

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types

LAST_ALBUMENTATIONS_RUN_KEY = "_last_albumentations_run"

## for camelCase to snake_case conversion
pattern = re.compile(r'(?<!^)(?=[A-Z])')

NAME_TO_TYPE = {
    "scale": "float",
    "slant_lower": "int",
    "slant_upper": "int",
    "drop_length": "int",
    "drop_width": "int",
    "translate_percent": "float",
    "translate_px": "int",
    "rotate": "float",
    "shear": "float",
}


def _camel_to_snake(name):
    return pattern.sub('_', name).lower()

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


################################################
################################################
######## Transformation Input Helpers ##########

def _count_leading_spaces(line):
    return len(line) - len(line.lstrip())

def _join_lines_with_indentation(lines):
    """
    Joins lines that are indented with the previous line.
    """
    joined_lines = []
    for line in lines:
        if not line.strip():
            continue  # Skip empty lines
        
        if joined_lines:
            if _count_leading_spaces(line) > _count_leading_spaces(joined_lines[-1]):
                joined_lines[-1] += ' ' + line.strip()
            else:
                joined_lines.append(line)
        else:
            joined_lines.append(line)
    
    return joined_lines


def _process_function_args(func):
    docstring = inspect.getdoc(func)
    
    args_section = docstring.split("Args:")[1].split("Targets:")[0]
    args_lines = args_section.splitlines()

    args = _join_lines_with_indentation(args_lines)
    args = [arg.strip() for arg in args if arg.strip() != ""]
    args = [_get_arg_details(arg) for arg in args]
    return [arg for arg in args if arg is not None]


def _get_arg_type_str(arg_name_and_type):
    arg_type = "(".join(arg_name_and_type.split("(")[1:])
    arg_type = ")".join(arg_type.split(")")[:-1])
    return arg_type
    
def _get_arg_details(arg):
    arg_name_and_type = arg.split(":")[0].strip()
    
    try:
        if "(" not in arg_name_and_type:
            arg_name = arg_name_and_type.split(":")[0].strip()
            if arg_name in NAME_TO_TYPE:
                arg_type = NAME_TO_TYPE[arg_name]
                arg_description = ''.join(arg.split(":")[1:]).strip()
        else:
            arg_name = arg_name_and_type.split("(")[0].strip()
            arg_type = _get_arg_type_str(arg_name_and_type)
            arg_description = ''.join(arg.split(":")[1:]).strip()
        return {
            "name": arg_name,
            "type": arg_type,
            "description": arg_description
        }
    except:
        return None

def tuple_creator(inputs, arg_type):
    def input_adder(name, **input_args):
        default = input_args.pop("default", None)

        first_input_args = {
            "view": types.FieldView(space=3),
        }
        second_input_args = {
            "view": types.FieldView(space=3),
        }
        if default is not None and type(default) == tuple and len(default) == 2:
            first_input_args["default"] = default[0]
            first_input_args["required"] = False
            second_input_args["default"] = default[1]
            second_input_args["required"] = False
        

        obj = types.Object()
        constructor = obj.float if arg_type == 'float' else obj.int
        constructor(
            "first",
            **first_input_args
        )
        constructor(
            "second",
            **second_input_args
        )
        inputs.define_property(name, obj, **input_args)
    return input_adder


def _get_input_factory(inputs, arg_type):
    if arg_type == 'bool':
        return inputs.bool
    elif arg_type == 'int':
        return inputs.int
    elif arg_type == 'float':
        return inputs.float
    elif arg_type == 'str':
        return inputs.str
    elif 'int, int' in arg_type:
        return tuple_creator(inputs, 'int')
    elif 'float, float' in arg_type:
        return tuple_creator(inputs, 'float')
    elif 'tuple of int' in arg_type:
        return tuple_creator(inputs, 'int')
    elif 'tuple of float' in arg_type:
        return tuple_creator(inputs, 'float')
    elif arg_type == "int, list of int":
        ## doesn't support list of int yet
        return inputs.int
    elif arg_type == 'int, float':
        return inputs.float
    elif arg_type == '(float, float) or float':
        ## doesn't support tuple yet
        return inputs.float
    elif arg_type == '(int, int) or int':
        ## doesn't support tuple yet
        return inputs.int
    elif 'float,' in arg_type or 'float or' in arg_type:
        ## doesn't support tuple yet
        return inputs.float
    elif 'number,' in arg_type or 'number or' in arg_type:
        ## doesn't support tuple yet
        return inputs.float
    return None




def _add_transform_inputs(inputs, transform_name):
    transform_args = _process_function_args(getattr(A, transform_name))
    
    t = getattr(A, transform_name)
    camel_case_name = _camel_to_snake(transform_name)

    parameters = inspect.signature(t).parameters
    
    for arg in transform_args:
        default = None
        arg_name = arg["name"]
        arg_type = arg["type"]
        arg_description = arg["description"]

        input_args = {
            "label": arg_name,
            "description": arg_description,
            "required": True,
        }

        if arg_name in parameters:
            default = parameters[arg_name].default
        if default is not None and default != None and default != inspect._empty:
            input_args["default"] = default
            input_args["required"] = False
        elif default is None:
            input_args["required"] = False


        input_factory = _get_input_factory(inputs, arg_type)
        if input_factory is not None:
            input_factory(
                f"{camel_case_name}__{arg_name}",
                **input_args
            )

################################################
################################################
######## Transformation Creation Helpers #######
    
def _extract_transform_inputs(ctx, transform_name):
    prefix = _camel_to_snake(transform_name) + "__"
    transform_params = {}
    for param in ctx.params:
        if param.startswith(prefix):
            param_name = param[len(prefix):]
            transform_params[param_name] = ctx.params[param]
    return transform_params


def _unwrap_dict_tuple(d):
    if isinstance(d, dict):
        if "first" in d and "second" in d:
            return (d["first"], d["second"])
    return d


def _unwrap_dict_tuples(args, kwargs):
    args = [_unwrap_dict_tuple(arg) for arg in args]
    kwargs = {key: _unwrap_dict_tuple(val) for key, val in kwargs.items()}
    return args, kwargs


def _create_transform(ctx, transform_name):
    input_dict = _extract_transform_inputs(ctx, transform_name)

    t = getattr(A, transform_name)
    params = inspect.signature(t).parameters

    args = []
    kwargs = {}

    for param_name, param in params.items():
        if param_name in input_dict:
            if param.default != inspect._empty:
                if input_dict[param_name] != None and input_dict[param_name] is not None:
                    kwargs[param_name] = input_dict[param_name]
            else:
                args.append(input_dict[param_name])

    args, kwargs = _unwrap_dict_tuples(args, kwargs)

    return t(*args, **kwargs)


################################################
################################################
################# Transformations ##############

### Embose -- Not yet supported
### FancyPCA -- Not yet supported
### HueSaturationValue -- Not yet supported
### ISONoise -- Not yet supported
### Normalize -- Not yet supported
### Posterize -- Not yet supported
### RandomCropNearBBox  | Target bbox --> not yet supported
### RandomGravel -- Not yet supported


### RandomGridShuffle


# def _random_grid_shuffle_input(ctx, inputs):
#     ## [int, int] for grid not supported yet
#     inputs.int(
#         "random_grid_shuffle__grid",
#         label="Grid",
#         description="Grid size for shuffling.",
#         required=True,
#         default=3,
#     )
#     inputs.int(
#         "random_grid_shuffle__p",
#         label="Probability",
#         description="The probability of applying the transform",
#         required=True,
#         default=0.5,
#     )


# def _random_grid_shuffle_transform(ctx):
#     grid = ctx.params.get("random_grid_shuffle__grid", None)
#     p = ctx.params.get("random_grid_shuffle__p", None)
#     return A.RandomGridShuffle(grid=grid, p=p)


### RandomRain

### DON'T REMOVE
def _random_rain_input(ctx, inputs):
    _add_transform_inputs(inputs, "RandomRain")
    ## drop color not supported yet
    ## rain type not supported yet

def _random_rain_transform(ctx):
    return _create_transform(ctx, "RandomRain")


### RandomShadow -- Not yet supported
### RandomSunFlare -- Not yet supported
### RingingOvershoot -- Not yet supported
### Rotate -- Not yet supported
### Sharpen -- Not yet supported
### ShiftScaleRotate -- Not yet supported
### Solarize -- Not yet supported
### Spatter -- Not yet supported
### Superpixels -- Not yet supported



####### Unifying functions #######

### The camel case conversion should superseed this
transform_name_to_label = {
    "affine": "Affine",
    "bbox_safe_random_crop": "BBoxSafeRandomCrop",
    "center_crop": "CenterCrop",
    "channel_dropout": "ChannelDropout",
    "channel_shuffle": "ChannelShuffle",
    "clahe": "CLAHE",
    "color_jitter": "ColorJitter",
    "crop": "Crop",
    "crop_and_pad": "CropAndPad",
    "downscale": "Downscale",
    "equalize": "Equalize",
    "flip": "Flip",
    "gauss_noise": "GaussNoise",
    "horizontal_flip": "HorizontalFlip",
    "image_compression": "ImageCompression",
    "invert_img": "InvertImg",
    "jpeg_compression": "JpegCompression",
    "longest_max_size": "LongestMaxSize",
    "optical_distortion": "OpticalDistortion",
    "pad_if_needed": "PadIfNeeded",
    "perspective": "Perspective",
    "pixel_dropout": "PixelDropout",
    "random_brightness": "RandomBrightness",
    "random_brightness_contrast": "RandomBrightnessContrast",
    "random_contrast": "RandomContrast",
    "random_crop": "RandomCrop",
    "random_crop_from_borders": "RandomCropFromBorders",
    # "random_crop_near_bbox": "RandomCropNearBBox",
    "random_fog": "RandomFog",
    "random_gamma": "RandomGamma",
    "random_grid_shuffle": "RandomGridShuffle",
    "random_rain": "RandomRain",
    "random_resized_crop": "RandomResizedCrop",
    "random_rotate90": "RandomRotate90",
    "random_scale": "RandomScale",
    "random_sized_bbox_safe_crop": "RandomSizedBBoxSafeCrop",
    "random_snow": "RandomSnow",
    "random_tone_curve": "RandomToneCurve",
    "resize": "Resize",
    "rgb_shift": "RGBShift",
    "transpose": "Transpose",
    "vertical_flip": "VerticalFlip",
}


def get_input_parser(transform_name):
    function_name = f"_{transform_name}_input"
    if function_name in globals():
        # Defined explicitly above
        input_parser_function = globals()[function_name]
    else:
        label = transform_name_to_label[transform_name]
        # Define the function dynamically using a lambda function
        input_parser_function = lambda ctx, inputs: _add_transform_inputs(inputs, label)
    return input_parser_function


def get_transform_func(transform_name):
    function_name = f"_{transform_name}_transform"
    if function_name in globals():
        # Defined explicitly above
        transform_function = globals()[function_name]
    else:
        label = transform_name_to_label[transform_name]
        # Define the function dynamically using a lambda function
        transform_function = lambda ctx: _create_transform(ctx, label)
    return transform_function



def _transforms_input(ctx, inputs, num=0):
    transform_choices = list(transform_name_to_label.keys())

    transforms_group = types.RadioGroup()

    for tc in transform_choices:
        transforms_group.add_choice(tc, label=transform_name_to_label[tc])

    inputs.enum(
        f"transforms__{num}",
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

        inputs.int(
            "num_augs",
            label="Number of augmentations per sample",
            description="The number of random augmentations to apply to each sample",
            default=1,
            view=types.FieldView(),
        )

        inputs.int(
            "num_transforms",
            label="Number of transforms",
            description="The number of transforms to compose and apply to samples",
            default=1,
            required=True,
        )

        num_transforms = ctx.params.get("num_transforms", 1)
        if num_transforms is not None and num_transforms < 1:
            inputs.view(
                "no_transforms_error", 
                types.Error(
                    label="No transforms", 
                    description="The number of transforms must be greater than 0")
            )
            return types.Property(inputs, view=form_view)

        if num_transforms is not None:
            for i in range(num_transforms):
                inputs.view(
                    f"transform_{i}_header",
                    types.Header(label=f"Transform {i+1}", divider=True),
                )
                _transforms_input(ctx, inputs, num=i)
                transform_name = ctx.params.get(f"transforms__{i}", None)
                try:
                    transform_input_parser = get_input_parser(transform_name)
                    transform_input_parser(ctx, inputs)
                except:
                    pass
        
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
        num_transforms = ctx.params.get("num_transforms", 1)
        transform_names = [ctx.params.get(f"transforms__{i}", None) for i in range(num_transforms)]
        transforms = [get_transform_func(transform_name)(ctx) for transform_name in transform_names]

        transform = A.Compose(
            transforms,
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
            icon="/assets/icon.svg",
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
            icon="/assets/icon.svg",
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
            icon="/assets/icon.svg",
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
            icon="/assets/icon.svg",
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