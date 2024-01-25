# Albumentations Data Augmentation Plugin for FiftyOne

Traditionally, data augmentation is performed on-the-fly during training. This is great... *if* you know exactly what augmentations you want to apply to your dataset. 

However, if you're just getting started with a new dataset, you may not know what augmentations are appropriate for your data. In this case, it can be helpful to apply a wide range of augmentations to your dataset and then manually inspect the results to see which augmentations are appropriate for your data.

This plugin provides a simple interface to apply, test, compose, and save augmentation transformations to your dataset using the [Albumentations](https://albumentations.ai/docs/) library and the [FiftyOne](https://voxel51.com/docs/fiftyone/) library for data curation and visualization. Both libraries are open-source and easy to use!

## Supported Data Types

The plugin currently supports image media types. For images, it has support for the following label types:

- [Object Detection](https://docs.voxel51.com/user_guide/using_datasets.html#object-detection)
- [Keypoint Detection](https://docs.voxel51.com/user_guide/using_datasets.html#keypoints)
- [Instance Segmentation](https://docs.voxel51.com/user_guide/using_datasets.html#instance-segmentations)
- [Semantic Segmentation](https://docs.voxel51.com/user_guide/using_datasets.html#semantic-segmentation)
- [Heatmap](https://docs.voxel51.com/user_guide/using_datasets.html#heatmaps)


## Usage

### Setup

To get started, launch the [FiftyOne App](https://docs.voxel51.com/user_guide/app.html) and load in your dataset. If you don't have a dataset yet, you can use the [FiftyOne Dataset Zoo](https://docs.voxel51.com/user_guide/dataset_zoo/index.html) to load in a sample dataset. 

💡 To load the `quickstart` dataset from the zoo, press the backtick key (\`) to open the operators list, and select `load_zoo_dataset` from the list. Then, select `quickstart` from the dropdown and press `Enter`.

### Applying Augmentations

Once you have a dataset loaded, you can apply augmentations to your dataset using the `augment_with_albumentations` operator. 

#### Configuration Options

You will have a few options to configure:

- `Number of Augmentations per Sample`: The number of augmentations to apply to each sample. This will create a new sample for each augmentation. The default is 1.
- `Number of Transformations per Augmentation`: The number of transformations to compose into each augmentation. The default is 1. 
The input form will dynamically update to allow you to specify the parameters for each transformation.
- `Target View`: The view to apply the augmentations to. The default is `None`, which will apply the augmentations to the entire dataset. Here are the options you will be able to choose between, depending on what you are viewing and/or have selected:

    - `Dataset`: Apply the augmentations to the entire dataset.
    - `Current View`: Apply the augmentations to the samples in the current view.
    - `Selected Samples`: Apply the augmentations to the selected samples.
- `Execution Mode`: The execution mode to use when applying the augmentations. If you select `delegated=True`, the operation will be queued as a job, which you can then launch in the background from your terminal. The default is `delegated=False`, which will apply the augmentations immediately. For more information on FiftyOne's delegated operations, see [here](https://docs.voxel51.com/plugins/using_plugins.html#delegated-operations).


#### Choosing Transformations

Each of the transforms that is composed into the augmentation transformation can be selected from one of two sources:

- **Albumentations Primitives**: These are the transformations that are provided by the Albumentations library. You can find the full list of available transformations [here](https://albumentations.ai/docs/api_reference/augmentations/transforms/). All but a few of the transformations are supported by this plugin. When you select a transformation from this list, the input form will dynamically update to allow you to specify the parameters for the transformation. 💡 If you're curious, this is achieved through utilization of Python's `inspect` module, which provides access to docstrings and function signatures.
- **Saved Transformations**: You can also select from a list of your saved transformations. This is useful if you find a set of hyperparameters that you like and want to save them for later use. For more information on saving transformations, see [here](#saving-transformations).

### Get Info About Last Transformation

When you apply an augmentation to your dataset with `augment_with_albumentations`, the operation will store a serialized version of the transformation in a [custom run](https://docs.voxel51.com/plugins/developing_plugins.html#storing-custom-runs) on your dataset. You can view the last transformation that was applied to your dataset by using the `get_last_albumentations_run_info` operator. This will open an output modal with a formatted snapshot of the last transformation that was applied to your dataset.

### View Last Augmentation

In addition to storing the last transformation that was applied to your dataset, the `augment_with_albumentations` operation will retain a reference list of all of the samples that were created by the last augmentation that was applied to your dataset. You can view the last augmentation that was applied to your dataset by using the `view_last_albumentations_run` operator.

### Saving Transformations

If you find a set of hyperparameters that you like and want to save them for later use, you can save the transformation that was applied to your dataset by using the `save_albumentations_transformation` operator. You will be prompted to enter a name for the transformation — choose something that will help you remember what the transformation does.

Once you have saved the transformation, it will then be available to select from the `Saved Transformations` dropdown when you apply augmentations to your dataset.

💡 You can only save the last transformation that was applied to your dataset. If you want to save a transformation that you applied to your dataset in the past, you will need to re-apply the transformation to your dataset and then save it.

### Saving Augmentations

By default, each time a new augmentation is applied to your dataset, all of the samples that were created by the *previous* augmentation are removed from your dataset, and the temporary files that were created to store the samples are deleted.

If you find a set of augmentations that you like, you can persist them to your dataset by using the `save_albumentations_augmentations` operator.

### Get Info About Saved Transformations

You can view the transformations that have been saved to your dataset by using the `get_albumentations_run_info` operator and selecting the name of the transformation that you want info about from the dropdown.

### Delete Saved Transformations

You can delete a saved transformation from your dataset by using the `delete_albumentations_run` operator and selecting the name of the transformation that you want to delete from the dropdown.


## Installation

```shell
fiftyone plugins download https://github.com/jacobmarks/fiftyone-albumentations-plugin
```

You will also need to make sure that Albumentations is installed:

```shell
pip install albumentations
```