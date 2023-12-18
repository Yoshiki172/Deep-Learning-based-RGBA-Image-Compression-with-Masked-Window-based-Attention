import fiftyone

if __name__ == '__main__':
    """Download the training/test data set from OpenImages."""

    dataset_train = fiftyone.zoo.load_zoo_dataset(
        "open-images-v6",
        split="train",
        max_samples=300000,
        label_types=["segmentations"],
        dataset_dir='openimages',
    )
