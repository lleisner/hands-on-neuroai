import torch
from torchvision import datasets, transforms
from hands_on_neuroai.data.datasets import (
    make_permuted_task, make_rotated_task, build_task_splits, get_image_shape, DatasetConfig
)

def test_build_task_splits():
    # Use FakeData for fast testing
    img_size = (1, 16, 16)
    fake = datasets.FakeData(size=50, image_size=img_size, num_classes=5, transform=transforms.ToTensor())
    # Patch get_datasets to return our fake dataset for this test
    import hands_on_neuroai.data.datasets as dsmod
    orig_get_datasets = dsmod.get_datasets
    dsmod.get_datasets = lambda *a, **kw: (fake, fake)

    h, w = 16, 16
    num_pixels = h * w
    task_transforms = [make_permuted_task(num_pixels, (h, w), seed=s) for s in range(3)]
    train_dsets, val_dsets, test_dsets = build_task_splits(
        'fake', DatasetConfig(), task_transforms, val_ratio=0.2
    )
    # Check lengths
    assert len(train_dsets) == len(val_dsets) == len(test_dsets) == 3
    # Check split sizes
    assert len(train_dsets[0]) == 40  # 80% of 50
    assert len(val_dsets[0]) == 10    # 20% of 50
    assert len(test_dsets[0]) == 50
    # Check transform application (permutation changes pixel order)
    x0, y0 = fake[0]
    px0, py0 = train_dsets[0][0]
    assert not torch.allclose(x0.view(-1), px0.view(-1))
    # Restore original get_datasets
    dsmod.get_datasets = orig_get_datasets
