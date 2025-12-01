import torch
from torchvision import datasets, transforms
from hands_on_neuroai.data.datasets import (
    make_permuted_task, make_rotated_task, make_split_task,
    build_task_datasets, get_image_shape
)

def test_permuted_task():
    img_size = (3, 16, 16)
    fake = datasets.FakeData(size=10, image_size=img_size, num_classes=10, transform=transforms.ToTensor())
    h, w = 16, 16
    num_pixels = h * w
    perm_task = make_permuted_task(num_pixels, (h, w), seed=42)
    ds = build_task_datasets('fake', type('C', (), {'root': '', 'num_workers': 0, 'pin_memory': False, 'download': False})(), [perm_task])[0][0]
    x, y = ds[0]
    assert x.shape == torch.Size([3, h, w])


def test_rotated_task():
    img_size = (1, 28, 28)
    fake = datasets.FakeData(size=10, image_size=img_size, num_classes=10, transform=transforms.ToPILImage())
    rot_task = make_rotated_task(45)
    ds = build_task_datasets('fake', type('C', (), {'root': '', 'num_workers': 0, 'pin_memory': False, 'download': False})(), [rot_task])[0][0]
    x, y = ds[0]
    assert x.shape == torch.Size([1, 28, 28]) or x.shape == torch.Size([28, 28])


def test_split_task():
    img_size = (1, 28, 28)
    fake = datasets.FakeData(size=10, image_size=img_size, num_classes=10, transform=transforms.ToTensor())
    split_task = make_split_task([0, 1, 2])
    ds = build_task_datasets('fake', type('C', (), {'root': '', 'num_workers': 0, 'pin_memory': False, 'download': False})(), [split_task])[0][0]
    for i in range(len(ds)):
        x, y = ds[i]
        assert y in [0, 1, 2]


def test_composed_task():
    img_size = (1, 28, 28)
    fake = datasets.FakeData(size=10, image_size=img_size, num_classes=10, transform=transforms.ToTensor())
    h, w = 28, 28
    num_pixels = h * w
    perm_task = make_permuted_task(num_pixels, (h, w), seed=1)
    rot_task = make_rotated_task(90)
    composed = lambda: [rot_task, perm_task]
    ds = build_task_datasets('fake', type('C', (), {'root': '', 'num_workers': 0, 'pin_memory': False, 'download': False})(), [rot_task, perm_task])[0][0]
    x, y = ds[0]
    assert x.shape == torch.Size([1, 28, 28])
