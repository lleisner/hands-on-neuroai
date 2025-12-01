
import torch
from torchvision import datasets, transforms
from hands_on_neuroai.data.datasets import TaskDataset, PermutePixels, Rotate, ComposeTaskTransforms, generate_pixel_permutation

def test_permute_pixels_with_fakedata():
    img_size = (3, 32, 32)
    fake = datasets.FakeData(size=8, image_size=img_size, num_classes=10, transform=transforms.ToTensor())
    h, w = 32, 32
    num_pixels = h * w
    perm_transform = PermutePixels(seed=0)
    perm_ds = TaskDataset(fake, perm_transform)
    x0, y0 = fake[0]
    px0, py0 = perm_ds[0]
    assert x0.shape == px0.shape == torch.Size([3, h, w])
    assert y0 == py0
    flat = x0.view(3, -1)
    pflat = px0.view(3, -1)
    assert not torch.allclose(flat, pflat)

def test_compose_rotate_permute_with_fakedata():
    img_size = (1, 28, 28)
    fake = datasets.FakeData(size=4, image_size=img_size, num_classes=10, transform=transforms.ToTensor())
    rotate = Rotate(90)
    permute = PermutePixels(seed=42)
    composed = ComposeTaskTransforms([rotate, permute])
    composed_ds = TaskDataset(fake, composed)
    x0, y0 = fake[0]
    cx0, cy0 = composed_ds[0]
    assert x0.shape == cx0.shape
    assert y0 == cy0
    # Should be different after rotation+permutation
    assert not torch.allclose(x0, cx0)
