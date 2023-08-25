import csv
import torch
import pathlib
from typing import Any, Callable, Optional, Tuple
import PIL
from torchvision.datasets.folder import make_dataset, has_file_allowed_extension
from torchvision.datasets.vision import VisionDataset
import os.path as osp


class TSRD(VisionDataset):
    """`German Traffic Sign Recognition Benchmark (GTSRB) <https://benchmark.ini.rub.de/>`_ Dataset.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), or ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        shots: int = -1,
        seed: int = 0,
        preload: bool = False
    ) -> None:
        root = osp.join(root, "GTSRB_annotation")
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = split
        self._base_folder = pathlib.Path(root)
        self._target_folder = (
            self._base_folder / ("train" if self._split == "train" else "test")
        )
        self.num_classes = 43
        print(f"-> Load TSRD dataset from:{self._target_folder}")
        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        samples = self.make_dataset(str(self._target_folder), extensions=(".png"))
        self._samples = samples
        self.transform = transform
        self.target_transform = target_transform

    def make_dataset(self, directory, extensions=(".png")):
        import os
        instances = []
        files = os.listdir(directory)
        for f in files:
            if has_file_allowed_extension(f, extensions):
                path = os.path.join(directory, f)
                item = path, int(f.split("_")[0])
                instances.append(item)
        return instances

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self._samples[index]
        sample = PIL.Image.open(path).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def _check_exists(self) -> bool:
        return self._target_folder.is_dir()


if __name__ == "__main__":
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = GTSRB(root="../data/GTSRB", split="test", transform=transform)
    print(len(dataset))

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=4)
    for x, y in data_loader:
        print(x.shape, y)
        exit(1)