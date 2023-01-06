import torch
import torchvision
import wget
import hashlib
from torchvision.datasets import ImageFolder
import os
from typing import Optional,Callable,Tuple,Any
import shutil
import subprocess as sp
PARTS={"train","valid","test"}


def compute_sha256(filename,block_size=4096*16):
    sha256_hash = hashlib.sha256()
    with open(filename,"rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(block_size),b""):
            sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


class CINIC10(torchvision.datasets.vision.VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        partition (str, optional): One of train,valid,test, creates selects which partition to use.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "cinic-10-batches-py"
    url = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz"
    filename = "CINIC-10.tar.gz"
    tgz_sha256 ='31b095acf6d75e25a9e028bae82a07a0f94ff6b00671be2802d34ac4efa81a9e'
    classes=["airplane",
     "automobile",
     "bird",
     "cat",
     "deer",
     "dog",
     "frog",
     "horse",
     "ship",
     "truck"]


    def __init__(
        self,
        root: str= os.path.expanduser("~/.datasets/cinic10"),
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        shots: int = -1,
        seed: int = 0,
        preload: bool = False
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        assert split in PARTS, f"{split} not in {PARTS}"
        self.partition = split
        self.root = root
        self.num_classes = 10
        if download:
            self.download()

        #if not self._check_integrity():
        #    raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")
        self.data = ImageFolder(os.path.join(self.root, split))



    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        if not os.path.exists(self.download_path):
            print(f"Folder {self.root} does not exist")
            return False

        cksum=compute_sha256(self.download_path)

        if not cksum==CINIC10.tgz_sha256:
            print(f"{self.download_path} cksum {cksum} does not match {CINIC10.tgz_sha256}")
            return False
        else:
            return True

    def download(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        print(self.root)
        if self._check_integrity():
            print("Files already downloaded and verified")
        else:
            wget.download(self.url,out=self.download_path)
        if not all(os.path.exists(os.path.join(self.root,k)) for k in PARTS):
            cwd=os.path.abspath(os.curdir)
            os.chdir(self.root)
            sp.call(["tar","xf",CINIC10.filename])
            for p in ["train","valid","test"]:
                assert os.path.exists(p)
                assert os.path.exists(self.root)
                os.rename(p,os.path.join(self.root,p))
            os.chdir(cwd)

    def extra_repr(self) -> str:
        return f"Split: {self.partition}"

    @property
    def download_path(self):
        return os.path.join(self.root, CINIC10.filename)

if __name__=="__main__":
    ds=CINIC10("/tmp/")