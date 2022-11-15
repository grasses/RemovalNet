from torchaudio.datasets import SPEECHCOMMANDS
import os


class SpeechCommands(SPEECHCOMMANDS):
    def __init__(self, root, split: str = None, transform=None, target_transform=None, shots=-1, seed=0, preload=False):
        super().__init__(root, download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if split == "val":
            self._walker = load_list("validation_list.txt")
        elif split == "test":
            self._walker = load_list("testing_list.txt")
        elif split == "train":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]