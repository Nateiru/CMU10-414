from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        self.transforms = transforms

        import gzip
        import struct
        def parse_mnist(image_filesname, label_filename):
            with gzip.open(image_filesname, 'rb') as f:
                img_magic, img_num, img_w, img_h = struct.unpack('>IIII', f.read(16))
                imgs = np.frombuffer(f.read(img_num * img_h * img_w), dtype=np.uint8).reshape(img_num, img_w*img_h).astype(np.float32)/255
            with gzip.open(label_filename, 'rb') as f:
                labels_magic, labels_num = struct.unpack('>II', f.read(8))
                labels = np.frombuffer(f.read(labels_num), dtype=np.uint8)
            return imgs, labels

        self.X, self.y = parse_mnist(image_filename, label_filename)

    def __getitem__(self, index) -> object:
        # 对一个样本进行 transform
        x = self.apply_transforms(self.X[index].reshape(28, 28, -1))
        return x.reshape(-1, 28*28), self.y[index]

    def __len__(self) -> int:
        return self.X.shape[0]
