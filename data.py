import cv2
import numpy as np

SIZE = 128  # all images are resized to 128x128 before extracting features


def load_cifar10(per_class=200, root="data"):
    # downloads CIFAR-10 the first time (about 160 MB), then reuses it
    from torchvision.datasets import CIFAR10
    ds = CIFAR10(root, train=True, download=True)

    X, y = [], []
    count = [0] * 10
    for img, label in zip(ds.data, ds.targets):
        if count[label] == per_class:
            continue
        count[label] += 1
        X.append(cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (SIZE, SIZE)))
        y.append(label)
        if sum(count) == 10 * per_class:
            break
    return np.array(X), np.array(y), list(ds.classes)
