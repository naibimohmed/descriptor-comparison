import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.cluster import MiniBatchKMeans


def f_hog(imgs):
    feats = []
    for img in imgs:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feats.append(hog(gray, orientations=9, pixels_per_cell=(16, 16),
                         cells_per_block=(2, 2)))
    return np.array(feats)


def f_lbp_color(imgs):
    # texture (LBP histogram) + color (HSV histogram) glued together
    feats = []
    for img in imgs:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, 8, 1, method="uniform")
        lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        col_hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4],
                                [0, 180, 0, 256, 0, 256]).flatten()
        col_hist = col_hist / (col_hist.sum() + 1e-9)

        feats.append(np.concatenate([lbp_hist, col_hist]))
    return np.array(feats)


class OrbBoVW:
    # ORB descriptors -> k-means vocabulary -> histogram of visual words
    def __init__(self, k=200):
        self.k = k
        self.orb = cv2.ORB_create(nfeatures=300, fastThreshold=5)

    # the cv2 ORB object can't be pickled, so rebuild it when loading a saved model
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["orb"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.orb = cv2.ORB_create(nfeatures=300, fastThreshold=5)

    def _orb(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, desc = self.orb.detectAndCompute(gray, None)
        return desc  # None if no keypoint found

    def fit(self, imgs):
        all_desc = [d for d in map(self._orb, imgs) if d is not None]
        bits = np.unpackbits(np.vstack(all_desc), axis=1).astype(np.float32)
        self.kmeans = MiniBatchKMeans(self.k, random_state=0, n_init=3,
                                      batch_size=2048).fit(bits)
        return self

    def transform(self, imgs):
        hists = np.zeros((len(imgs), self.k), np.float32)
        for i, img in enumerate(imgs):
            desc = self._orb(img)
            if desc is None:
                continue
            words = self.kmeans.predict(np.unpackbits(desc, axis=1).astype(np.float32))
            hists[i] = np.bincount(words, minlength=self.k)
        hists /= hists.sum(axis=1, keepdims=True) + 1e-9
        return hists
