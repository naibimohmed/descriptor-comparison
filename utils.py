import json
import os
import time

import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from data import SIZE
from descriptors import OrbBoVW, f_cnn, f_hog, f_lbp_color

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "cifar10")
METHODS = ["HOG", "LBP + HSV", "ORB + BoVW", "CNN ResNet-50"]
FUNCS = {"HOG": f_hog, "LBP + HSV": f_lbp_color, "CNN ResNet-50": f_cnn}


def file_name(method):
    return method.replace(" + ", "_").replace(" ", "").replace("-", "")


def get_extractor(method, bovw=None):
    if method == "ORB + BoVW":
        return bovw.transform
    return FUNCS[method]


def train_method(method, Xtr, Xte, ytr, yte):
    t = time.time()
    bovw = OrbBoVW().fit(Xtr) if method == "ORB + BoVW" else None
    extract = get_extractor(method, bovw)
    Ftr, Fte = extract(Xtr), extract(Xte)
    t_feat = time.time() - t

    clf = make_pipeline(StandardScaler(), LinearSVC(C=0.1, max_iter=5000))
    t = time.time()
    clf.fit(Ftr, ytr)
    t_fit = time.time() - t

    pred = clf.predict(Fte)
    row = {
        "method": method,
        "dim": int(Ftr.shape[1]),
        "accuracy": float(accuracy_score(yte, pred)),
        "f1": float(f1_score(yte, pred, average="macro")),
        "t_feat": t_feat,
        "t_fit": t_fit,
    }
    return row, confusion_matrix(yte, pred), bovw, clf


def train_all(X, y, classes, out_dir=MODEL_DIR, methods=METHODS, log=print, progress=None):
    os.makedirs(out_dir, exist_ok=True)
    train, test = train_test_split(np.arange(len(X)), test_size=0.3, stratify=y, random_state=0)
    Xtr, Xte, ytr, yte = X[train], X[test], y[train], y[test]
    log(f"{len(X)} images, train {len(Xtr)} / test {len(Xte)}")

    rows = []
    for i, method in enumerate(methods):
        log("-> " + method)
        row, cm, bovw, clf = train_method(method, Xtr, Xte, ytr, yte)
        joblib.dump({"clf": clf, "bovw": bovw}, os.path.join(out_dir, file_name(method) + ".joblib"))
        np.save(os.path.join(out_dir, "cm_" + file_name(method) + ".npy"), cm)
        rows.append(row)
        log(f"   accuracy {row['accuracy']:.3f}, f1 {row['f1']:.3f}")
        if progress:
            progress(int(100 * (i + 1) / len(methods)))

    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump({"classes": classes, "results": rows}, f, indent=1)
    # keep some test images for the "random test image" button
    np.savez_compressed(os.path.join(out_dir, "samples.npz"), X=Xte[:60], y=yte[:60])
    log("Models saved in " + out_dir)
    return rows


def load_models(folder=MODEL_DIR):
    # returns (info, models, matrices), or None if nothing was trained yet
    path = os.path.join(folder, "info.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        info = json.load(f)

    models, matrices = {}, {}
    for row in info["results"]:
        method = row["method"]
        saved = joblib.load(os.path.join(folder, file_name(method) + ".joblib"))
        models[method] = (get_extractor(method, saved["bovw"]), saved["clf"])
        matrices[method] = np.load(os.path.join(folder, "cm_" + file_name(method) + ".npy"))
    return info, models, matrices


def predict_all(models, classes, img):
    # img is a BGR image of any size; returns {method: class name}
    small = np.array([cv2.resize(img, (SIZE, SIZE))])
    out = {}
    for method, (extract, clf) in models.items():
        try:
            out[method] = classes[int(clf.predict(extract(small))[0])]
        except Exception as e:
            out[method] = "error: " + str(e)
    return out
