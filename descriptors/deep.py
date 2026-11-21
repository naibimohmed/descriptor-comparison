import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
_model = None


def get_model():
    # load ResNet-50 only once, it's slow
    global _model
    if _model is None:
        import torch
        import torchvision
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        _model = torchvision.models.resnet50(weights=weights)
        _model.fc = torch.nn.Identity()  # drop the classifier, keep the 2048 features
        _model.eval()
    return _model


def f_cnn(imgs, batch_size=32):
    import torch
    model = get_model()
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), batch_size):
            chunk = imgs[i:i + batch_size]
            batch = np.stack([cv2.resize(cv2.cvtColor(x, cv2.COLOR_BGR2RGB), (224, 224))
                              for x in chunk]).astype(np.float32) / 255
            batch = (batch - MEAN) / STD
            batch = torch.from_numpy(batch.transpose(0, 3, 1, 2))
            out.append(model(batch).numpy())
    return np.vstack(out)
