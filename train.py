import argparse

from data import load_cifar10
from utils import MODEL_DIR, train_all


def main():
    p = argparse.ArgumentParser(description="Train the 4 models on CIFAR-10 and save them")
    p.add_argument("--per-class", type=int, default=200, help="images used per class")
    p.add_argument("--out", default=MODEL_DIR)
    args = p.parse_args()

    X, y, classes = load_cifar10(args.per_class)
    rows = train_all(X, y, classes, args.out)

    print()
    for r in rows:
        print(f"{r['method']:15s} accuracy={r['accuracy']:.3f}  f1={r['f1']:.3f}  dim={r['dim']}")


if __name__ == "__main__":
    main()
