import os
import urllib.request
import numpy as np

MNIST_URL   = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
DATA_PATH   = os.path.join(os.path.dirname(__file__), "data", "mnist.npz")
EMNIST_PATH = os.path.join(os.path.dirname(__file__), "data", "emnist_digits.npz")


def _download_mnist():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print("Downloading MNIST (~11 MB)...")
        urllib.request.urlretrieve(MNIST_URL, DATA_PATH)
        print("Download complete.")


def one_hot(y, num_classes=10):
    """Convert a label array of shape (N,) to one-hot encoding (N, num_classes)."""
    out = np.zeros((y.shape[0], num_classes))
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def load_mnist():
    """
    Returns X_train, y_train, X_test, y_test where:
      X_train: (60000, 784) float64, pixels in [0, 1]
      y_train: (60000,)     int, labels 0-9
      X_test:  (10000, 784) float64, pixels in [0, 1]
      y_test:  (10000,)     int, labels 0-9
    """
    _download_mnist()
    with np.load(DATA_PATH) as f:
        X_train, y_train = f["x_train"], f["y_train"]
        X_test, y_test = f["x_test"], f["y_test"]

    X_train = X_train.reshape(-1, 784).astype(np.float64) / 255.0
    X_test  = X_test.reshape(-1, 784).astype(np.float64) / 255.0

    return X_train, y_train, X_test, y_test


def load_combined():
    """
    Returns MNIST + EMNIST digits merged into one training set.
      X_train: (300000, 784) float64  — 60k MNIST + 240k EMNIST
      y_train: (300000,)     int
      X_test:  (50000, 784)  float64  — 10k MNIST + 40k EMNIST
      y_test:  (50000,)      int
    EMNIST must already be downloaded to data/emnist_digits.npz.
    """
    X_m, y_m, X_mt, y_mt = load_mnist()

    if not os.path.exists(EMNIST_PATH):
        raise FileNotFoundError(
            f"EMNIST not found at {EMNIST_PATH}. "
            "Run the extraction script first."
        )
    with np.load(EMNIST_PATH) as f:
        X_e  = f["x_train"].reshape(-1, 784).astype(np.float64) / 255.0
        y_e  = f["y_train"].astype(np.int64)
        X_et = f["x_test"].reshape(-1, 784).astype(np.float64) / 255.0
        y_et = f["y_test"].astype(np.int64)

    X_train = np.concatenate([X_m,  X_e],  axis=0)
    y_train = np.concatenate([y_m,  y_e],  axis=0)
    X_test  = np.concatenate([X_mt, X_et], axis=0)
    y_test  = np.concatenate([y_mt, y_et], axis=0)

    return X_train, y_train, X_test, y_test


def ascii_preview(x, label):
    """Render a flat 784-vector as ASCII art and print its label."""
    chars = " .:-=+*#%@"
    img = (x * 255).reshape(28, 28)
    for row in img:
        print("".join(chars[int(p / 255 * (len(chars) - 1))] for p in row))
    print(f"Label: {label}")


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist()

    print(f"X_train: {X_train.shape}  |  dtype: {X_train.dtype}  |  range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}  |  dtype: {X_test.dtype}")
    print(f"y_test:  {y_test.shape}")

    y_train_oh = one_hot(y_train)
    y_test_oh = one_hot(y_test)
    print(f"\ny_train one-hot: {y_train_oh.shape}")
    print(f"y_test  one-hot: {y_test_oh.shape}")

    print("\n--- Sample digit (index 0) ---")
    ascii_preview(X_train[0], y_train[0])
