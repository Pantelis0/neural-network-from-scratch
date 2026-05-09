import numpy as np


class DenseLayer:
    def __init__(self, n_in, n_out):
        # He initialisation: keeps variance stable through deep ReLU networks
        self.W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
        self.b = np.zeros((1, n_out))
        self.X_cache = None
        self.dW = None
        self.db = None

    def forward(self, X):
        self.X_cache = X
        return X @ self.W + self.b


if __name__ == "__main__":
    np.random.seed(0)
    layer = DenseLayer(784, 128)
    X = np.random.randn(32, 784)
    Z = layer.forward(X)
    print(f"W shape:      {layer.W.shape}")
    print(f"b shape:      {layer.b.shape}")
    print(f"Input shape:  {X.shape}")
    print(f"Output shape: {Z.shape}")
    assert Z.shape == (32, 128), f"Expected (32, 128), got {Z.shape}"
    print("Shape check passed.")
