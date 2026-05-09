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

    def backward(self, dZ):
        N = dZ.shape[0]
        self.dW = self.X_cache.T @ dZ / N
        self.db = dZ.mean(axis=0, keepdims=True)
        return dZ @ self.W.T  # dA_prev — passed to the layer before this one


if __name__ == "__main__":
    np.random.seed(0)
    layer = DenseLayer(784, 128)
    X = np.random.randn(32, 784)
    Z = layer.forward(X)

    # Forward shape check
    assert Z.shape == (32, 128), f"Expected (32, 128), got {Z.shape}"

    # Backward shape check — simulate a gradient flowing back from the next layer
    dZ = np.random.randn(32, 128)
    dA_prev = layer.backward(dZ)

    print(f"W shape:      {layer.W.shape}  |  dW shape: {layer.dW.shape}")
    print(f"b shape:      {layer.b.shape}  |  db shape: {layer.db.shape}")
    print(f"X shape:      {X.shape}  |  dA_prev shape: {dA_prev.shape}")

    assert layer.dW.shape == layer.W.shape,   f"dW mismatch: {layer.dW.shape} vs {layer.W.shape}"
    assert layer.db.shape == layer.b.shape,   f"db mismatch: {layer.db.shape} vs {layer.b.shape}"
    assert dA_prev.shape  == X.shape,         f"dA_prev mismatch: {dA_prev.shape} vs {X.shape}"

    print("All shape checks passed.")
