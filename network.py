import numpy as np
from layers import DenseLayer
from activations import relu, relu_backward, softmax
from losses import cross_entropy_loss, softmax_cross_entropy_backward


class Network:
    def __init__(self, layer_sizes):
        self.layers = [
            DenseLayer(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        ]
        # Cache pre-activation outputs for each hidden layer (needed by relu_backward)
        self.Z_cache = []

    def forward(self, X):
        self.Z_cache = []
        out = X
        # Hidden layers: linear → ReLU
        for layer in self.layers[:-1]:
            Z = layer.forward(out)
            self.Z_cache.append(Z)
            out = relu(Z)
        # Output layer: linear → softmax
        Z_last = self.layers[-1].forward(out)
        self.Z_cache.append(Z_last)
        return softmax(Z_last)

    def backward(self, probs, y_onehot):
        # Gradient from the combined softmax + cross-entropy
        dZ = softmax_cross_entropy_backward(probs, y_onehot)

        # Output layer backward (no activation backward — softmax was already
        # absorbed into softmax_cross_entropy_backward)
        dA = self.layers[-1].backward(dZ)

        # Hidden layers in reverse: ReLU backward, then dense backward
        for layer, Z in zip(reversed(self.layers[:-1]), reversed(self.Z_cache[:-1])):
            dZ = relu_backward(dA, Z)
            dA = layer.backward(dZ)

    def update(self, lr):
        for layer in self.layers:
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db


if __name__ == "__main__":
    from data_loader import load_mnist, one_hot

    np.random.seed(0)
    X_train, y_train, _, _ = load_mnist()
    y_train_oh = one_hot(y_train)

    # Small batch for the test
    X_batch = X_train[:64]
    y_batch = y_train_oh[:64]

    net = Network([784, 128, 64, 10])

    probs = net.forward(X_batch)
    loss_before = cross_entropy_loss(probs, y_batch)

    net.backward(probs, y_batch)
    net.update(lr=0.1)

    probs_after = net.forward(X_batch)
    loss_after = cross_entropy_loss(probs_after, y_batch)

    print(f"Loss before update: {loss_before:.4f}")
    print(f"Loss after update:  {loss_after:.4f}")
    assert loss_after < loss_before, "Loss did not decrease after one update"
    print("Check passed — loss decreased.")
