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


def save_model(network, path):
    arrays = {}
    for i, layer in enumerate(network.layers):
        arrays[f"W{i}"] = layer.W
        arrays[f"b{i}"] = layer.b
    np.savez(path, **arrays)


def load_model(path, layer_sizes):
    network = Network(layer_sizes)
    data = np.load(path)
    for i, layer in enumerate(network.layers):
        layer.W = data[f"W{i}"]
        layer.b = data[f"b{i}"]
    return network


if __name__ == "__main__":
    from data_loader import load_mnist, one_hot
    from train import train, evaluate
    import os

    np.random.seed(0)
    X_train, y_train, X_test, y_test = load_mnist()
    y_train_oh = one_hot(y_train)
    y_test_oh  = one_hot(y_test)

    # Train a small network quickly just to get non-random weights
    net = Network([784, 128, 64, 10])
    train(net, X_train, y_train_oh, epochs=5, batch_size=64, lr=0.1)

    # Predictions before save
    sample = X_test[:10]
    probs_original = net.forward(sample)

    # Save
    save_path = "saved_models/model.npz"
    save_model(net, save_path)
    print(f"Model saved to {save_path}")

    # Load into a fresh network
    net2 = load_model(save_path, [784, 128, 64, 10])
    probs_loaded = net2.forward(sample)

    # Verify bit-for-bit identical
    assert np.array_equal(probs_original, probs_loaded), "Predictions differ after reload!"
    print("Round-trip check passed — predictions identical before and after save/load.")
    print(f"Predicted digits: {np.argmax(probs_loaded, axis=1)}")
    print(f"True labels:      {y_test[:10]}")
