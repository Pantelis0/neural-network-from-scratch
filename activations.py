import numpy as np


def relu(Z):
    return np.maximum(0, Z)


def relu_backward(dA, Z):
    # Gradient flows only where Z was positive; zero elsewhere
    return dA * (Z > 0)


def softmax(Z):
    # Subtract row-wise max for numerical stability before exponentiating
    Z = Z - Z.max(axis=1, keepdims=True)
    exp = np.exp(Z)
    return exp / exp.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    Z = np.array([-1.0, 0.0, 2.0])

    out = relu(Z)
    print(f"relu([-1, 0, 2])          → {out}")
    assert np.array_equal(out, [0.0, 0.0, 2.0]), f"relu failed: {out}"

    dA = np.ones(3)
    grad = relu_backward(dA, Z)
    print(f"relu_backward(ones, Z)    → {grad}")
    assert np.array_equal(grad, [0.0, 0.0, 1.0]), f"relu_backward failed: {grad}"

    print("All checks passed.")
