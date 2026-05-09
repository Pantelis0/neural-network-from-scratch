import numpy as np


def cross_entropy_loss(probs, y_onehot):
    N = probs.shape[0]
    # Clip to avoid log(0)
    probs = np.clip(probs, 1e-15, 1.0)
    return -np.sum(y_onehot * np.log(probs)) / N


def softmax_cross_entropy_backward(probs, y_onehot):
    # Combined gradient of softmax + cross-entropy: (probs - y) / N
    # Derivation: dL/dZ = probs - y_onehot (the softmax cancels cleanly with
    # the cross-entropy derivative, leaving this compact form)
    N = probs.shape[0]
    return (probs - y_onehot) / N


if __name__ == "__main__":
    from activations import softmax
    from data_loader import one_hot

    np.random.seed(42)

    # Simulate an untrained network: random logits, 1000 samples, 10 classes
    # Large N gives a stable estimate; with only 32 samples variance is too high
    N, C = 1000, 10
    Z = np.random.randn(N, C)
    y = np.random.randint(0, C, size=N)
    y_oh = one_hot(y)

    probs = softmax(Z)

    # Sanity check: each row sums to 1
    assert np.allclose(probs.sum(axis=1), 1.0), "Softmax rows don't sum to 1"

    loss = cross_entropy_loss(probs, y_oh)
    print(f"Loss on random logits: {loss:.4f}  (expected ≈ {np.log(C):.4f})")
    assert abs(loss - np.log(C)) < 0.5, f"Loss {loss:.4f} too far from ln(10)={np.log(C):.4f}"

    dZ = softmax_cross_entropy_backward(probs, y_oh)
    print(f"dZ shape: {dZ.shape}  (expected ({N}, {C}))")
    assert dZ.shape == (N, C), f"Wrong dZ shape: {dZ.shape}"

    print("All checks passed.")
