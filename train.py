import numpy as np
import matplotlib.pyplot as plt
from network import Network
from losses import cross_entropy_loss
from data_loader import load_mnist, one_hot


def train(network, X, y_onehot, epochs, batch_size, lr):
    N = X.shape[0]
    loss_history = []

    for epoch in range(1, epochs + 1):
        # Shuffle at the start of every epoch
        idx = np.random.permutation(N)
        X, y_onehot = X[idx], y_onehot[idx]

        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, N, batch_size):
            X_batch = X[start:start + batch_size]
            y_batch = y_onehot[start:start + batch_size]

            probs = network.forward(X_batch)
            epoch_loss += cross_entropy_loss(probs, y_batch)
            network.backward(probs, y_batch)
            network.update(lr)
            num_batches += 1

        mean_loss = epoch_loss / num_batches
        loss_history.append(mean_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3} | loss: {mean_loss:.4f}")

    return loss_history


if __name__ == "__main__":
    np.random.seed(0)

    X_train, y_train, _, _ = load_mnist()
    y_train_oh = one_hot(y_train)

    net = Network([784, 128, 64, 10])
    history = train(net, X_train, y_train_oh, epochs=20, batch_size=64, lr=0.1)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(history) + 1), history, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=120)
    plt.show()
    print("Loss curve saved to loss_curve.png")
