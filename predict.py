import numpy as np
from network import load_model
from data_loader import load_mnist, ascii_preview

LAYER_SIZES = [784, 128, 64, 10]
MODEL_PATH  = "saved_models/model.npz"


def predict(network, x):
    """Run inference on a single flat image vector (784,). Returns digit and all scores."""
    probs = network.forward(x.reshape(1, -1))
    digit = int(np.argmax(probs))
    return digit, probs[0]


if __name__ == "__main__":
    _, _, X_test, y_test = load_mnist()
    net = load_model(MODEL_PATH, LAYER_SIZES)

    # Predict on 5 samples and print results
    indices = [0, 1, 2, 3, 4]
    correct = 0

    for i in indices:
        digit, scores = predict(net, X_test[i])
        true_label = y_test[i]
        status = "✓" if digit == true_label else "✗"

        print(f"\n--- Sample {i} ---")
        ascii_preview(X_test[i], true_label)
        print(f"Predicted: {digit}  True: {true_label}  {status}")
        print("Confidence scores:")
        for cls, score in enumerate(scores):
            bar = "█" * int(score * 40)
            marker = " ← predicted" if cls == digit else ""
            print(f"  {cls}: {score:.4f}  {bar}{marker}")

        if digit == true_label:
            correct += 1

    print(f"\n{correct}/{len(indices)} correct on spot check.")
