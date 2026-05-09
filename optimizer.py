import numpy as np


class Adam:
    """
    Adam optimizer (Kingma & Ba, 2014).

    Maintains a first-moment (mean) and second-moment (uncentered variance)
    estimate for every parameter. Bias-correction in the first steps prevents
    the estimates from being pulled toward zero before they've warmed up.

    Default hyperparameters match the paper: lr=0.001, b1=0.9, b2=0.999, eps=1e-8.
    """

    def __init__(self, network, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.network = network
        self.lr      = lr
        self.beta1   = beta1
        self.beta2   = beta2
        self.eps     = eps
        self.t       = 0  # timestep — incremented each step for bias correction

        # Initialise moment estimates to zero for every W and b in every layer
        self.m = [{"W": np.zeros_like(l.W), "b": np.zeros_like(l.b)}
                  for l in network.layers]
        self.v = [{"W": np.zeros_like(l.W), "b": np.zeros_like(l.b)}
                  for l in network.layers]

    def step(self):
        self.t += 1
        t, b1, b2, eps, lr = self.t, self.beta1, self.beta2, self.eps, self.lr

        # Bias-correction factors grow toward 1 as t increases
        bc1 = 1 - b1 ** t
        bc2 = 1 - b2 ** t

        for i, layer in enumerate(self.network.layers):
            for key, grad in (("W", layer.dW), ("b", layer.db)):
                # Update biased moment estimates
                self.m[i][key] = b1 * self.m[i][key] + (1 - b1) * grad
                self.v[i][key] = b2 * self.v[i][key] + (1 - b2) * grad ** 2

                # Correct bias
                m_hat = self.m[i][key] / bc1
                v_hat = self.v[i][key] / bc2

                # Parameter update
                if key == "W":
                    layer.W -= lr * m_hat / (np.sqrt(v_hat) + eps)
                else:
                    layer.b -= lr * m_hat / (np.sqrt(v_hat) + eps)
