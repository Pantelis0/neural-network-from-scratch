import tkinter as tk
import numpy as np
from network import load_model

LAYER_SIZES = [784, 128, 64, 10]
MODEL_PATH  = "saved_models/model.npz"
GRID        = 28       # internal image resolution
CELL        = 16       # pixels per grid cell on screen
CANVAS_SIZE = GRID * CELL  # 448px canvas
BRUSH       = 1        # radius in grid cells


class DrawDemo:
    def __init__(self, root, network):
        self.network = network
        self.pixels = np.zeros((GRID, GRID))  # internal 28x28 image

        root.title("Draw a digit")
        root.resizable(False, False)

        # ── Canvas ──────────────────────────────────────────────────────────
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="black", cursor="crosshair")
        self.canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<Button-1>", self._on_drag)

        # ── Buttons ─────────────────────────────────────────────────────────
        tk.Button(root, text="Predict", font=("Helvetica", 14, "bold"),
                  command=self._predict, bg="#4CAF50", fg="white",
                  width=10).grid(row=1, column=0, padx=5, pady=8)

        tk.Button(root, text="Clear", font=("Helvetica", 14),
                  command=self._clear, bg="#f44336", fg="white",
                  width=10).grid(row=1, column=1, padx=5, pady=8)

        # ── Result panel ─────────────────────────────────────────────────────
        self.result_var = tk.StringVar(value="Draw a digit, then click Predict.")
        tk.Label(root, textvariable=self.result_var,
                 font=("Helvetica", 16), width=36,
                 anchor="w").grid(row=2, column=0, columnspan=3, padx=10)

        self.bar_frame = tk.Frame(root)
        self.bar_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=6)
        self.bar_labels = []
        self.bar_canvases = []
        BAR_W = 220
        for i in range(10):
            row_f = tk.Frame(self.bar_frame)
            row_f.pack(fill="x", pady=1)
            tk.Label(row_f, text=str(i), width=2,
                     font=("Helvetica", 11)).pack(side="left")
            c = tk.Canvas(row_f, width=BAR_W, height=14, bg="#e0e0e0",
                          highlightthickness=0)
            c.pack(side="left", padx=4)
            lbl = tk.Label(row_f, text="", width=7,
                           font=("Helvetica", 10), anchor="w")
            lbl.pack(side="left")
            self.bar_canvases.append((c, BAR_W))
            self.bar_labels.append(lbl)

    # ── Drawing ──────────────────────────────────────────────────────────────

    def _on_drag(self, event):
        gx = event.x // CELL
        gy = event.y // CELL
        for dy in range(-BRUSH, BRUSH + 1):
            for dx in range(-BRUSH, BRUSH + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < GRID and 0 <= ny < GRID:
                    # Softer at the edges of the brush
                    strength = 1.0 if abs(dx) + abs(dy) <= BRUSH else 0.5
                    self.pixels[ny, nx] = min(1.0, self.pixels[ny, nx] + strength)
                    self._draw_cell(nx, ny)

    def _draw_cell(self, gx, gy):
        x0, y0 = gx * CELL, gy * CELL
        x1, y1 = x0 + CELL, y0 + CELL
        v = int(self.pixels[gy, gx] * 255)
        colour = f"#{v:02x}{v:02x}{v:02x}"
        self.canvas.create_rectangle(x0, y0, x1, y1, fill=colour, outline="")

    # ── Actions ──────────────────────────────────────────────────────────────

    def _predict(self):
        x = self.pixels.flatten()
        if x.max() == 0:
            self.result_var.set("Canvas is empty — draw something first.")
            return
        probs = self.network.forward(x.reshape(1, -1))[0]
        digit = int(np.argmax(probs))
        conf  = probs[digit] * 100
        self.result_var.set(f"Prediction: {digit}   ({conf:.1f}% confidence)")
        self._update_bars(probs, digit)

    def _clear(self):
        self.pixels[:] = 0
        self.canvas.delete("all")
        self.result_var.set("Draw a digit, then click Predict.")
        for (c, w), lbl in zip(self.bar_canvases, self.bar_labels):
            c.delete("all")
            lbl.config(text="")

    def _update_bars(self, probs, predicted):
        for i, ((c, w), lbl) in enumerate(zip(self.bar_canvases, self.bar_labels)):
            c.delete("all")
            bar_w = int(probs[i] * w)
            colour = "#4CAF50" if i == predicted else "#2196F3"
            if bar_w > 0:
                c.create_rectangle(0, 0, bar_w, 14, fill=colour, outline="")
            lbl.config(text=f"{probs[i]*100:5.1f}%")


if __name__ == "__main__":
    net = load_model(MODEL_PATH, LAYER_SIZES)
    root = tk.Tk()
    DrawDemo(root, net)
    root.mainloop()
