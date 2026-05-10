import tkinter as tk
import numpy as np
from network import load_model

LAYER_SIZES = [784, 128, 64, 10]
MODEL_PATH  = "saved_models/model.npz"
GRID        = 28
CELL        = 14       # 14px per cell → 392px canvas (fits on any screen)
CANVAS_SIZE = GRID * CELL
BRUSH       = 1


class DrawDemo:
    def __init__(self, root, network):
        self.network = network
        self.pixels = np.zeros((GRID, GRID))

        root.title("Draw a digit")
        root.resizable(False, False)

        # ── Left column: canvas + buttons + result ───────────────────────────
        left = tk.Frame(root)
        left.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="n")

        self.canvas = tk.Canvas(left, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="black", cursor="crosshair")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<Button-1>", self._on_drag)

        btn_row = tk.Frame(left)
        btn_row.pack(pady=6)
        tk.Button(btn_row, text="Predict", font=("Helvetica", 13, "bold"),
                  command=self._predict, bg="#4CAF50", fg="white",
                  width=9).pack(side="left", padx=4)
        tk.Button(btn_row, text="Clear", font=("Helvetica", 13),
                  command=self._clear, bg="#f44336", fg="white",
                  width=9).pack(side="left", padx=4)

        self.result_var = tk.StringVar(value="Draw a digit, then\nclick Predict.")
        tk.Label(left, textvariable=self.result_var,
                 font=("Helvetica", 15, "bold"), justify="center",
                 width=22).pack(pady=(4, 0))

        # ── Right column: confidence bars ────────────────────────────────────
        right = tk.Frame(root)
        right.grid(row=0, column=1, padx=(5, 10), pady=10, sticky="n")

        tk.Label(right, text="Confidence", font=("Helvetica", 12, "bold")
                 ).pack(anchor="w", pady=(4, 6))

        self.bar_labels   = []
        self.bar_canvases = []
        BAR_W = 180

        for i in range(10):
            row_f = tk.Frame(right)
            row_f.pack(fill="x", pady=2)
            tk.Label(row_f, text=str(i), width=2,
                     font=("Helvetica", 12, "bold")).pack(side="left")
            c = tk.Canvas(row_f, width=BAR_W, height=18, bg="#e8e8e8",
                          highlightthickness=0)
            c.pack(side="left", padx=4)
            lbl = tk.Label(row_f, text="", width=6,
                           font=("Helvetica", 11), anchor="w")
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
            self.result_var.set("Canvas is empty —\ndraw something first.")
            return
        probs = self.network.forward(x.reshape(1, -1))[0]
        digit = int(np.argmax(probs))
        conf  = probs[digit] * 100
        self.result_var.set(f"Prediction:  {digit}\n{conf:.1f}% confidence")
        self._update_bars(probs, digit)

    def _clear(self):
        self.pixels[:] = 0
        self.canvas.delete("all")
        self.result_var.set("Draw a digit, then\nclick Predict.")
        for (c, w), lbl in zip(self.bar_canvases, self.bar_labels):
            c.delete("all")
            lbl.config(text="")

    def _update_bars(self, probs, predicted):
        for i, ((c, w), lbl) in enumerate(zip(self.bar_canvases, self.bar_labels)):
            c.delete("all")
            bar_w = int(probs[i] * w)
            colour = "#4CAF50" if i == predicted else "#2196F3"
            if bar_w > 0:
                c.create_rectangle(0, 0, bar_w, 18, fill=colour, outline="")
            lbl.config(text=f"{probs[i]*100:5.1f}%")


if __name__ == "__main__":
    net = load_model(MODEL_PATH, LAYER_SIZES)
    root = tk.Tk()
    DrawDemo(root, net)
    root.mainloop()
