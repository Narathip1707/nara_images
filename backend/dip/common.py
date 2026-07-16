"""Helpers shared by every dip module.

Conventions follow the lecture notebooks (`slide Aj`):
  - images are numpy uint8, gray = (rows, cols), color = (rows, cols, 3) in BGR
  - indexing is [y, x] i.e. [row, col]
"""
import numpy as np

LUMA = (0.2989, 0.5870, 0.1140)  # R, G, B weights taught in Gray_level_image_Processing


def is_color(img):
    return img.ndim == 3


def to_gray(img):
    """RGB -> gray by the linear equation. Input BGR (cv2.imread order)."""
    if not is_color(img):
        return img
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    gray = LUMA[0] * r + LUMA[1] * g + LUMA[2] * b
    return np.clip(gray, 0, 255).astype(np.uint8)


def u8(arr):
    return np.clip(arr, 0, 255).astype(np.uint8)


def calc_hist(img):
    """256-bin histogram of a gray image (manual count, as taught)."""
    return np.bincount(img.ravel(), minlength=256).astype(np.int64)


def per_channel(img, fn):
    """Apply a gray->gray fn to each channel of a color image."""
    if not is_color(img):
        return fn(img)
    return np.dstack([fn(img[:, :, c]) for c in range(img.shape[2])])


class Steps:
    """Collects the step-by-step trace shown in the 'คำนวณทีละขั้น' panel.

    Every item is JSON-serializable so it can go straight to the frontend.
    """

    def __init__(self, title, formula=None):
        self.title = title
        self.formula = formula
        self.items = []

    def text(self, body, label=None):
        self.items.append({"type": "text", "label": label, "body": str(body)})
        return self

    def value(self, label, val, note=None):
        self.items.append({"type": "value", "label": label, "value": val, "note": note})
        return self

    def matrix(self, label, m, note=None):
        self.items.append({
            "type": "matrix", "label": label, "note": note,
            "rows": [[round(float(v), 4) for v in row] for row in np.asarray(m)],
        })
        return self

    def table(self, label, columns, rows, highlight=None, note=None):
        self.items.append({
            "type": "table", "label": label, "columns": columns,
            "rows": rows, "highlight": highlight, "note": note,
        })
        return self

    def chart(self, label, series, marker=None, note=None):
        """series: {name: [y values]} plotted against index."""
        self.items.append({
            "type": "chart", "label": label, "note": note, "marker": marker,
            "series": {k: [float(v) for v in vs] for k, vs in series.items()},
        })
        return self

    def to_dict(self):
        return {"title": self.title, "formula": self.formula, "items": self.items}
