"""Loading and visualization helpers for the handwritten digit CSV files."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_digit_arrays(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized 28x28 images as rows and integer labels."""
    frame = pd.read_csv(path)
    if list(frame.columns) != ["label", "image"]:
        raise ValueError(f"{path}: expected exactly the columns label,image")
    if frame.empty or frame.isna().any().any():
        raise ValueError(f"{path}: empty dataset or missing values")
    labels = frame["label"].to_numpy()
    if not np.isin(labels, np.arange(10)).all() or not np.equal(labels, labels.astype(int)).all():
        raise ValueError(f"{path}: labels must be integers from 0 to 9")
    images = []
    for row, raw in enumerate(frame["image"]):
        image = np.fromstring(raw.strip().removeprefix("[").removesuffix("]"), sep=",", dtype=np.float32)
        if image.size != 784 or not np.isfinite(image).all():
            raise ValueError(f"{path}: row {row} must contain 784 finite pixels")
        images.append(image)
    X = np.stack(images)
    if X.min() < 0 or X.max() > 1:
        raise ValueError(f"{path}: pixels must be normalized to [0, 1]")
    return X, labels.astype(np.int64)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Keep the original DataFrame API with each image as a NumPy array."""
    X, y = load_digit_arrays(path)
    return pd.DataFrame({"label": y, "image": list(X)})


def get_image(row: pd.Series, size: tuple[int, int] = (28, 28)) -> np.ndarray:
    """Reshape the flat image vector back to a 2-D array."""
    return row["image"].reshape(size)


def plot_sample(row: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(get_image(row), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Label: {int(row['label'])}", fontsize=13)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_dataset("datasets/digits_test.csv")
    plot_sample(df.iloc[0])
