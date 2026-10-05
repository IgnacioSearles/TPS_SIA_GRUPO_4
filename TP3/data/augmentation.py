"""Label-preserving transformations of digit images, as pure NumPy.

Small shifts, rotations and rescalings of a handwritten digit are still the same
digit, so they can be used as extra training samples. Seeing a 7 in many
positions teaches the shape rather than the exact pixels it occupied.

Apply these to the **training half only**, after the split. Augmenting before
splitting would let a rotated copy of a training image land in validation, which
turns the validation score into a number that cannot be trusted.

`scipy` is deliberately not used: it is not in this project's requirements, and
the affine warp and bilinear sampling below are short enough to own outright.
Every function takes the flattened `(n, 784)` layout the loader produces, never
mutates its input, and keeps pixels inside [0, 1].
"""

import numpy as np

IMAGE_SIZE = 28
N_PIXELS = IMAGE_SIZE * IMAGE_SIZE


def _as_images(X: np.ndarray) -> np.ndarray:
    """Reshape flat rows into square images, failing loudly on the wrong width."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != N_PIXELS:
        raise ValueError(f"Expected images flattened to (n, {N_PIXELS}), got shape {X.shape}")
    return X.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)


def _bilinear_sample(images: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Read each image at fractional coordinates, blending the four neighbouring pixels.

    Coordinates outside the image read as 0, which is the digits' background, so a
    stroke pushed past the edge simply leaves the frame.
    """
    count = len(images)
    row_low, column_low = np.floor(rows).astype(int), np.floor(columns).astype(int)
    row_weight, column_weight = rows - row_low, columns - column_low
    batch = np.arange(count)[:, None, None]

    def pixel_at(row: np.ndarray, column: np.ndarray) -> np.ndarray:
        inside = (row >= 0) & (row < IMAGE_SIZE) & (column >= 0) & (column < IMAGE_SIZE)
        clipped_row = np.clip(row, 0, IMAGE_SIZE - 1)
        clipped_column = np.clip(column, 0, IMAGE_SIZE - 1)
        return images[batch, clipped_row, clipped_column] * inside

    top = pixel_at(row_low, column_low) * (1 - column_weight) + pixel_at(row_low, column_low + 1) * column_weight
    bottom = pixel_at(row_low + 1, column_low) * (1 - column_weight) + pixel_at(row_low + 1, column_low + 1) * column_weight
    return top * (1 - row_weight) + bottom * row_weight


def affine_warp(
    X: np.ndarray,
    angle_degrees: np.ndarray,
    scale: np.ndarray,
    shift: np.ndarray,
) -> np.ndarray:
    """Rotate, rescale and translate every image by its own parameters, around the image centre.

    `shift` is (n, 2) as (columns, rows) in pixels — positive moves right and down.
    Works backwards, as image resampling must: for each destination pixel it asks
    which source pixel lands there, so no output pixel is left unwritten.
    """
    images = _as_images(X)
    count = len(images)
    angle_degrees = np.asarray(angle_degrees, dtype=float).reshape(-1)
    scale = np.asarray(scale, dtype=float).reshape(-1)
    shift = np.asarray(shift, dtype=float).reshape(-1, 2)
    for name, values in (("angle_degrees", angle_degrees), ("scale", scale), ("shift", shift)):
        if len(values) != count:
            raise ValueError(f"'{name}' has {len(values)} entries for {count} images")
    if np.any(scale <= 0):
        raise ValueError("scale must be positive")

    centre = (IMAGE_SIZE - 1) / 2
    grid_rows, grid_columns = np.meshgrid(np.arange(IMAGE_SIZE), np.arange(IMAGE_SIZE), indexing="ij")
    destination_rows = grid_rows[None] - centre - shift[:, 1, None, None]
    destination_columns = grid_columns[None] - centre - shift[:, 0, None, None]

    radians = np.deg2rad(angle_degrees)[:, None, None]
    cosine, sine = np.cos(radians), np.sin(radians)
    inverse_scale = (1.0 / scale)[:, None, None]
    source_columns = (cosine * destination_columns + sine * destination_rows) * inverse_scale + centre
    source_rows = (-sine * destination_columns + cosine * destination_rows) * inverse_scale + centre

    warped = _bilinear_sample(images, source_rows, source_columns)
    return np.clip(warped.reshape(count, N_PIXELS), 0.0, 1.0)


def random_affine(
    X: np.ndarray,
    rng: np.random.Generator,
    max_rotation: float = 10.0,
    max_scale: float = 0.1,
    max_shift: float = 2.0,
) -> np.ndarray:
    """Give every image its own small random rotation, rescaling and shift.

    Defaults are deliberately gentle: past roughly 15 degrees or 3 pixels the
    strokes start leaving the frame, and a 1 that has been rotated far enough
    stops being a 1.
    """
    count = len(X)
    return affine_warp(
        X,
        angle_degrees=rng.uniform(-max_rotation, max_rotation, count),
        scale=rng.uniform(1 - max_scale, 1 + max_scale, count),
        shift=rng.uniform(-max_shift, max_shift, (count, 2)),
    )


def add_gaussian_noise(X: np.ndarray, rng: np.random.Generator, sigma: float = 0.1) -> np.ndarray:
    """Add independent Gaussian noise to every pixel, then clip back into [0, 1].

    Doubles as the robustness check the assignment offers as an optional: score the
    finished model on a test set noised at increasing sigma.
    """
    X = np.asarray(X, dtype=float)
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    if sigma == 0:
        return X.copy()
    return np.clip(X + rng.normal(0.0, sigma, X.shape), 0.0, 1.0)


def augment(
    X: np.ndarray,
    rng: np.random.Generator,
    max_rotation: float = 10.0,
    max_scale: float = 0.1,
    max_shift: float = 2.0,
    sigma: float = 0.0,
) -> np.ndarray:
    """One augmented copy of every image: geometry first, then noise.

    Geometry comes first so the noise is not smeared by the interpolation. Call
    this once per epoch with a fresh draw rather than building one fixed augmented
    dataset — the point is that the model rarely sees the same pixels twice.
    """
    X = _as_images(X).reshape(-1, N_PIXELS)
    count = len(X)
    # Draw all parameters first, then warp in chunks. This preserves the same
    # seeded transformations as random_affine while avoiding multi-gigabyte
    # temporary arrays for the expanded, balanced E3 training set.
    angles = rng.uniform(-max_rotation, max_rotation, count)
    scales = rng.uniform(1 - max_scale, 1 + max_scale, count)
    shifts = rng.uniform(-max_shift, max_shift, (count, 2))
    warped = np.empty_like(X)
    chunk_size = 1024
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        warped[start:stop] = affine_warp(X[start:stop], angles[start:stop],
                                         scales[start:stop], shifts[start:stop])
    return add_gaussian_noise(warped, rng, sigma=sigma)
