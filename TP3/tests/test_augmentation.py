import numpy as np
import pytest

from data.augmentation import (
    IMAGE_SIZE,
    add_gaussian_noise,
    affine_warp,
    augment,
    random_affine,
)


def digit_like(n=4, seed=0):
    """A few images with ink in the middle and a blank border, like the real digits."""
    rng = np.random.default_rng(seed)
    images = np.zeros((n, IMAGE_SIZE, IMAGE_SIZE))
    images[:, 8:20, 10:18] = rng.uniform(0.4, 1.0, size=(n, 12, 8))
    return images.reshape(n, -1)


def test_identity_transform_returns_the_image_untouched():
    X = digit_like()

    result = affine_warp(X, angle_degrees=np.zeros(4), scale=np.ones(4), shift=np.zeros((4, 2)))

    assert np.allclose(result, X, atol=1e-12)


def test_integer_shift_moves_the_ink_by_that_many_pixels():
    X = digit_like(n=1)

    result = affine_warp(X, np.zeros(1), np.ones(1), np.array([[1.0, 0.0]]))

    shifted = np.roll(X.reshape(1, IMAGE_SIZE, IMAGE_SIZE), 1, axis=2)
    # Column 0 wraps around in np.roll but is filled with background by the warp.
    assert np.allclose(result.reshape(1, IMAGE_SIZE, IMAGE_SIZE)[:, :, 1:], shifted[:, :, 1:])


def test_a_full_turn_is_the_same_image():
    X = digit_like()

    result = affine_warp(X, np.full(4, 360.0), np.ones(4), np.zeros((4, 2)))

    assert np.allclose(result, X, atol=1e-10)


def test_two_half_turns_return_the_original():
    X = digit_like()

    once = affine_warp(X, np.full(4, 180.0), np.ones(4), np.zeros((4, 2)))
    twice = affine_warp(once, np.full(4, 180.0), np.ones(4), np.zeros((4, 2)))

    assert not np.allclose(once, X)
    assert np.allclose(twice, X, atol=1e-10)


def test_warp_keeps_pixels_in_range_and_does_not_mutate_the_input():
    X = digit_like()
    original = X.copy()

    result = random_affine(X, np.random.default_rng(0))

    assert result.shape == X.shape
    assert result.min() >= 0.0 and result.max() <= 1.0
    assert np.array_equal(X, original)


def test_small_nudges_keep_most_of_the_ink():
    X = digit_like()

    result = random_affine(X, np.random.default_rng(1), max_rotation=10, max_scale=0.1, max_shift=2)

    # The digit moves, but it must not slide off the frame or dissolve.
    kept = result.sum(axis=1) / X.sum(axis=1)
    assert np.all(kept > 0.8), kept
    assert not np.allclose(result, X)


def test_augmentation_is_reproducible_from_the_seed():
    X = digit_like()

    first = random_affine(X, np.random.default_rng(5))
    second = random_affine(X, np.random.default_rng(5))

    assert np.array_equal(first, second)


def test_noise_raises_the_spread_of_a_flat_image():
    # A mid-grey image, so clipping at 0 and 1 does not distort what is measured.
    X = np.full((4, IMAGE_SIZE * IMAGE_SIZE), 0.5)

    noisy = add_gaussian_noise(X, np.random.default_rng(0), sigma=0.1)

    assert noisy.std() == pytest.approx(0.1, abs=0.01)
    assert noisy.mean() == pytest.approx(0.5, abs=0.01)


def test_noise_changes_every_image_but_keeps_shape_and_range():
    X = digit_like()

    noisy = add_gaussian_noise(X, np.random.default_rng(0), sigma=0.1)

    assert noisy.shape == X.shape
    assert noisy.min() >= 0.0 and noisy.max() <= 1.0
    assert not np.allclose(noisy, X)


def test_zero_noise_changes_nothing():
    X = digit_like()

    assert np.array_equal(add_gaussian_noise(X, np.random.default_rng(0), sigma=0.0), X)


def test_augment_combines_both_and_keeps_one_row_per_image():
    X = digit_like(n=7)

    result = augment(X, np.random.default_rng(0))

    assert result.shape == X.shape
    assert result.min() >= 0.0 and result.max() <= 1.0
    assert not np.allclose(result, X)


def test_augment_with_everything_off_is_the_identity():
    X = digit_like()

    result = augment(X, np.random.default_rng(0), max_rotation=0, max_scale=0, max_shift=0, sigma=0)

    assert np.allclose(result, X, atol=1e-12)


def test_warp_rejects_wrongly_shaped_input():
    with pytest.raises(ValueError, match="784"):
        affine_warp(np.zeros((2, 100)), np.zeros(2), np.ones(2), np.zeros((2, 2)))
