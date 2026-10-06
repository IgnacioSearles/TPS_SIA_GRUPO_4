import pytest

from training.schedules import learning_rate_at


def test_no_schedule_keeps_the_learning_rate():
    assert learning_rate_at(None, 0.03, epoch=70, total_epochs=150) == 0.03


def test_cosine_goes_from_base_to_final_fraction():
    schedule = {"name": "cosine", "final_fraction": 0.01}

    assert learning_rate_at(schedule, 0.03, epoch=1, total_epochs=150) == pytest.approx(0.03)
    assert learning_rate_at(schedule, 0.03, epoch=150, total_epochs=150) == pytest.approx(0.0003)


def test_cosine_halfway_is_the_midpoint_and_never_increases():
    schedule = {"name": "cosine", "final_fraction": 0.0}
    rates = [learning_rate_at(schedule, 1.0, epoch, total_epochs=101) for epoch in range(1, 102)]

    assert rates[50] == pytest.approx(0.5)
    assert all(later <= earlier for earlier, later in zip(rates, rates[1:]))


@pytest.mark.parametrize("schedule, epoch", [({"name": "step"}, 1), ({"name": "cosine", "final_fraction": 2}, 1),
                                             (None, 0), (None, 151)])
def test_invalid_input_fails_fast(schedule, epoch):
    with pytest.raises(ValueError):
        learning_rate_at(schedule, 0.03, epoch=epoch, total_epochs=150)
