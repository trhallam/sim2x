import pytest

from sim2x import wavelets as wv


def test_zero_phase_time_axis():
    time = wv.zero_phase_time_axis(128, 0.01)
    assert time.size == (128 + 1)
