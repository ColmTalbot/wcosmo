import pytest

from wcosmo import constants


def test_known_constant_raises_error():
    with pytest.raises(AttributeError):
        constants.unknown
