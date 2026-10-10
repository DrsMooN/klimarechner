import pytest

from engine.inputs import lade_konfiguration
from engine.weather import synthetisches_wetterjahr


@pytest.fixture(scope="session")
def konfig():
    return lade_konfiguration()


@pytest.fixture(scope="session")
def wetter():
    return synthetisches_wetterjahr(seed=1)
