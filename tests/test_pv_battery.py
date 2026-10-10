import numpy as np
import pytest

from engine.inputs import Speicher
from engine.pv_battery import energiebilanz

SPEICHER = Speicher(
    status="neu",
    kapazitaet_kwh=5.0,
    ladeleistung_kw=2.0,
    entladeleistung_kw=2.0,
    wirkungsgrad=0.95,
    soc_min=0.1,
    soc_max=1.0,
)
KEIN_SPEICHER = SPEICHER.model_copy(update={"status": "keine"})


def _zufallsreihen(form, seed=0):
    rng = np.random.default_rng(seed)
    pv = np.clip(rng.normal(1.0, 1.5, size=form), 0, None)
    last = rng.uniform(0.1, 3.0, size=form)
    return pv, last


def _pruefe_erhaltung(b, speicher):
    # AC-Bilanz je Stunde
    links = b.pv + b.netzbezug + b.entladung
    rechts = b.last + b.einspeisung + b.ladung
    np.testing.assert_allclose(links, rechts, atol=1e-6)
    # Speicherbilanz: Inhaltsänderung = Ladung - Entladung - Verluste
    start = speicher.soc_min * speicher.kapazitaet_kwh if speicher.aktiv else 0.0
    vorher = np.concatenate([np.full(b.soc_kwh.shape[:-1] + (1,), start), b.soc_kwh[..., :-1]], -1)
    if speicher.aktiv:
        np.testing.assert_allclose(
            b.soc_kwh - vorher, b.ladung - b.entladung - b.verluste, atol=1e-6
        )
    for fluss in (b.netzbezug, b.einspeisung, b.ladung, b.entladung, b.verluste):
        assert (fluss >= -1e-12).all()


@pytest.mark.parametrize("speicher", [SPEICHER, KEIN_SPEICHER])
def test_energieerhaltung_je_stunde(speicher):
    pv, last = _zufallsreihen(8760)
    b = energiebilanz(pv, last, speicher)
    _pruefe_erhaltung(b, speicher)


def test_speichergrenzen():
    pv, last = _zufallsreihen(8760, seed=2)
    b = energiebilanz(pv, last, SPEICHER)
    assert b.soc_kwh.min() >= 0.5 - 1e-9
    assert b.soc_kwh.max() <= 5.0 + 1e-9
    assert b.ladung.max() <= 2.0 + 1e-9
    assert b.entladung.max() <= 2.0 + 1e-9
    assert b.entladung.sum() > 0


def test_vektorisiert_ueber_pfade():
    pv, last = _zufallsreihen((3, 500), seed=4)
    b = energiebilanz(pv, last, SPEICHER)
    _pruefe_erhaltung(b, SPEICHER)
    einzeln = energiebilanz(pv[1], last[1], SPEICHER)
    np.testing.assert_allclose(b.netzbezug[1], einzeln.netzbezug)


def test_ohne_pv():
    last = np.full(24, 1.0)
    b = energiebilanz(np.zeros(24), last, SPEICHER)
    np.testing.assert_allclose(b.netzbezug, last)
    assert b.eigenverbrauchsquote == 0.0
    assert b.autarkiegrad == 0.0


def test_speicher_erhoeht_eigenverbrauch():
    pv, last = _zufallsreihen(8760, seed=5)
    mit = energiebilanz(pv, last, SPEICHER)
    ohne = energiebilanz(pv, last, KEIN_SPEICHER)
    assert mit.eigenverbrauchsquote > ohne.eigenverbrauchsquote
    assert mit.autarkiegrad > ohne.autarkiegrad
