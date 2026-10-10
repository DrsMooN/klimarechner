import numpy as np
import pytest

from engine.inputs import lade_konfiguration
from engine.simulation import simuliere_jahr


def _beide(ueberschreibung, wetter):
    konfig = lade_konfiguration({"eingaben": ueberschreibung})
    return (
        simuliere_jahr(konfig, wetter, mit_split=False),
        simuliere_jahr(konfig, wetter, mit_split=True),
    )


def _gleich(a, b):
    for x, y in [
        (a.bestand_waerme, b.bestand_waerme),
        (a.bilanz.netzbezug, b.bilanz.netzbezug),
        (a.bilanz.einspeisung, b.bilanz.einspeisung),
    ]:
        np.testing.assert_allclose(x, y, atol=1e-9)


def test_waermebilanz(konfig, wetter):
    for strategie in ("uebergangszeit", "bevorzugt_split", "vollheizung_teilraeume"):
        ref, var = _beide({"nutzung": {"heizstrategie": strategie}}, wetter)
        np.testing.assert_allclose(
            var.split_waerme + var.bestand_waerme + var.unterdeckung, var.heizbedarf
        )
        assert (var.bestand_waerme >= -1e-12).all()
        assert var.brennstoff_kwh < ref.brennstoff_kwh


def test_energieerhaltung_variante(wetter):
    _, var = _beide({"speicher": {"status": "neu"}}, wetter)
    b = var.bilanz
    np.testing.assert_allclose(
        b.pv + b.netzbezug + b.entladung, b.last + b.einspeisung + b.ladung, atol=1e-6
    )
    np.testing.assert_allclose(b.last, var.haushalt + var.split_strom, atol=1e-9)


@pytest.mark.parametrize(
    "ueberschreibung",
    [
        {"gebaeude": {"split_flaeche_m2": 0}, "nutzung": {"kuehlen_aktiv": False}},
        {"geraete": {"anzahl_innengeraete": 0}},
    ],
)
def test_ohne_split_ist_variante_gleich_referenz(ueberschreibung, wetter):
    ref, var = _beide(ueberschreibung, wetter)
    _gleich(ref, var)


def test_ohne_pv_und_speicher(wetter):
    ref, var = _beide({"pv": {"status": "keine"}}, wetter)
    assert var.bilanz.pv.sum() == 0
    np.testing.assert_allclose(var.bilanz.netzbezug, var.bilanz.last)
    assert var.pv_anteil_split_heizen == 0


def test_kuehlen_aus(wetter):
    _, var = _beide({"nutzung": {"kuehlen_aktiv": False}}, wetter)
    assert var.split_strom_kuehlen.sum() == 0


def test_neue_pv_nur_in_variante(wetter):
    ref, var = _beide({"pv": {"status": "neu"}}, wetter)
    assert ref.bilanz.pv.sum() == 0
    assert var.bilanz.pv.sum() > 0


def test_uebergangszeit_respektiert_schwelle(wetter):
    _, var = _beide({"nutzung": {"split_schwelltemperatur_c": 5}}, wetter)
    kalt = wetter.t_aussen_c < 5
    assert var.split_waerme[kalt].sum() == 0
    assert var.split_waerme[~kalt].sum() > 0


def test_vollheizung_hat_unterdeckung_bei_kleinen_geraeten(wetter):
    _, var = _beide(
        {
            "nutzung": {"heizstrategie": "vollheizung_teilraeume"},
            "geraete": {"nennleistung_heizen_kw": 0.5},
        },
        wetter,
    )
    assert var.unterdeckung.sum() > 0


def test_jaz_plausibel(wetter):
    for strategie in ("uebergangszeit", "bevorzugt_split"):
        _, var = _beide({"nutzung": {"heizstrategie": strategie}}, wetter)
        assert 2.5 < var.jaz_heizen < 6.5


def test_reproduzierbar(konfig, wetter):
    a = simuliere_jahr(konfig, wetter, mit_split=True, rng=np.random.default_rng(7))
    b = simuliere_jahr(konfig, wetter, mit_split=True, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a.bilanz.netzbezug, b.bilanz.netzbezug)
    assert a.kennzahlen() == b.kennzahlen()


def test_herkunft_split_strom_summiert_sich(konfig, wetter):
    var = simuliere_jahr(
        lade_konfiguration({"eingaben": {"speicher": {"status": "neu"}}}), wetter, mit_split=True
    )
    herkunft = var.herkunft_split_strom_heizen()
    np.testing.assert_allclose(
        herkunft["pv"] + herkunft["speicher"] + herkunft["netz"], var.split_strom_heizen, atol=1e-9
    )
