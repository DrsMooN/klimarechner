import numpy as np
import pytest

from engine.economics import bewerte
from engine.inputs import lade_konfiguration
from engine.montecarlo import (
    rechne,
    simuliere_energie,
    ziehe_preispfade,
    ziehe_variation,
)
from engine.simulation import Variation, haushaltsprofil, pruefe_plausibilitaet, simuliere
from engine.weather import synthetisches_wetterjahr

WETTER = [synthetisches_wetterjahr(2000 + i, seed=i) for i in range(3)]


def _konfig(**montecarlo):
    mc = {"pfade": 200, "seed": 1, "parametersaetze": 8, **montecarlo}
    return lade_konfiguration({"eingaben": {"montecarlo": mc}})


def test_batch_entspricht_einzelsimulation(konfig, wetter):
    """Ein Parametersatz in der Batch-Simulation = Einzelsimulation mit gleichen Werten."""
    variation = Variation.neutral(konfig, anzahl=3)
    haushalt = haushaltsprofil(konfig) * konfig.eingaben.haushalt.jahresverbrauch_kwh
    batch = simuliere(konfig, wetter, mit_split=True, haushalt=haushalt, variation=variation)
    einzel = simuliere(konfig, wetter, mit_split=True, haushalt=haushalt)
    assert batch.bilanz.netzbezug.shape == (3, 8760)
    for k in range(3):
        np.testing.assert_allclose(batch.bilanz.netzbezug[k], einzel.bilanz.netzbezug)
        np.testing.assert_allclose(batch.split_waerme[k], einzel.split_waerme)


def test_variation_wirkt_in_die_richtige_richtung(konfig, wetter):
    variation = Variation(
        heizgrenze_delta_k=np.array([0.0, 1.0, 0.0]),
        t_soll_kuehlen_c=np.full(3, 25.0),
        schwelltemperatur_c=np.full(3, 5.0),
        cop_abschlag=np.array([0.0, 0.0, 0.1]),
    )
    haushalt = haushaltsprofil(konfig) * 3500
    s = simuliere(konfig, wetter, mit_split=True, haushalt=haushalt, variation=variation).summen()
    # höherer Sollwert -> mehr Heizbedarf
    assert s.heizbedarf[1] > s.heizbedarf[0]
    # COP-Abschlag -> mehr Strom für dieselbe Wärme
    assert s.split_waerme[2] == pytest.approx(s.split_waerme[0])
    assert s.split_strom_heizen[2] == pytest.approx(s.split_strom_heizen[0] / 0.9)


def test_energieerhaltung_im_batch(konfig, wetter):
    k = lade_konfiguration({"eingaben": {"speicher": {"status": "neu"}}})
    rng = np.random.default_rng(0)
    variation = ziehe_variation(k, 4, rng)
    haushalt = haushaltsprofil(k) * 3500
    b = simuliere(k, wetter, mit_split=True, haushalt=haushalt, variation=variation).bilanz
    np.testing.assert_allclose(
        b.pv + b.netzbezug + b.entladung, b.last + b.einspeisung + b.ladung, atol=1e-6
    )


def test_preispfade_statistik():
    konfig = _konfig()
    s = konfig.preisszenario
    p = ziehe_preispfade(konfig, 20000, np.random.default_rng(0))
    assert p.strom.shape == (20000, 15)
    np.testing.assert_allclose(p.strom[:, 0], 0.35)
    # Erwartungswert entspricht dem deterministischen Pfad
    assert p.strom[:, -1].mean() == pytest.approx(0.35 * (1 + s.strom_drift) ** 14, rel=0.02)
    # Korrelation der jährlichen Renditen
    r_strom = np.diff(np.log(p.strom), axis=1).ravel()
    r_brenn = np.diff(np.log(p.brennstoff), axis=1).ravel()
    assert np.corrcoef(r_strom, r_brenn)[0, 1] == pytest.approx(s.korrelation, abs=0.03)


def test_reproduzierbar_mit_seed():
    a = rechne(_konfig(), WETTER)
    b = rechne(_konfig(), WETTER)
    np.testing.assert_array_equal(a.ersparnis, b.ersparnis)
    assert a.kennzahlen() == b.kennzahlen()
    c = rechne(_konfig(seed=2), WETTER)
    assert not np.array_equal(a.ersparnis, c.ersparnis)


def test_ohne_streuung_gleich_deterministisch():
    """Ohne Volatilität, Verhalten, Technik und Rauschen = deterministische Rechnung."""
    ohne = {
        "verhalten": {"sigma_t_soll_k": 0, "sigma_schwelle_k": 0},
        "technik": {"cop_abschlag_min": 0, "cop_abschlag_max": 0},
        "haushalt": {"rauschen_sigma": 0},
        "preisszenarien": {"moderat": {"strom_volatilitaet": 0, "brennstoff_volatilitaet": 0}},
    }
    konfig = lade_konfiguration(
        {"eingaben": {"montecarlo": {"pfade": 5, "parametersaetze": 2}}, "annahmen": ohne}
    )
    wetter = WETTER[:1]
    mc = rechne(konfig, wetter)
    from engine.simulation import simuliere_jahr

    det = bewerte(
        konfig,
        [simuliere_jahr(konfig, wetter[0], mit_split=False)],
        [simuliere_jahr(konfig, wetter[0], mit_split=True)],
    )
    for pfad in range(5):
        np.testing.assert_allclose(mc.ersparnis[pfad], det.ersparnis, rtol=1e-9)
    np.testing.assert_allclose(mc.barwert, det.barwert, rtol=1e-9)


def test_kennzahlen_und_perzentile():
    ergebnis = rechne(_konfig(), WETTER)
    k = ergebnis.kennzahlen()
    assert k["barwert_eur"][10] <= k["barwert_eur"][50] <= k["barwert_eur"][90]
    assert 0 <= k["wahrscheinlichkeit_amortisation"] <= 1
    assert ergebnis.kumulierter_cashflow.shape == (200, 16)
    np.testing.assert_allclose(ergebnis.kumulierter_cashflow[:, 0], -6000)


def test_amortisation_wird_wahrscheinlicher_mit_kleiner_investition():
    teuer = rechne(_konfig(), WETTER)
    guenstig = rechne(
        lade_konfiguration(
            {
                "eingaben": {
                    "montecarlo": {"pfade": 200, "seed": 1, "parametersaetze": 8},
                    "kosten": {"investition_split_eur": 1500},
                }
            }
        ),
        WETTER,
    )
    assert guenstig.wahrscheinlichkeit_amortisation > teuer.wahrscheinlichkeit_amortisation
    assert np.isfinite(guenstig.amortisation_jahre).any()


def test_energie_wird_wiederverwendet():
    konfig = _konfig()
    energie = simuliere_energie(konfig, WETTER, np.random.default_rng(5))
    assert energie.referenz.netzbezug.shape == (3, 8)
    a = rechne(konfig, WETTER, energie=energie)
    b = rechne(konfig, [], energie=energie)
    np.testing.assert_array_equal(a.ersparnis, b.ersparnis)


def test_plausibilitaetswarnung(konfig, caplog):
    assert pruefe_plausibilitaet(konfig, 3.5) == []
    assert pruefe_plausibilitaet(konfig, 0.0) == []
    warnungen = pruefe_plausibilitaet(konfig, 6.0)
    assert len(warnungen) == 1
    assert "6.00" in caplog.text
