import numpy as np
import pytest

from engine.economics import (
    amortisationszeit,
    barwert,
    bewerte,
    deterministischer_preispfad,
    interner_zinsfuss,
    jahreskosten,
    kuehlkosten,
)
from engine.inputs import lade_konfiguration
from engine.simulation import simuliere_jahr


def _faelle(ueberschreibung, wetter):
    konfig = lade_konfiguration({"eingaben": ueberschreibung})
    return (
        konfig,
        [simuliere_jahr(konfig, wetter, mit_split=False)],
        [simuliere_jahr(konfig, wetter, mit_split=True)],
    )


def test_preispfad_mit_drift_und_co2():
    konfig = lade_konfiguration({"eingaben": {"preise": {"szenario": "steigend"}}})
    p = deterministischer_preispfad(konfig)
    s = konfig.preisszenario
    assert p.jahre == 15
    assert p.strom[0] == pytest.approx(0.35)
    assert p.strom[2] == pytest.approx(0.35 * (1 + s.strom_drift) ** 2)
    co2 = 0.201 * 50 / 1000
    assert p.brennstoff[0] == pytest.approx(0.11 + co2)
    np.testing.assert_allclose(p.einspeisung, 0.08)


def test_barwert_und_zinsfuss():
    cf = np.full(10, 1000.0)
    assert barwert(5000, cf, 0.0) == pytest.approx(5000)
    irr = interner_zinsfuss(5000, cf)
    assert barwert(5000, cf, irr) == pytest.approx(0, abs=1e-6)
    assert irr == pytest.approx(0.1510, abs=1e-4)
    # nie positiv -> kein Zinsfuß
    assert interner_zinsfuss(5000, np.full(10, -10.0)) is None


def test_amortisation():
    assert amortisationszeit(5000, np.full(10, 1000.0)) == pytest.approx(5.0)
    assert amortisationszeit(2500, np.full(10, 1000.0)) == pytest.approx(2.5)
    assert amortisationszeit(5000, np.full(3, 1000.0)) is None
    assert amortisationszeit(0, np.full(3, 1000.0)) == 0.0


def test_jahreskosten_aus_bilanz(konfig, wetter):
    ergebnis = simuliere_jahr(konfig, wetter, mit_split=True)
    k = jahreskosten(ergebnis.summen(), 0.30, 0.10, 0.08, wartung=150)
    b = ergebnis.bilanz
    assert k.strom == pytest.approx(b.netzbezug.sum() * 0.30 - b.einspeisung.sum() * 0.08)
    assert k.brennstoff == pytest.approx(ergebnis.brennstoff_kwh * 0.10)
    assert k.gesamt == pytest.approx(k.strom + k.brennstoff + 150)


def test_ersparnis_ist_kostendifferenz(wetter):
    konfig, ref, var = _faelle({}, wetter)
    w = bewerte(konfig, ref, var)
    for t in range(w.preise.jahre):
        assert w.ersparnis[t] == pytest.approx(
            w.kosten_referenz[t].gesamt - w.kosten_variante[t].gesamt
        )
    assert w.kumulierter_cashflow[0] == -6000
    assert w.kumulierter_cashflow[-1] == pytest.approx(-6000 + w.ersparnis.sum())
    assert w.kosten_variante[0].wartung == 150
    assert w.kosten_referenz[0].wartung == 0


def test_ohne_split_keine_ersparnis_ausser_wartung(wetter):
    konfig, ref, var = _faelle(
        {"geraete": {"anzahl_innengeraete": 0}, "kosten": {"wartung_eur_jahr": 0}}, wetter
    )
    w = bewerte(konfig, ref, var)
    np.testing.assert_allclose(w.ersparnis, 0, atol=1e-9)
    assert w.amortisation_jahre is None
    assert w.barwert == pytest.approx(-6000)


def test_kuehlen_kostet(wetter):
    konfig, ref, mit = _faelle({"pv": {"status": "keine"}}, wetter)
    _, _, ohne = _faelle({"pv": {"status": "keine"}, "nutzung": {"kuehlen_aktiv": False}}, wetter)
    w_mit, w_ohne = bewerte(konfig, ref, mit), bewerte(konfig, ref, ohne)
    assert w_ohne.kuehlkosten.sum() == 0
    assert w_mit.kuehlkosten.sum() > 0
    # ohne PV ist der Kühlstrom komplett Netzbezug: Ersparnis sinkt genau um die Kühlkosten
    np.testing.assert_allclose(w_ohne.ersparnis - w_mit.ersparnis, w_mit.kuehlkosten, rtol=1e-9)
    assert kuehlkosten(mit[0].summen(), 0.35, 0.08) == pytest.approx(
        mit[0].split_strom_kuehlen.sum() * 0.35
    )


def test_kuehlstrom_aus_pv_kostet_entgangene_einspeisung(wetter):
    _, _, mit = _faelle({"pv": {"status": "vorhanden"}}, wetter)
    s = mit[0].summen()
    assert 0 < s.kuehlstrom_netz < s.split_strom_kuehlen
    erwartet = s.kuehlstrom_netz * 0.35 + (s.split_strom_kuehlen - s.kuehlstrom_netz) * 0.08
    assert kuehlkosten(s, 0.35, 0.08) == pytest.approx(erwartet)
    assert kuehlkosten(s, 0.35, 0.08) > kuehlkosten(s, 0.35, 0.0)


def test_hoehere_brennstoffpreise_verbessern_ergebnis(wetter):
    konfig_m, ref, var = _faelle({"preise": {"szenario": "moderat"}}, wetter)
    konfig_s = lade_konfiguration({"eingaben": {"preise": {"szenario": "steigend"}}})
    moderat, steigend = bewerte(konfig_m, ref, var), bewerte(konfig_s, ref, var)
    assert steigend.barwert > moderat.barwert


def test_wetterjahre_rotieren(konfig, wetter):
    from engine.weather import synthetisches_wetterjahr

    zweites = synthetisches_wetterjahr(seed=9)
    ref = [simuliere_jahr(konfig, w, mit_split=False) for w in (wetter, zweites)]
    var = [simuliere_jahr(konfig, w, mit_split=True) for w in (wetter, zweites)]
    w = bewerte(konfig, ref, var)
    # Jahr 1 und 3 nutzen dasselbe Wetter, Kosten unterscheiden sich nur durch Preise
    assert w.kosten_referenz[0].brennstoff / w.preise.brennstoff[0] == pytest.approx(
        w.kosten_referenz[2].brennstoff / w.preise.brennstoff[2]
    )
    with pytest.raises(ValueError):
        bewerte(konfig, ref, var[:1])


def test_kennzahlen_vollstaendig(konfig, wetter):
    ref = [simuliere_jahr(konfig, wetter, mit_split=False)]
    var = [simuliere_jahr(konfig, wetter, mit_split=True)]
    kennzahlen = bewerte(konfig, ref, var).kennzahlen()
    assert kennzahlen["ersparnis_jahr1_eur"] > 0
    assert kennzahlen["amortisation_dynamisch_jahre"] is None or (
        kennzahlen["amortisation_dynamisch_jahre"] >= kennzahlen["amortisation_jahre"]
    )
