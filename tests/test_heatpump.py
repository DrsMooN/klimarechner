import numpy as np
import pytest

from engine.heatpump import _bins, erstelle_effizienzmodell, kennlinie, saisonzahl
from engine.inputs import lade_konfiguration


def test_kennlinie_interpoliert_und_extrapoliert():
    punkte = {-7: 2.0, 7: 4.0}
    werte = kennlinie(punkte, np.array([-14.0, -7.0, 0.0, 7.0, 14.0]))
    np.testing.assert_allclose(werte, [1.0, 2.0, 3.0, 4.0, 5.0])


def test_carnot_kalibrierung_trifft_scop(konfig):
    a = konfig.annahmen.waermepumpe
    modell = erstelle_effizienzmodell(konfig.eingaben.geraete, a)
    t, gewichte = _bins(a.referenzklima_heizen, heizen=True)
    assert saisonzahl(modell.cop(t), gewichte) == pytest.approx(
        konfig.eingaben.geraete.scop, rel=0.01
    )


def test_carnot_kalibrierung_trifft_seer(konfig):
    a = konfig.annahmen.waermepumpe
    modell = erstelle_effizienzmodell(konfig.eingaben.geraete, a)
    t, gewichte = _bins(a.referenzklima_kuehlen, heizen=False)
    assert saisonzahl(modell.eer(t), gewichte) == pytest.approx(
        konfig.eingaben.geraete.seer, rel=0.01
    )


def test_cop_steigt_mit_aussentemperatur(konfig):
    modell = erstelle_effizienzmodell(konfig.eingaben.geraete, konfig.annahmen.waermepumpe)
    cop = modell.cop(np.array([-15.0, -7.0, 10.0, 15.0]))
    assert np.all(np.diff(cop) > 0)
    eer = modell.eer(np.array([20.0, 30.0, 40.0]))
    assert np.all(np.diff(eer) < 0)


def test_pruefpunkte_mit_abschlaegen():
    konfig = lade_konfiguration(
        {"eingaben": {"geraete": {"cop_pruefpunkte": {-7: 2.5, 2: 3.5, 7: 4.5, 12: 5.5}}}}
    )
    a = konfig.annahmen.waermepumpe
    modell = erstelle_effizienzmodell(konfig.eingaben.geraete, a)
    assert modell.guetegrad_heizen is None
    # 7 °C liegt außerhalb des Abtaubands, nur Teillastabschlag
    assert modell.cop(np.array([7.0]))[0] == pytest.approx(4.5 * (1 - a.teillast_abschlag))
    # 2 °C im Abtauband
    erwartet = 3.5 * (1 - a.abtau_abschlag) * (1 - a.teillast_abschlag)
    assert modell.cop(np.array([2.0]))[0] == pytest.approx(erwartet)


def test_cop_wird_begrenzt():
    konfig = lade_konfiguration({"eingaben": {"geraete": {"cop_pruefpunkte": {-7: 1.0, 12: 7.0}}}})
    a = konfig.annahmen.waermepumpe
    modell = erstelle_effizienzmodell(konfig.eingaben.geraete, a)
    cop = modell.cop(np.array([-40.0, 60.0]))
    assert cop[0] == a.cop_min
    assert cop[1] == a.cop_max


def test_max_heizleistung(konfig):
    modell = erstelle_effizienzmodell(konfig.eingaben.geraete, konfig.annahmen.waermepumpe)
    nenn = konfig.eingaben.geraete.gesamtleistung_heizen_kw
    leistung = modell.max_heizleistung_kw(np.array([-20.0, -7.0, 10.0]))
    np.testing.assert_allclose(leistung, [0.70 * nenn, 0.85 * nenn, nenn])
