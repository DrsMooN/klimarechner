import numpy as np
import pytest

from engine.loads import (
    dynamisierungsfaktor,
    haushaltslast_kw,
    heizbedarf_kw,
    kuehlbedarf_kw,
    lade_lastprofil,
    normiertes_lastprofil,
    verzoegert,
    waermeverlustkoeffizient_w_k,
)


def test_waermeverlustkoeffizient(konfig):
    # 100 W/m² * 140 m² / (21 - (-12)) K
    assert waermeverlustkoeffizient_w_k(konfig) == pytest.approx(14000 / 33)


def test_heizbedarf_ab_heizgrenze():
    t = np.array([-12.0, 0.0, 14.9, 15.0, 25.0])
    bedarf = heizbedarf_kw(t, h_w_k=1000, heizgrenze_c=15)
    np.testing.assert_allclose(bedarf, [27.0, 15.0, 0.1, 0.0, 0.0])


def test_kuehlbedarf():
    t = np.array([19.0, 20.0, 30.0, 22.0])
    g = np.array([800.0, 0.0, 500.0, 0.0])
    bedarf = kuehlbedarf_kw(
        t, g, h_w_k=100, t_soll_c=25, kuehlgrenze_c=20, solarfaktor_m2=1.0, interne_lasten_w=200
    )
    # unter Kühlgrenze null; bei 20 °C: -500 + 0 + 200 < 0 -> null
    np.testing.assert_allclose(bedarf, [0.0, 0.0, 1.2, 0.0])


def test_verzoegerung_verschiebt_solare_gewinne_in_den_abend():
    stunde = np.arange(24 * 10)
    g = np.where((stunde % 24 >= 10) & (stunde % 24 < 15), 800.0, 0.0)
    glatt = verzoegert(g, zeitkonstante_h=6)
    np.testing.assert_array_equal(verzoegert(g, 0), g)
    # Summe bleibt (bis auf den Rest am Ende) erhalten, Abendstunden bekommen Last
    assert glatt.sum() == pytest.approx(g.sum(), rel=0.02)
    assert glatt[24 * 5 + 20] > 0
    assert glatt.max() < g.max()

    t = np.full_like(g, 26.0)
    ohne = kuehlbedarf_kw(t, g, 100, 25, 20, 1.0, 0)
    mit = kuehlbedarf_kw(t, g, 100, 25, 20, 1.0, 0, zeitkonstante_solar_h=6)
    abends = stunde % 24 >= 18
    assert mit[abends].sum() > ohne[abends].sum()


def test_normprofil_summe_und_tagesgang():
    profil = normiertes_lastprofil(lade_lastprofil("h0_naeherung"))
    assert profil.shape == (8760,)
    assert profil.sum() == pytest.approx(1.0)
    assert (profil > 0).all()
    # Winter höher als Sommer (Dynamisierung und Typtage)
    januar, juli = profil[: 31 * 24].sum(), profil[181 * 24 : 212 * 24].sum()
    assert januar > juli


def test_dynamisierung_wertebereich():
    f = dynamisierungsfaktor(np.arange(1, 366))
    assert 0.75 < f.min() < f.max() < 1.3


def test_haushaltslast_mit_rauschen_haelt_jahressumme(konfig):
    profil = normiertes_lastprofil(lade_lastprofil("h0_naeherung"))
    ohne = haushaltslast_kw(profil, 3500)
    mit = haushaltslast_kw(profil, 3500, np.random.default_rng(1), 0.15)
    assert ohne.sum() == pytest.approx(3500)
    assert mit.sum() == pytest.approx(3500)
    assert (mit >= 0).all()
    assert not np.allclose(ohne, mit)


def test_heizbedarf_je_flaeche_plausibel(konfig, wetter):
    h = waermeverlustkoeffizient_w_k(konfig)
    je_m2 = heizbedarf_kw(wetter.t_aussen_c, h, 15).sum() / 140
    # Baujahr 1979–1994, grob 120–220 kWh/m²a Nutzwärme
    assert 120 < je_m2 < 220
