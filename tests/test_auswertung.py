import numpy as np
import pytest

from engine import auswertung
from engine.inputs import lade_konfiguration
from engine.montecarlo import rechne
from engine.weather import synthetisches_wetterjahr

WETTER = [synthetisches_wetterjahr(2000 + i, seed=i) for i in range(3)]


@pytest.fixture(scope="module")
def mc():
    konfig = lade_konfiguration(
        {"eingaben": {"montecarlo": {"pfade": 200, "seed": 3, "parametersaetze": 5}}}
    )
    return konfig, rechne(konfig, WETTER)


def test_zahlenformat():
    assert auswertung.zahl(12345.67, 1) == "12.345,7"
    assert auswertung.euro(147.3) == "150 €"
    assert auswertung.euro(-4375) == "-4.380 €"
    assert auswertung.prozent(0.434) == "43 %"


def test_kernaussagen(mc):
    konfig, ergebnis = mc
    aussagen = auswertung.kernaussagen(ergebnis, konfig)
    assert 3 <= len(aussagen) <= 4
    assert any("Ersparnis im ersten Jahr" in a for a in aussagen)
    assert aussagen[0].startswith("Ersparnis über 15 Jahre")
    assert "Investition von 6.000 €" in aussagen[0]
    assert not any("mortis" in a for a in aussagen)


def test_kennzahlentabelle(mc):
    _, ergebnis = mc
    tabelle = auswertung.kennzahlentabelle(ergebnis)
    assert list(tabelle.columns) == ["P10", "P50", "P90"]
    assert "Barwert" in tabelle.index
    assert "Ersparnis im Betrachtungszeitraum" in tabelle.index
    assert not any("Amortisation" in name for name in tabelle.index)


def test_repraesentatives_wetterjahr_ist_median(konfig):
    jahre = [synthetisches_wetterjahr(2000 + i, seed=i) for i in range(5)]
    gewaehlt = auswertung.repraesentatives_wetterjahr(konfig, jahre)
    grad = sorted(np.maximum(0, 15 - w.t_aussen_c).sum() for w in jahre)
    assert np.maximum(0, 15 - gewaehlt.t_aussen_c).sum() == grad[2]


def test_monatsbilanz_summiert_split_strom(konfig):
    detail = auswertung.detailjahr(
        lade_konfiguration({"eingaben": {"speicher": {"status": "neu"}}}), WETTER
    )
    monat = auswertung.monatsbilanz(detail)
    assert list(monat.index)[0] == "Jan" and len(monat) == 12
    assert monat.to_numpy().sum() == pytest.approx(detail.split_strom.sum())
    assert (monat["Speicher"] > 0).any()


def test_beispielwoche(konfig):
    detail = auswertung.detailjahr(konfig, WETTER)
    woche = auswertung.beispielwoche(detail)
    assert len(woche) == 168
    assert woche.index[0].month == 4
    assert set(woche.columns) == {"PV-Erzeugung", "Gesamtlast", "Split-Leistung", "Speicherinhalt"}


def test_diagramme_lassen_sich_bauen(mc, konfig):
    pytest.importorskip("plotly")
    from engine import diagramme

    _, ergebnis = mc
    detail = auswertung.detailjahr(konfig, WETTER)
    for fig in (
        diagramme.faecher_ersparnis(ergebnis),
        diagramme.faecher_heizen(ergebnis),
        diagramme.faecher_kuehlen(ergebnis),
        diagramme.investition_ersparnis(ergebnis),
        diagramme.monatsbilanz(auswertung.monatsbilanz(detail)),
        diagramme.beispielwoche(auswertung.beispielwoche(detail), mit_speicher=True),
    ):
        assert fig.data
