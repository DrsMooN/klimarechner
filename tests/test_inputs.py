import pytest
from pydantic import ValidationError

from engine.inputs import (
    Ausbaustatus,
    Baualtersklasse,
    Heizstrategie,
    Konfiguration,
    lade_defaults,
    lade_konfiguration,
)


def test_defaults_sind_gueltig():
    konfig = lade_konfiguration()
    assert konfig.meta.platzhalter is True
    assert konfig.eingaben.gebaeude.baualtersklasse == Baualtersklasse.B1979_1994
    assert konfig.eingaben.nutzung.heizstrategie == Heizstrategie.UEBERGANGSZEIT
    assert konfig.preisszenario is konfig.annahmen.preisszenarien["moderat"]


def test_abgeleitete_groessen():
    konfig = lade_konfiguration()
    assert konfig.eingaben.gebaeude.split_anteil == pytest.approx(60 / 140)
    assert konfig.eingaben.geraete.gesamtleistung_heizen_kw == pytest.approx(6.4)
    assert konfig.spezifische_heizlast_w_m2 == 100
    assert konfig.eingaben.pv.aktiv
    assert not konfig.eingaben.speicher.aktiv


def test_ueberschreibung_wird_zusammengefuehrt():
    konfig = lade_konfiguration({"eingaben": {"gebaeude": {"split_flaeche_m2": 40}}})
    assert konfig.eingaben.gebaeude.split_flaeche_m2 == 40
    # Rest der Gruppe bleibt erhalten
    assert konfig.eingaben.gebaeude.beheizte_flaeche_m2 == 140


def test_kennlinie_wird_als_ganzes_ersetzt():
    neu = {-10: 0.6, 10: 1.0}
    konfig = lade_konfiguration({"annahmen": {"waermepumpe": {"leistungsfaktor_heizen": neu}}})
    assert konfig.annahmen.waermepumpe.leistungsfaktor_heizen == neu


def test_cop_pruefpunkte_werden_sortiert():
    punkte = {12: 6.0, -7: 2.8, 7: 4.9, 2: 3.8}
    konfig = lade_konfiguration({"eingaben": {"geraete": {"cop_pruefpunkte": punkte}}})
    assert list(konfig.eingaben.geraete.cop_pruefpunkte) == [-7, 2, 7, 12]


def test_cop_pruefpunkte_als_text_schluessel():
    # z. B. aus JSON oder einer Web-Maske
    konfig = lade_konfiguration(
        {"eingaben": {"geraete": {"cop_pruefpunkte": {"-7": 2.8, "7": 4.9}}}}
    )
    assert konfig.eingaben.geraete.cop_pruefpunkte == {-7.0: 2.8, 7.0: 4.9}


def test_json_rundreise():
    konfig = lade_konfiguration({"eingaben": {"geraete": {"cop_pruefpunkte": {-7: 2.8, 7: 4.9}}}})
    wieder = Konfiguration.model_validate_json(konfig.model_dump_json())
    assert wieder == konfig


@pytest.mark.parametrize(
    ("ueberschreibung", "meldung"),
    [
        ({"gebaeude": {"split_flaeche_m2": 200}}, "Split-Fläche"),
        ({"gebaeude": {"t_soll_heizen_c": 22, "t_soll_kuehlen_c": 21}}, "Sollwert Kühlen"),
        ({"gebaeude": {"heizgrenztemperatur_c": 22}}, "Heizgrenztemperatur"),
        ({"gebaeude": {"baualtersklasse": "1890"}}, "baualtersklasse"),
        ({"geraete": {"cop_pruefpunkte": {7: 4.0}}}, "mindestens zwei"),
        ({"geraete": {"cop_pruefpunkte": {-7: 0, 7: 4.0}}}, "größer 0"),
        ({"speicher": {"soc_min": 0.9, "soc_max": 0.5}}, "soc_min"),
        ({"preise": {"szenario": "utopisch"}}, "Unbekanntes Preisszenario"),
        ({"kosten": {"investition_pv_eur": 10000}}, "Investition PV"),
        ({"kosten": {"investition_speicher_eur": 5000}}, "Investition Speicher"),
        ({"gebaeude": {"unbekannt": 1}}, "unbekannt"),
    ],
)
def test_ungueltige_eingaben(ueberschreibung, meldung):
    with pytest.raises(ValidationError, match=meldung):
        lade_konfiguration({"eingaben": ueberschreibung})


def test_investition_bei_neuer_pv_erlaubt():
    konfig = lade_konfiguration(
        {"eingaben": {"pv": {"status": "neu"}, "kosten": {"investition_pv_eur": 12000}}}
    )
    assert konfig.eingaben.pv.status == Ausbaustatus.NEU
    assert konfig.eingaben.kosten.investition_gesamt_eur == 18000


def test_heizlasttabelle_muss_vollstaendig_sein():
    daten = lade_defaults()
    del daten["annahmen"]["spezifische_heizlast_w_m2"]["vor_1949"]
    with pytest.raises(ValidationError, match="vor_1949"):
        Konfiguration.model_validate(daten)


def test_eingaben_sind_unveraenderlich():
    konfig = lade_konfiguration()
    with pytest.raises(ValidationError):
        konfig.eingaben.gebaeude.split_flaeche_m2 = 10
