"""Streamlit-Oberfläche des Klimarechners (Prototyp).

Start::

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import httpx
import streamlit as st
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import auswertung, diagramme  # noqa: E402
from engine.inputs import Konfiguration, lade_defaults, lade_konfiguration  # noqa: E402
from engine.montecarlo import rechne  # noqa: E402
from engine.report import erstelle_bericht  # noqa: E402
from engine.weather import (  # noqa: E402
    PVGISAnfrage,
    PVGISClient,
    PVGISFehler,
    Wetterjahr,
    synthetisches_wetterjahr,
)

WETTER_PVGIS = "PVGIS (historische Wetterjahre, Internet nötig)"
WETTER_SYNTHETISCH = "Synthetisches Testwetter (offline, nur Demo)"

BAUALTERSKLASSEN = {
    "vor_1949": "vor 1949",
    "1949_1978": "1949–1978",
    "1979_1994": "1979–1994",
    "1995_2009": "1995–2009",
    "ab_2010": "ab 2010",
    "effizienzhaus": "Effizienzhaus",
}
HEIZSTRATEGIEN = {
    "uebergangszeit": "Übergangszeit (Split nur oberhalb der Schwelltemperatur)",
    "bevorzugt_split": "Bevorzugt Split (Rest über Bestandsheizung)",
    "vollheizung_teilraeume": "Vollheizung der Split-Räume",
}
AUSBAU = {"keine": "keine", "vorhanden": "vorhanden", "neu": "neu (Teil der Investition)"}
HEIZTRAEGER = {"gas": "Gas", "oel": "Öl"}
SZENARIEN = {"moderat": "moderat", "steigend": "steigend", "volatil": "volatil"}
PRUEFPUNKTE = (-7, 2, 7, 12)


def _auswahl(label: str, optionen: dict[str, str], wert: str, **kwargs) -> str:
    schluessel = list(optionen)
    return st.selectbox(
        label, schluessel, index=schluessel.index(wert), format_func=optionen.get, **kwargs
    )


def eingabemaske(d: dict) -> tuple[dict, str]:
    """Eingabeformular; gibt Überschreibungen (YAML-Struktur) und Wetterquelle zurück."""
    e, a = d["eingaben"], d["annahmen"]
    neu: dict = {"eingaben": {}, "annahmen": {}}

    spalte1, spalte2, spalte3 = st.columns(3)
    with spalte1:
        st.subheader("Gebäude")
        g = e["gebaeude"]
        gebaeude = {
            "beheizte_flaeche_m2": st.number_input(
                "Beheizte Fläche gesamt [m²]", 10.0, 2000.0, float(g["beheizte_flaeche_m2"]), 5.0
            ),
            "split_flaeche_m2": st.number_input(
                "Fläche der Räume mit Split-Gerät [m²]",
                0.0,
                2000.0,
                float(g["split_flaeche_m2"]),
                5.0,
            ),
            "baualtersklasse": _auswahl(
                "Baualtersklasse", BAUALTERSKLASSEN, str(g["baualtersklasse"])
            ),
            "t_soll_heizen_c": st.number_input(
                "Raumsolltemperatur Heizen [°C]", 15.0, 26.0, float(g["t_soll_heizen_c"]), 0.5
            ),
            "t_soll_kuehlen_c": st.number_input(
                "Raumsolltemperatur Kühlen [°C]", 20.0, 30.0, float(g["t_soll_kuehlen_c"]), 0.5
            ),
            "heizgrenztemperatur_c": st.number_input(
                "Heizgrenztemperatur [°C]", 5.0, 22.0, float(g["heizgrenztemperatur_c"]), 0.5
            ),
        }
        neu["eingaben"]["gebaeude"] = gebaeude

        st.subheader("Bestandsheizung")
        b = e["bestand"]
        neu["eingaben"]["bestand"] = {
            "heiztraeger": _auswahl("Heizträger", HEIZTRAEGER, b["heiztraeger"]),
            "kesselwirkungsgrad": st.number_input(
                "Kesselwirkungsgrad [%]", 50, 110, int(b["kesselwirkungsgrad"] * 100), 1
            )
            / 100,
        }

        st.subheader("Standort")
        s = e["standort"]
        neu["eingaben"]["standort"] = {
            "breitengrad": st.number_input(
                "Breitengrad", -90.0, 90.0, float(s["breitengrad"]), 0.01, format="%.3f"
            ),
            "laengengrad": st.number_input(
                "Längengrad", -180.0, 180.0, float(s["laengengrad"]), 0.01, format="%.3f"
            ),
        }

    with spalte2:
        st.subheader("Split-Geräte")
        r = e["geraete"]
        neu["eingaben"]["geraete"] = {
            "anzahl_innengeraete": st.number_input(
                "Anzahl Innengeräte", 0, 12, int(r["anzahl_innengeraete"]), 1
            ),
            "nennleistung_heizen_kw": st.number_input(
                "Nennleistung Heizen je Gerät [kW]",
                0.0,
                20.0,
                float(r["nennleistung_heizen_kw"]),
                0.1,
            ),
            "nennleistung_kuehlen_kw": st.number_input(
                "Nennleistung Kühlen je Gerät [kW]",
                0.0,
                20.0,
                float(r["nennleistung_kuehlen_kw"]),
                0.1,
            ),
            "scop": st.number_input("SCOP", 1.5, 10.0, float(r["scop"]), 0.1),
            "seer": st.number_input("SEER", 1.5, 15.0, float(r["seer"]), 0.1),
        }

        st.subheader("Nutzung")
        n = e["nutzung"]
        neu["eingaben"]["nutzung"] = {
            "heizstrategie": _auswahl("Heizstrategie", HEIZSTRATEGIEN, n["heizstrategie"]),
            "split_schwelltemperatur_c": st.number_input(
                "Schwelltemperatur Split [°C]",
                -20.0,
                20.0,
                float(n["split_schwelltemperatur_c"]),
                0.5,
                help="Nur bei Strategie Übergangszeit: darunter heizt die Bestandsheizung.",
            ),
            "kuehlen_aktiv": st.checkbox("Kühlen aktiv", bool(n["kuehlen_aktiv"])),
        }

        st.subheader("Haushalt")
        neu["eingaben"]["haushalt"] = {
            "jahresverbrauch_kwh": st.number_input(
                "Jahresstromverbrauch ohne Klima [kWh]",
                0,
                50_000,
                int(e["haushalt"]["jahresverbrauch_kwh"]),
                100,
            )
        }

    with spalte3:
        st.subheader("PV-Anlage")
        p = e["pv"]
        pv = {
            "status": _auswahl("PV", AUSBAU, p["status"]),
            "leistung_kwp": st.number_input(
                "Leistung [kWp]", 0.0, 100.0, float(p["leistung_kwp"]), 0.5
            ),
            "azimut_grad": st.number_input(
                "Ausrichtung [°] (0 = Süd, −90 = Ost, 90 = West)",
                -180.0,
                180.0,
                float(p["azimut_grad"]),
                5.0,
            ),
            "neigung_grad": st.number_input(
                "Neigung [°]", 0.0, 90.0, float(p["neigung_grad"]), 5.0
            ),
        }
        neu["eingaben"]["pv"] = pv

        st.subheader("Speicher")
        sp = e["speicher"]
        speicher = {
            "status": _auswahl("Speicher", AUSBAU, sp["status"]),
            "kapazitaet_kwh": st.number_input(
                "Kapazität [kWh]", 0.0, 200.0, float(sp["kapazitaet_kwh"]), 0.5
            ),
        }
        leistung = st.number_input(
            "Lade-/Entladeleistung [kW]", 0.0, 50.0, float(sp["ladeleistung_kw"]), 0.5
        )
        speicher["ladeleistung_kw"] = speicher["entladeleistung_kw"] = leistung
        neu["eingaben"]["speicher"] = speicher

        st.subheader("Preise")
        pr = e["preise"]
        neu["eingaben"]["preise"] = {
            "strom_bezug_eur_kwh": st.number_input(
                "Strompreis [€/kWh]", 0.0, 2.0, float(pr["strom_bezug_eur_kwh"]), 0.01
            ),
            "brennstoff_eur_kwh": st.number_input(
                "Brennstoffpreis [€/kWh]", 0.0, 1.0, float(pr["brennstoff_eur_kwh"]), 0.01
            ),
            "einspeiseverguetung_eur_kwh": st.number_input(
                "Einspeisevergütung [€/kWh]",
                0.0,
                1.0,
                float(pr["einspeiseverguetung_eur_kwh"]),
                0.005,
                format="%.3f",
            ),
            "szenario": _auswahl("Preisentwicklung", SZENARIEN, pr["szenario"]),
        }

        st.subheader("Kosten und Rahmen")
        k = e["kosten"]
        kosten = {
            "investition_split_eur": st.number_input(
                "Investition Split [€]", 0, 100_000, int(k["investition_split_eur"]), 100
            ),
            "investition_pv_eur": st.number_input(
                "Investition PV [€] (nur bei neuer PV)",
                0,
                100_000,
                int(k["investition_pv_eur"]),
                500,
            ),
            "investition_speicher_eur": st.number_input(
                "Investition Speicher [€] (nur bei neuem Speicher)",
                0,
                100_000,
                int(k["investition_speicher_eur"]),
                500,
            ),
            "wartung_eur_jahr": st.number_input(
                "Wartung [€/Jahr]", 0, 5_000, int(k["wartung_eur_jahr"]), 10
            ),
        }
        # Investitionen gehören nur zu neuen Komponenten
        if pv["status"] != "neu":
            kosten["investition_pv_eur"] = 0
        if speicher["status"] != "neu":
            kosten["investition_speicher_eur"] = 0
        neu["eingaben"]["kosten"] = kosten
        neu["eingaben"]["rahmen"] = {
            "betrachtungszeitraum_jahre": st.number_input(
                "Betrachtungszeitraum [Jahre]",
                1,
                40,
                int(e["rahmen"]["betrachtungszeitraum_jahre"]),
                1,
            ),
            "kalkulationszins": st.number_input(
                "Kalkulationszins [%]",
                -5.0,
                20.0,
                float(e["rahmen"]["kalkulationszins"]) * 100,
                0.5,
            )
            / 100,
        }

    with st.expander("Erweitert"):
        e1, e2, e3 = st.columns(3)
        with e1:
            st.markdown("**COP-Prüfpunkte (EN 14825)**")
            pruefpunkte_nutzen = st.checkbox(
                "Prüfpunkte statt SCOP verwenden", e["geraete"]["cop_pruefpunkte"] is not None
            )
            vorhanden = e["geraete"]["cop_pruefpunkte"] or {}
            standard = {-7: 2.8, 2: 3.8, 7: 4.9, 12: 6.0}
            punkte = {
                t: st.number_input(
                    f"COP bei {t} °C",
                    0.5,
                    10.0,
                    float(vorhanden.get(t, standard[t])),
                    0.1,
                    key=f"cop_{t}",
                )
                for t in PRUEFPUNKTE
            }
            if pruefpunkte_nutzen:
                neu["eingaben"]["geraete"]["cop_pruefpunkte"] = punkte
        with e2:
            st.markdown("**Modell**")
            neu["annahmen"]["waermepumpe"] = {
                "praxisabschlag": st.number_input(
                    "Praxisabschlag auf COP/EER [%]",
                    0,
                    50,
                    int(round(a["waermepumpe"]["praxisabschlag"] * 100)),
                    1,
                    help="Abschlag Feld gegenüber Prüfstand (Takten, Ventilator, Regelung).",
                )
                / 100,
            }
        with e3:
            st.markdown("**Monte-Carlo**")
            mc = e["montecarlo"]
            neu["eingaben"]["montecarlo"] = {
                "pfade": st.number_input("Pfade", 50, 20_000, int(mc["pfade"]), 50),
                "seed": st.number_input("Seed", 0, 2**31 - 1, int(mc["seed"]), 1),
                "parametersaetze": st.number_input(
                    "Parametersätze", 1, 1000, int(mc["parametersaetze"]), 10
                ),
            }
        wetterquelle = st.radio(
            "Wetterdaten", (WETTER_PVGIS, WETTER_SYNTHETISCH), key="wetterquelle"
        )
    return neu, wetterquelle


@st.cache_data(show_spinner="Lade Wetterdaten von PVGIS …", max_entries=20)
def pvgis_wetter(anfrage_json: str) -> list[Wetterjahr]:
    konfig = lade_konfiguration()
    anfrage = PVGISAnfrage(**json.loads(anfrage_json))
    return PVGISClient(konfig.annahmen.pvgis).wetterjahre(anfrage)


@st.cache_data(show_spinner=False)
def synthetisches_wetter(breitengrad: float) -> list[Wetterjahr]:
    return [synthetisches_wetterjahr(2000 + i, breitengrad, seed=i) for i in range(10)]


def lade_wetter(konfig: Konfiguration, quelle: str) -> list[Wetterjahr]:
    if quelle == WETTER_SYNTHETISCH:
        return synthetisches_wetter(konfig.eingaben.standort.breitengrad)
    anfrage = PVGISAnfrage.aus_konfiguration(konfig)
    return pvgis_wetter(json.dumps(asdict(anfrage)))


@st.cache_data(show_spinner="Rechne Monte-Carlo-Simulation …", max_entries=10)
def berechne(konfig_json: str, quelle: str, _wetter: list[Wetterjahr]):
    konfig = Konfiguration.model_validate_json(konfig_json)
    ergebnis = rechne(konfig, _wetter)
    detail = auswertung.detailjahr(konfig, _wetter)
    return ergebnis, detail


@st.cache_data(show_spinner="Erstelle PDF-Bericht …", max_entries=10)
def bericht_pdf(konfig_json: str, wetterjahre: tuple[int, ...], _ergebnis, _detail) -> bytes:
    konfig = Konfiguration.model_validate_json(konfig_json)
    return erstelle_bericht(konfig, _ergebnis, _detail, len(wetterjahre))


def fehlermeldungen(fehler: ValidationError) -> list[str]:
    meldungen = []
    for f in fehler.errors():
        ort = " → ".join(str(teil) for teil in f["loc"] if teil != "eingaben")
        text = f["msg"].removeprefix("Value error, ")
        meldungen.append(f"{ort}: {text}" if ort else text)
    return meldungen


def ergebnisseite(konfig: Konfiguration, ergebnis, detail, wetter: list[Wetterjahr]) -> None:
    st.header("Ergebnis")
    for aussage in auswertung.kernaussagen(ergebnis, konfig):
        st.markdown(f"- **{aussage}**")
    for warnung in ergebnis.warnungen(konfig):
        st.warning(warnung)

    k = ergebnis.kennzahlen()
    m1, m2, m3, m4 = st.columns(4)
    amort = k["amortisation_jahre"][50]
    m1.metric(
        "Amortisation (mittlerer Fall)",
        f"{auswertung.zahl(amort, 1)} Jahre" if amort != float("inf") else "nicht erreicht",
    )
    m2.metric("Ersparnis im 1. Jahr (P50)", auswertung.euro(k["ersparnis_jahr1_eur"][50]))
    m3.metric("Barwert (P50)", auswertung.euro(k["barwert_eur"][50]))
    m4.metric("Jahresarbeitszahl Heizen", auswertung.zahl(k["jaz_heizen"][50], 1))

    g1, g2 = st.columns(2)
    g1.plotly_chart(diagramme.faecher_cashflow(ergebnis), width="stretch")
    g2.plotly_chart(diagramme.histogramm_amortisation(ergebnis), width="stretch")

    monat = auswertung.monatsbilanz(detail)
    woche = auswertung.beispielwoche(detail)
    g3, g4 = st.columns(2)
    g3.plotly_chart(diagramme.monatsbilanz(monat), width="stretch")
    g4.plotly_chart(diagramme.beispielwoche(woche, konfig.eingaben.speicher.aktiv), width="stretch")
    st.caption(
        f"Monatsbilanz und Beispielwoche: Wetterjahr {detail.wetterjahr} "
        "(mittlerer Heizbedarf), Eingabewerte ohne Streuung. "
        f"Monte-Carlo: {ergebnis.pfade} Pfade über {len(wetter)} Wetterjahre."
    )

    st.download_button(
        "PDF-Bericht herunterladen",
        bericht_pdf(konfig.model_dump_json(), tuple(w.jahr for w in wetter), ergebnis, detail),
        file_name="klimarechner-bericht.pdf",
        mime="application/pdf",
        type="primary",
    )

    with st.expander("Kennzahlen als Tabelle"):
        st.dataframe(auswertung.kennzahlentabelle(ergebnis), width="stretch")
    with st.expander("Monatsbilanz als Tabelle [kWh]"):
        st.dataframe(monat.round(0), width="stretch")


def main() -> None:
    st.set_page_config(page_title="Klimarechner", page_icon="❄️", layout="wide")
    st.title("Wirtschaftlichkeit Split-Klimaanlage")
    st.caption(
        "Prototyp. Schätzung auf Basis vereinfachter Modelle, keine normgerechte "
        "Auslegung, keine Garantie. Alle Standardwerte sind Platzhalter."
    )

    with st.form("eingaben"):
        ueberschreibungen, quelle = eingabemaske(lade_defaults())
        abgeschickt = st.form_submit_button("Berechnen", type="primary")

    if abgeschickt:
        try:
            konfig = lade_konfiguration(ueberschreibungen)
        except ValidationError as fehler:
            st.error(
                "Bitte Eingaben prüfen:\n\n" + "\n".join(f"- {m}" for m in fehlermeldungen(fehler))
            )
            return
        try:
            wetter = lade_wetter(konfig, quelle)
        except (PVGISFehler, OSError, httpx.HTTPError) as fehler:
            st.error(
                f"Wetterdaten konnten nicht geladen werden ({fehler}). Unter „Erweitert“ "
                "kann synthetisches Testwetter gewählt werden."
            )
            return
        ergebnis, detail = berechne(konfig.model_dump_json(), quelle, wetter)
        st.session_state["ergebnis"] = (konfig, ergebnis, detail, wetter)

    if "ergebnis" in st.session_state:
        ergebnisseite(*st.session_state["ergebnis"])


main()
