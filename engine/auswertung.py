"""Aufbereitung der Ergebnisse für Oberfläche und Bericht (ohne UI-Abhängigkeit).

Liefert Kernaussagen in Klartext, eine Kennzahlentabelle sowie Monatsbilanz
und Beispielwoche aus einer deterministischen Simulation des
repräsentativen Wetterjahrs.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np
import pandas as pd

from engine.inputs import Ausbaustatus, Konfiguration
from engine.montecarlo import PERZENTILE, MonteCarloErgebnis
from engine.simulation import Jahresergebnis, sicherer_anteil, simuliere_jahr
from engine.weather import Wetterjahr

MONATSNAMEN = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
_TAGE_JE_MONAT = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def zahl(wert: float, nachkomma: int = 0) -> str:
    """Deutsche Zahlendarstellung: 12.345,6"""
    text = f"{wert:,.{nachkomma}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def euro(wert: float) -> str:
    return f"{zahl(round(wert, -1) if abs(wert) >= 100 else wert)} €"


def prozent(anteil: float) -> str:
    return f"{zahl(anteil * 100)} %"


def jahre(wert: float) -> str:
    return f"{zahl(wert, 1)} Jahren" if wert != 1 else "1 Jahr"


def kernaussagen(ergebnis: MonteCarloErgebnis, konfig: Konfiguration) -> list[str]:
    """Drei bis vier Kernaussagen in Klartext für Kopfzeile und Bericht."""
    e = konfig.eingaben
    k = ergebnis.kennzahlen()
    zeitraum = e.rahmen.betrachtungszeitraum_jahre
    aussagen = []

    w = ergebnis.wahrscheinlichkeit_amortisation
    amort = k["amortisation_jahre"]
    if w >= 0.8:
        aussagen.append(
            f"Amortisation mit 80 % Wahrscheinlichkeit nach {zahl(amort[10], 1)} bis "
            f"{jahre(amort[90])}, im mittleren Fall nach {jahre(amort[50])}."
        )
    elif w > 0:
        mitte = f" Im mittleren Fall nach {jahre(amort[50])}." if np.isfinite(amort[50]) else ""
        aussagen.append(
            f"Die Investition amortisiert sich innerhalb von {zeitraum} Jahren in "
            f"{prozent(w)} der berechneten Fälle.{mitte}"
        )
    else:
        aussagen.append(
            f"Innerhalb von {zeitraum} Jahren amortisiert sich die Investition in keinem "
            "der berechneten Fälle."
        )

    ersparnis = k["ersparnis_jahr1_eur"]
    aussagen.append(
        f"Ersparnis im ersten Jahr: {euro(ersparnis[10])} bis {euro(ersparnis[90])}, "
        f"im mittleren Fall {euro(ersparnis[50])}."
    )

    brennstoff = k["eingesparter_brennstoff_kwh_jahr"][50]
    jaz = k["jaz_heizen"][50]
    if brennstoff > 0:
        aussagen.append(
            f"Das Split-Gerät ersetzt im Mittel {zahl(brennstoff)} kWh Brennstoff pro Jahr "
            f"bei einer Jahresarbeitszahl von etwa {zahl(jaz, 1)}."
        )

    if e.pv.status != Ausbaustatus.KEINE and brennstoff > 0:
        pv = k["pv_anteil_split_heizen"][50]
        aussagen.append(f"{prozent(pv)} des Heizstroms für das Split-Gerät kommen aus der PV.")
    return aussagen


KENNZAHLEN_BESCHRIFTUNG = {
    "barwert_eur": ("Barwert", "€"),
    "interner_zinsfuss": ("Interner Zinsfuß", "%"),
    "amortisation_jahre": ("Amortisation", "Jahre"),
    "ersparnis_jahr1_eur": ("Ersparnis im 1. Jahr", "€"),
    "kuehlkosten_eur_jahr": ("Kühlkosten pro Jahr", "€"),
    "jaz_heizen": ("Jahresarbeitszahl Heizen", ""),
    "pv_anteil_split_heizen": ("PV-Anteil am Split-Heizstrom", "%"),
    "eingesparter_brennstoff_kwh_jahr": ("Eingesparter Brennstoff pro Jahr", "kWh"),
    "autarkiegrad": ("Autarkiegrad", "%"),
    "eigenverbrauchsquote": ("Eigenverbrauchsquote", "%"),
}


def _zelle(wert: float, einheit: str) -> str:
    if np.isposinf(wert):
        return "nicht erreicht"
    if np.isneginf(wert) or np.isnan(wert):
        return "–"
    if einheit == "%":
        return prozent(wert)
    if einheit == "€":
        return euro(wert)
    if einheit == "Jahre":
        return zahl(wert, 1)
    if einheit == "":
        return zahl(wert, 2)
    return zahl(wert)


def kennzahlentabelle(ergebnis: MonteCarloErgebnis) -> pd.DataFrame:
    """Kennzahlen als formatierte Tabelle mit Spalten P10, P50, P90."""
    k = ergebnis.kennzahlen()
    zeilen = []
    for schluessel, (name, einheit) in KENNZAHLEN_BESCHRIFTUNG.items():
        werte = k[schluessel]
        zeilen.append(
            {
                "Kennzahl": f"{name} [{einheit}]" if einheit not in ("", "%", "€") else name,
                **{f"P{q}": _zelle(werte[q], einheit) for q in PERZENTILE},
            }
        )
    return pd.DataFrame(zeilen).set_index("Kennzahl")


# --- Deterministische Detailansicht -----------------------------------------


def repraesentatives_wetterjahr(
    konfig: Konfiguration, wetterjahre: Sequence[Wetterjahr]
) -> Wetterjahr:
    """Wetterjahr mit dem mittleren Heizbedarf (Median nach Heizgradstunden)."""
    grenze = konfig.eingaben.gebaeude.heizgrenztemperatur_c
    gradstunden = [np.maximum(0, grenze - w.t_aussen_c).sum() for w in wetterjahre]
    reihenfolge = np.argsort(gradstunden)
    return wetterjahre[reihenfolge[len(reihenfolge) // 2]]


def detailjahr(konfig: Konfiguration, wetterjahre: Sequence[Wetterjahr]) -> Jahresergebnis:
    """Variante im repräsentativen Wetterjahr mit den Eingabewerten (ohne Streuung)."""
    return simuliere_jahr(konfig, repraesentatives_wetterjahr(konfig, wetterjahre), mit_split=True)


def herkunft_split_strom(ergebnis: Jahresergebnis) -> dict[str, np.ndarray]:
    """Gesamter Split-Strom (Heizen und Kühlen) je Stunde nach Quelle."""
    b = ergebnis.bilanz
    anteil = sicherer_anteil(ergebnis.split_strom, b.last)
    return {
        "PV direkt": anteil * b.direktverbrauch,
        "Speicher": anteil * b.entladung,
        "Netz": anteil * b.netzbezug,
    }


def monatsbilanz(ergebnis: Jahresergebnis) -> pd.DataFrame:
    """Split-Strom je Monat nach Quelle in kWh (Zeilen = Monate)."""
    grenzen = np.cumsum([0] + [t * 24 for t in _TAGE_JE_MONAT])
    daten = {
        quelle: [reihe[a:b].sum() for a, b in zip(grenzen[:-1], grenzen[1:], strict=True)]
        for quelle, reihe in herkunft_split_strom(ergebnis).items()
    }
    return pd.DataFrame(daten, index=MONATSNAMEN)


def beispielwoche(ergebnis: Jahresergebnis, starttag: date | None = None) -> pd.DataFrame:
    """Stundenwerte einer Woche (Default: erste volle Aprilwoche ab 1.4.)."""
    starttag = starttag or date(2025, 4, 1)
    tag_im_jahr = starttag.timetuple().tm_yday - 1
    von, bis = tag_im_jahr * 24, (tag_im_jahr + 7) * 24
    b = ergebnis.bilanz
    index = [pd.Timestamp(starttag) + timedelta(hours=h) for h in range(bis - von)]
    return pd.DataFrame(
        {
            "PV-Erzeugung": b.pv[von:bis],
            "Gesamtlast": b.last[von:bis],
            "Split-Leistung": ergebnis.split_strom[von:bis],
            "Speicherinhalt": b.soc_kwh[von:bis],
        },
        index=pd.DatetimeIndex(index, name="Zeit"),
    )
