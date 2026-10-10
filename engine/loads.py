"""Heiz-, Kühl- und Haushaltslast (stündlich, vereinfacht).

Alle Lasten in kW als Mittelwert der Stunde, also zugleich kWh je Stunde.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from engine.inputs import Konfiguration
from engine.weather import STUNDEN_PRO_JAHR

LASTPROFIL_VERZEICHNIS = Path(__file__).resolve().parent.parent / "data" / "lastprofile"

# Referenzjahr für die Lage der Wochentage (kein Schaltjahr, beginnt mittwochs)
REFERENZJAHR = 2025


def waermeverlustkoeffizient_w_k(konfig: Konfiguration) -> float:
    """H [W/K] des Gebäudes aus spezifischer Heizlast und Norm-Außentemperatur."""
    gebaeude = konfig.eingaben.gebaeude
    heizlast_w = konfig.spezifische_heizlast_w_m2 * gebaeude.beheizte_flaeche_m2
    delta_t = gebaeude.t_soll_heizen_c - konfig.annahmen.norm_aussentemperatur_c
    return heizlast_w / delta_t


def heizbedarf_kw(t_aussen_c: np.ndarray, h_w_k: float, heizgrenze_c: float) -> np.ndarray:
    """Wärmebedarf des ganzen Gebäudes nach dem Gradtagzahl-Prinzip.

    Bezugstemperatur ist die Heizgrenztemperatur, nicht der Raumsollwert:
    Die Differenz deckt pauschal solare und interne Gewinne ab. Ab der
    Heizgrenze ist der Bedarf null.
    """
    return h_w_k * np.maximum(0.0, heizgrenze_c - t_aussen_c) / 1000.0


def kuehlbedarf_kw(
    t_aussen_c: np.ndarray,
    einstrahlung_w_m2: np.ndarray,
    h_w_k: float,
    t_soll_c: float,
    kuehlgrenze_c: float,
    solarfaktor_m2: float,
    interne_lasten_w: float,
) -> np.ndarray:
    """Kühlbedarf der Split-Räume; null unterhalb der Kühlgrenztemperatur.

    ``h_w_k`` ist hier der Wärmeverlustkoeffizient der Split-Räume.
    """
    last_w = h_w_k * (t_aussen_c - t_soll_c) + solarfaktor_m2 * einstrahlung_w_m2 + interne_lasten_w
    return np.where(t_aussen_c >= kuehlgrenze_c, np.maximum(0.0, last_w) / 1000.0, 0.0)


# --- Haushaltsstrom ---------------------------------------------------------


def lade_lastprofil(name: str, verzeichnis: Path = LASTPROFIL_VERZEICHNIS) -> pd.DataFrame:
    profil = pd.read_csv(verzeichnis / f"{name}.csv", index_col="stunde")
    if list(profil.index) != list(range(24)):
        raise ValueError(f"Lastprofil {name}: erwartet Stunden 0–23")
    return profil


def _saison(tag: date) -> str:
    md = (tag.month, tag.day)
    if md >= (11, 1) or md <= (3, 20):
        return "winter"
    if (5, 15) <= md <= (9, 14):
        return "sommer"
    return "uebergang"


def _tagtyp(tag: date) -> str:
    return {5: "samstag", 6: "sonntag"}.get(tag.weekday(), "werktag")


def dynamisierungsfaktor(tag_im_jahr: np.ndarray) -> np.ndarray:
    """BDEW-Dynamisierung für Haushaltsprofile (Tag 1–365)."""
    t = tag_im_jahr
    return -3.92e-10 * t**4 + 3.2e-7 * t**3 - 7.02e-5 * t**2 + 2.1e-3 * t + 1.24


def normiertes_lastprofil(profil: pd.DataFrame, jahr: int = REFERENZJAHR) -> np.ndarray:
    """8.760 Stundenwerte mit Summe 1 aus den Typtagen eines Profils."""
    werte = np.empty(STUNDEN_PRO_JAHR)
    start = date(jahr, 1, 1)
    for d in range(365):
        tag = start + timedelta(days=d)
        spalte = f"{_saison(tag)}_{_tagtyp(tag)}"
        werte[d * 24 : (d + 1) * 24] = profil[spalte].to_numpy()
    werte *= np.repeat(dynamisierungsfaktor(np.arange(1, 366)), 24)
    return werte / werte.sum()


def haushaltslast_kw(
    normprofil: np.ndarray,
    jahresverbrauch_kwh: float,
    rng: np.random.Generator | None = None,
    rauschen_sigma: float = 0.0,
) -> np.ndarray:
    """Haushaltslast, optional mit multiplikativem Rauschen je Stunde.

    Das Rauschen ändert nur die Verteilung, die Jahressumme bleibt gleich.
    """
    last = normprofil * jahresverbrauch_kwh
    if rng is None or rauschen_sigma == 0 or jahresverbrauch_kwh == 0:
        return last
    faktor = np.maximum(0.0, 1.0 + rng.normal(0.0, rauschen_sigma, size=last.shape))
    verrauscht = last * faktor
    return verrauscht * (jahresverbrauch_kwh / verrauscht.sum())
