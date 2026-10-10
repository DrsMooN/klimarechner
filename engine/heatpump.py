"""Temperaturabhängige Effizienz und Leistungsgrenzen der Split-Geräte.

COP (Heizen) und EER (Kühlen) kommen entweder aus Prüfpunkten (lineare
Interpolation, außerhalb linear extrapoliert) oder aus einem
Carnot-Gütegrad-Modell, dessen Gütegrad so kalibriert ist, dass die
Saisonzahl über ein Referenzklima dem angegebenen SCOP bzw. SEER entspricht.

Abtau- und Teillastabschläge wirken in beiden Fällen. Beim Carnot-Modell
sind sie in der Kalibrierung enthalten, die Saisonzahl stimmt also inklusive
Abschlägen. Der Praxisabschlag (Prüfstand → Feld) wirkt danach zusätzlich
und ist bewusst nicht in der Kalibrierung enthalten.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.inputs import Geraete, Referenzklima, Waermepumpenannahmen

KELVIN = 273.15


def kennlinie(punkte: dict[float, float], x: np.ndarray) -> np.ndarray:
    """Lineare Interpolation mit linearer Extrapolation an beiden Enden."""
    xs = np.array(sorted(punkte), dtype=float)
    ys = np.array([punkte[k] for k in sorted(punkte)], dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.interp(x, xs, ys)
    steigung_links = (ys[1] - ys[0]) / (xs[1] - xs[0])
    steigung_rechts = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    y = np.where(x < xs[0], ys[0] + (x - xs[0]) * steigung_links, y)
    return np.where(x > xs[-1], ys[-1] + (x - xs[-1]) * steigung_rechts, y)


@dataclass(frozen=True)
class Effizienzmodell:
    geraete: Geraete
    annahmen: Waermepumpenannahmen
    guetegrad_heizen: float | None
    """Kalibrierter Carnot-Gütegrad, ``None`` bei Prüfpunkten"""
    guetegrad_kuehlen: float | None

    # --- Heizen ---

    def _carnot_cop(self, t_aussen_c: np.ndarray) -> np.ndarray:
        c = self.annahmen.carnot
        t_kond = c.kondensation_heizen_c + KELVIN
        t_verd = np.asarray(t_aussen_c, dtype=float) - c.verdampfer_abstand_k + KELVIN
        return t_kond / np.maximum(t_kond - t_verd, 1.0)

    def _abschlag_heizen(self, t_aussen_c: np.ndarray) -> np.ndarray:
        a = self.annahmen
        unten, oben = a.abtau_band_c
        abtau = np.where((t_aussen_c >= unten) & (t_aussen_c <= oben), a.abtau_abschlag, 0.0)
        return (1.0 - abtau) * (1.0 - a.teillast_abschlag)

    def cop(self, t_aussen_c: np.ndarray, zusatzabschlag: float = 0.0) -> np.ndarray:
        t = np.asarray(t_aussen_c, dtype=float)
        if self.geraete.cop_pruefpunkte is not None:
            basis = kennlinie(self.geraete.cop_pruefpunkte, t)
        else:
            basis = self.guetegrad_heizen * self._carnot_cop(t)
        wert = basis * self._abschlag_heizen(t) * self._praxisfaktor(zusatzabschlag)
        return np.clip(wert, self.annahmen.cop_min, self.annahmen.cop_max)

    def _praxisfaktor(self, zusatzabschlag) -> np.ndarray | float:
        return (1.0 - self.annahmen.praxisabschlag) * (1.0 - zusatzabschlag)

    def max_heizleistung_kw(self, t_aussen_c: np.ndarray) -> np.ndarray:
        faktor = self.annahmen.leistungsfaktor_heizen
        xs = sorted(faktor)
        return self.geraete.gesamtleistung_heizen_kw * np.interp(
            t_aussen_c, xs, [faktor[k] for k in xs]
        )

    # --- Kühlen ---

    def _carnot_eer(self, t_aussen_c: np.ndarray) -> np.ndarray:
        c = self.annahmen.carnot
        t_verd = c.verdampfung_kuehlen_c + KELVIN
        t_kond = np.asarray(t_aussen_c, dtype=float) + c.kondensator_abstand_k + KELVIN
        return t_verd / np.maximum(t_kond - t_verd, 1.0)

    def eer(self, t_aussen_c: np.ndarray, zusatzabschlag: float = 0.0) -> np.ndarray:
        t = np.asarray(t_aussen_c, dtype=float)
        if self.geraete.eer_pruefpunkte is not None:
            basis = kennlinie(self.geraete.eer_pruefpunkte, t)
        else:
            basis = self.guetegrad_kuehlen * self._carnot_eer(t)
        wert = basis * (1.0 - self.annahmen.teillast_abschlag) * self._praxisfaktor(zusatzabschlag)
        return np.clip(wert, self.annahmen.cop_min, self.annahmen.cop_max)

    def max_kuehlleistung_kw(self, t_aussen_c: np.ndarray) -> np.ndarray:
        return np.full(np.shape(t_aussen_c), self.geraete.gesamtleistung_kuehlen_kw)


def _bins(klima: Referenzklima, heizen: bool) -> tuple[np.ndarray, np.ndarray]:
    """Temperaturen und Energiegewichte (Stunden × relative Last) des Referenzklimas."""
    t = np.array(sorted(klima.stunden), dtype=float)
    stunden = np.array([klima.stunden[k] for k in sorted(klima.stunden)], dtype=float)
    if heizen:
        grenze = klima.heizgrenze_c
        last = (grenze - t) / (grenze - klima.auslegungstemperatur_c)
    else:
        grenze = klima.kuehlgrenze_c
        last = (t - grenze) / (klima.auslegungstemperatur_c - grenze)
    return t, stunden * np.clip(last, 0.0, None)


def saisonzahl(werte: np.ndarray, gewichte: np.ndarray) -> float:
    """Energiegewichtete Saisonzahl: Wärme / Strom."""
    return float(gewichte.sum() / (gewichte / werte).sum())


def erstelle_effizienzmodell(geraete: Geraete, annahmen: Waermepumpenannahmen) -> Effizienzmodell:
    """Modell mit kalibriertem Gütegrad (ohne Prüfpunkte).

    Kalibriert wird so, dass COP/EER **ohne** Praxisabschlag über das
    Referenzklima den Datenblattwert SCOP/SEER ergeben.
    """
    roh = Effizienzmodell(geraete, annahmen, guetegrad_heizen=1.0, guetegrad_kuehlen=1.0)

    guetegrad_heizen = None
    if geraete.cop_pruefpunkte is None:
        t, gewichte = _bins(annahmen.referenzklima_heizen, heizen=True)
        roh_cop = roh._carnot_cop(t) * roh._abschlag_heizen(t)
        guetegrad_heizen = geraete.scop / saisonzahl(roh_cop, gewichte)

    guetegrad_kuehlen = None
    if geraete.eer_pruefpunkte is None:
        t, gewichte = _bins(annahmen.referenzklima_kuehlen, heizen=False)
        roh_eer = roh._carnot_eer(t) * (1.0 - annahmen.teillast_abschlag)
        guetegrad_kuehlen = geraete.seer / saisonzahl(roh_eer, gewichte)

    return Effizienzmodell(geraete, annahmen, guetegrad_heizen, guetegrad_kuehlen)
