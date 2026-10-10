"""Regelbasierter Eigenverbrauch mit optionalem Speicher und stündliche Energiebilanz.

Strategie: PV deckt zuerst die Last, Überschuss lädt den Speicher, der Rest
wird eingespeist. Ein Defizit wird zuerst aus dem Speicher, dann aus dem Netz
gedeckt. Der Speicher wird nur aus PV geladen.

Alle Flüsse sind AC-seitige Energien je Stunde [kWh]. Die Funktion rechnet
vektorisiert über führende Dimensionen (z. B. Monte-Carlo-Pfade); die Zeit
ist immer die letzte Achse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.inputs import Speicher


@dataclass(frozen=True)
class Energiebilanz:
    last: np.ndarray
    pv: np.ndarray
    direktverbrauch: np.ndarray
    """PV, die in derselben Stunde direkt die Last deckt"""
    ladung: np.ndarray
    entladung: np.ndarray
    verluste: np.ndarray
    """Lade- und Entladeverluste des Speichers"""
    einspeisung: np.ndarray
    netzbezug: np.ndarray
    soc_kwh: np.ndarray
    """Speicherinhalt am Ende der Stunde"""

    @property
    def eigenverbrauchsquote(self) -> float:
        pv = self.pv.sum()
        return float((pv - self.einspeisung.sum()) / pv) if pv > 0 else 0.0

    @property
    def autarkiegrad(self) -> float:
        last = self.last.sum()
        return float(1.0 - self.netzbezug.sum() / last) if last > 0 else 0.0


def energiebilanz(pv_kw: np.ndarray, last_kw: np.ndarray, speicher: Speicher) -> Energiebilanz:
    pv_kw, last_kw = np.broadcast_arrays(np.asarray(pv_kw, float), np.asarray(last_kw, float))
    direkt = np.minimum(pv_kw, last_kw)
    ueberschuss = pv_kw - direkt
    defizit = last_kw - direkt

    ladung = np.zeros_like(pv_kw)
    entladung = np.zeros_like(pv_kw)
    soc = np.zeros_like(pv_kw)

    if speicher.aktiv:
        eta = speicher.wirkungsgrad
        e_min = speicher.soc_min * speicher.kapazitaet_kwh
        e_max = speicher.soc_max * speicher.kapazitaet_kwh
        inhalt = np.full(pv_kw.shape[:-1], e_min)
        for t in range(pv_kw.shape[-1]):
            laden = np.minimum.reduce(
                [
                    ueberschuss[..., t],
                    np.full_like(inhalt, speicher.ladeleistung_kw),
                    (e_max - inhalt) / eta,
                ]
            )
            laden = np.maximum(laden, 0.0)
            inhalt = inhalt + laden * eta
            entladen = np.minimum.reduce(
                [
                    defizit[..., t],
                    np.full_like(inhalt, speicher.entladeleistung_kw),
                    (inhalt - e_min) * eta,
                ]
            )
            entladen = np.maximum(entladen, 0.0)
            inhalt = inhalt - entladen / eta
            ladung[..., t] = laden
            entladung[..., t] = entladen
            soc[..., t] = inhalt
        verluste = ladung * (1.0 - eta) + entladung * (1.0 / eta - 1.0)
    else:
        verluste = np.zeros_like(pv_kw)

    return Energiebilanz(
        last=last_kw,
        pv=pv_kw,
        direktverbrauch=direkt,
        ladung=ladung,
        entladung=entladung,
        verluste=verluste,
        einspeisung=ueberschuss - ladung,
        netzbezug=defizit - entladung,
        soc_kwh=soc,
    )
