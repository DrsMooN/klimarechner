"""Wirtschaftlichkeit: Kosten je Jahr, Cashflows, Barwert, Zinsfuß, Amortisation.

Referenz und Variante werden mit denselben Wetter- und Preispfaden
gerechnet. Ersparnis = Kosten Referenz − Kosten Variante. Kühlen ist in den
Stromkosten der Variante enthalten (Mehrkosten, keine Ersparnis) und wird
zusätzlich separat ausgewiesen.

Die Energiesimulation hängt nicht von Preisen ab; Preise werden erst hier
auf die Jahresergebnisse angewendet.

Aufruf zum Ausprobieren::

    python -m engine.economics --synthetisch
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from engine.inputs import Konfiguration, lade_konfiguration
from engine.simulation import Jahresergebnis, Jahressummen, simuliere_jahr
from engine.weather import lade_wetterjahre, synthetisches_wetterjahr


@dataclass(frozen=True)
class Preispfad:
    """Preise je Betriebsjahr (Index 0 = erstes Jahr) in €/kWh."""

    strom: np.ndarray
    brennstoff: np.ndarray
    """inklusive CO2-Aufschlag des Szenarios"""
    einspeisung: np.ndarray

    @property
    def jahre(self) -> int:
        return len(self.strom)


def deterministischer_preispfad(konfig: Konfiguration) -> Preispfad:
    """Preise wachsen mit der Drift des gewählten Szenarios, ohne Streuung.

    Die Einspeisevergütung bleibt konstant (fester Satz über die Laufzeit).
    """
    e = konfig.eingaben
    szenario = konfig.preisszenario
    n = e.rahmen.betrachtungszeitraum_jahre
    t = np.arange(n)
    co2_eur_kwh = (
        konfig.annahmen.brennstoffe.co2_kg_kwh[e.bestand.heiztraeger]
        * szenario.co2_aufschlag_eur_t
        / 1000.0
    )
    return Preispfad(
        strom=e.preise.strom_bezug_eur_kwh * (1 + szenario.strom_drift) ** t,
        brennstoff=e.preise.brennstoff_eur_kwh * (1 + szenario.brennstoff_drift) ** t + co2_eur_kwh,
        einspeisung=np.full(n, e.preise.einspeiseverguetung_eur_kwh),
    )


@dataclass(frozen=True)
class Jahreskosten:
    strom: float | np.ndarray
    """Netzbezug × Preis − Einspeisung × Vergütung"""
    brennstoff: float | np.ndarray
    wartung: float

    @property
    def gesamt(self) -> float | np.ndarray:
        return self.strom + self.brennstoff + self.wartung


def jahreskosten(
    summen: Jahressummen,
    strompreis,
    brennstoffpreis,
    einspeiseverguetung,
    wartung: float = 0.0,
) -> Jahreskosten:
    """Kosten eines Jahres; funktioniert mit Skalaren und (broadcastbaren) Arrays."""
    return Jahreskosten(
        strom=summen.netzbezug * strompreis - summen.einspeisung * einspeiseverguetung,
        brennstoff=summen.brennstoff * brennstoffpreis,
        wartung=wartung,
    )


def kuehlkosten(summen: Jahressummen, strompreis, einspeiseverguetung):
    """Stromkosten des Kühlens.

    Kühlstrom aus dem Netz kostet den Bezugspreis, Kühlstrom aus PV oder
    Speicher die entgangene Einspeisevergütung.
    """
    aus_pv = summen.split_strom_kuehlen - summen.kuehlstrom_netz
    return summen.kuehlstrom_netz * strompreis + aus_pv * einspeiseverguetung


def barwert(investition: float, cashflows: np.ndarray, zins: float) -> float:
    t = np.arange(1, len(cashflows) + 1)
    return float(-investition + (cashflows / (1 + zins) ** t).sum())


def interner_zinsfuss(investition: float, cashflows: np.ndarray) -> float | None:
    """Zins, bei dem der Barwert null ist; ``None``, wenn es keinen gibt."""
    if investition <= 0:
        return None
    unten, oben = -0.99, 10.0
    if barwert(investition, cashflows, unten) * barwert(investition, cashflows, oben) > 0:
        return None
    return float(brentq(lambda z: barwert(investition, cashflows, z), unten, oben))


def amortisationszeit(investition: float, cashflows: np.ndarray) -> float | None:
    """Jahre bis der kumulierte Cashflow die Investition deckt (linear interpoliert).

    ``None``, wenn das innerhalb des Betrachtungszeitraums nicht passiert.
    """
    if investition <= 0:
        return 0.0
    kumuliert = -investition
    for jahr, cf in enumerate(cashflows, start=1):
        if kumuliert + cf >= 0:
            return jahr - 1 + (-kumuliert / cf)
        kumuliert += cf
    return None


@dataclass(frozen=True)
class Wirtschaftlichkeit:
    investition: float
    preise: Preispfad
    kosten_referenz: list[Jahreskosten]
    kosten_variante: list[Jahreskosten]
    kuehlkosten: np.ndarray
    zins: float

    @property
    def ersparnis(self) -> np.ndarray:
        """Cashflow je Betriebsjahr (positiv = Variante günstiger)."""
        return np.array(
            [
                r.gesamt - v.gesamt
                for r, v in zip(self.kosten_referenz, self.kosten_variante, strict=True)
            ]
        )

    @property
    def kumulierter_cashflow(self) -> np.ndarray:
        """Kumulierter Cashflow ab Jahr 0 (Investition), Länge Jahre + 1."""
        return np.concatenate([[-self.investition], -self.investition + np.cumsum(self.ersparnis)])

    @property
    def barwert(self) -> float:
        return barwert(self.investition, self.ersparnis, self.zins)

    @property
    def interner_zinsfuss(self) -> float | None:
        return interner_zinsfuss(self.investition, self.ersparnis)

    @property
    def amortisation_jahre(self) -> float | None:
        return amortisationszeit(self.investition, self.ersparnis)

    @property
    def amortisation_dynamisch_jahre(self) -> float | None:
        """Amortisation mit abgezinsten Cashflows."""
        t = np.arange(1, self.preise.jahre + 1)
        return amortisationszeit(self.investition, self.ersparnis / (1 + self.zins) ** t)

    def kennzahlen(self) -> dict[str, float | None]:
        return {
            "investition_eur": self.investition,
            "ersparnis_jahr1_eur": float(self.ersparnis[0]),
            "ersparnis_summe_eur": float(self.ersparnis.sum()),
            "kuehlkosten_summe_eur": float(self.kuehlkosten.sum()),
            "barwert_eur": self.barwert,
            "interner_zinsfuss": self.interner_zinsfuss,
            "amortisation_jahre": self.amortisation_jahre,
            "amortisation_dynamisch_jahre": self.amortisation_dynamisch_jahre,
        }


def bewerte(
    konfig: Konfiguration,
    referenz: Sequence[Jahresergebnis],
    variante: Sequence[Jahresergebnis],
    preise: Preispfad | None = None,
) -> Wirtschaftlichkeit:
    """Wirtschaftlichkeit über den Betrachtungszeitraum.

    ``referenz`` und ``variante`` sind Jahresergebnisse je Wetterjahr in
    gleicher Reihenfolge; Betriebsjahr t nutzt Wetterjahr ``t mod len``.
    """
    if len(referenz) != len(variante) or not referenz:
        raise ValueError("Referenz und Variante brauchen gleich viele (≥ 1) Jahresergebnisse")
    preise = preise or deterministischer_preispfad(konfig)
    e = konfig.eingaben

    kosten_ref, kosten_var, kuehl = [], [], []
    for t in range(preise.jahre):
        ref = referenz[t % len(referenz)].summen()
        var = variante[t % len(variante)].summen()
        p = (preise.strom[t], preise.brennstoff[t], preise.einspeisung[t])
        kosten_ref.append(jahreskosten(ref, *p))
        kosten_var.append(jahreskosten(var, *p, wartung=e.kosten.wartung_eur_jahr))
        kuehl.append(kuehlkosten(var, preise.strom[t], preise.einspeisung[t]))

    return Wirtschaftlichkeit(
        investition=e.kosten.investition_gesamt_eur,
        preise=preise,
        kosten_referenz=kosten_ref,
        kosten_variante=kosten_var,
        kuehlkosten=np.array(kuehl),
        zins=e.rahmen.kalkulationszins,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Wirtschaftlichkeit mit Default-Eingaben")
    parser.add_argument(
        "--synthetisch", action="store_true", help="synthetisches Wetter statt PVGIS"
    )
    args = parser.parse_args()

    konfig = lade_konfiguration()
    if args.synthetisch:
        wetterjahre = [synthetisches_wetterjahr(breitengrad=konfig.eingaben.standort.breitengrad)]
    else:
        wetterjahre = lade_wetterjahre(konfig)
    referenz = [simuliere_jahr(konfig, w, mit_split=False) for w in wetterjahre]
    variante = [simuliere_jahr(konfig, w, mit_split=True) for w in wetterjahre]
    w = bewerte(konfig, referenz, variante)

    kopf = ["Jahr", "Strom €/kWh", "Brennst. €/kWh", "Ersparnis €", "kumuliert €"]
    print(f"{kopf[0]:>4} {kopf[1]:>12} {kopf[2]:>15} {kopf[3]:>12} {kopf[4]:>12}")
    for t, (ersparnis, kum) in enumerate(
        zip(w.ersparnis, w.kumulierter_cashflow[1:], strict=True), start=1
    ):
        print(
            f"{t:4d} {w.preise.strom[t - 1]:12.3f} {w.preise.brennstoff[t - 1]:15.3f}"
            f" {ersparnis:12.0f} {kum:12.0f}"
        )
    print()
    for schluessel, wert in w.kennzahlen().items():
        print(f"  {schluessel:30s} {'–' if wert is None else f'{wert:12.2f}'}")


if __name__ == "__main__":
    main()
