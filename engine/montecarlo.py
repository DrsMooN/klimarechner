"""Monte-Carlo-Wirtschaftlichkeit über Wetter, Preise, Verhalten und Technik.

Ablauf:

1. Es werden K Parametersätze gezogen (Heiz-Sollwert als Verschiebung der
   Heizgrenze, Kühl-Sollwert, Split-Schwelltemperatur, COP-Abschlag) und je
   Satz ein verrauschtes Haushaltsprofil.
2. Für jedes Wetterjahr werden Referenz und Variante für alle K Sätze auf
   einmal stündlich simuliert; gespeichert werden nur Jahressummen
   (Form ``[Y, K]``). Preise spielen dabei keine Rolle.
3. Jeder der n Pfade bekommt einen Parametersatz und zieht je Betriebsjahr
   ein Wetterjahr (mit Zurücklegen) sowie korrelierte log-normale
   Preispfade für Strom und Brennstoff.
4. Kosten, Cashflows und Kennzahlen werden vektorisiert über alle Pfade
   gerechnet und als Perzentile zusammengefasst.

Aufruf zum Ausprobieren::

    python -m engine.montecarlo --synthetisch
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, fields

import numpy as np

from engine.economics import (
    Preispfad,
    amortisationszeit,
    interner_zinsfuss,
    jahreskosten,
    kuehlkosten,
)
from engine.inputs import Konfiguration, lade_konfiguration
from engine.loads import haushaltslast_kw
from engine.simulation import (
    Jahressummen,
    Variation,
    haushaltsprofil,
    pruefe_plausibilitaet,
    simuliere,
)
from engine.weather import Wetterjahr, lade_wetterjahre, synthetisches_wetterjahr

PERZENTILE = (10, 50, 90)


def ziehe_variation(konfig: Konfiguration, anzahl: int, rng: np.random.Generator) -> Variation:
    """Zufällige Parametersätze für Verhalten und Technik."""
    e, a = konfig.eingaben, konfig.annahmen
    sigma_soll = a.verhalten.sigma_t_soll_k
    return Variation(
        heizgrenze_delta_k=rng.normal(0.0, sigma_soll, anzahl),
        t_soll_kuehlen_c=rng.normal(e.gebaeude.t_soll_kuehlen_c, sigma_soll, anzahl),
        schwelltemperatur_c=rng.normal(
            e.nutzung.split_schwelltemperatur_c, a.verhalten.sigma_schwelle_k, anzahl
        ),
        cop_abschlag=rng.uniform(a.technik.cop_abschlag_min, a.technik.cop_abschlag_max, anzahl),
    )


def ziehe_preispfade(konfig: Konfiguration, pfade: int, rng: np.random.Generator) -> Preispfad:
    """Korrelierte log-normale Preispfade, Form ``[pfade, jahre]``.

    Das erste Betriebsjahr nutzt die Startpreise. Danach wächst jeder Preis
    je Jahr um den Faktor ``exp(r)`` mit ``r ~ N(ln(1+drift) − σ²/2, σ)``,
    sodass der Erwartungswert dem deterministischen Pfad entspricht.
    """
    e, s = konfig.eingaben, konfig.preisszenario
    n = e.rahmen.betrachtungszeitraum_jahre

    z_strom = rng.standard_normal((pfade, n - 1))
    z_unabh = rng.standard_normal((pfade, n - 1))
    z_brenn = s.korrelation * z_strom + np.sqrt(1 - s.korrelation**2) * z_unabh

    def pfad(start: float, drift: float, sigma: float, z: np.ndarray) -> np.ndarray:
        r = np.log1p(drift) - sigma**2 / 2 + sigma * z
        wachstum = np.exp(np.concatenate([np.zeros((pfade, 1)), np.cumsum(r, axis=1)], axis=1))
        return start * wachstum

    co2_eur_kwh = (
        konfig.annahmen.brennstoffe.co2_kg_kwh[e.bestand.heiztraeger]
        * s.co2_aufschlag_eur_t
        / 1000.0
    )
    return Preispfad(
        strom=pfad(e.preise.strom_bezug_eur_kwh, s.strom_drift, s.strom_volatilitaet, z_strom),
        brennstoff=pfad(
            e.preise.brennstoff_eur_kwh, s.brennstoff_drift, s.brennstoff_volatilitaet, z_brenn
        )
        + co2_eur_kwh,
        einspeisung=np.full((pfade, n), e.preise.einspeiseverguetung_eur_kwh),
    )


def _staple(summen: Sequence[Jahressummen]) -> Jahressummen:
    """Liste von Jahressummen ``[K]`` je Wetterjahr zu Arrays ``[Y, K]``."""
    return Jahressummen(
        **{f.name: np.stack([getattr(s, f.name) for s in summen]) for f in fields(Jahressummen)}
    )


def _auswahl(summen: Jahressummen, wetter_idx: np.ndarray, satz_idx: np.ndarray) -> Jahressummen:
    """Jahressummen je Pfad und Betriebsjahr, Form ``[n, jahre]``."""
    return Jahressummen(
        **{
            f.name: getattr(summen, f.name)[wetter_idx, satz_idx[:, None]]
            for f in fields(Jahressummen)
        }
    )


def _summe_ueber_jahre(summen: Jahressummen) -> Jahressummen:
    return Jahressummen(
        **{f.name: getattr(summen, f.name).sum(axis=-1) for f in fields(Jahressummen)}
    )


@dataclass(frozen=True)
class Energiesimulation:
    """Gecachte Jahressummen aller Wetterjahre × Parametersätze."""

    referenz: Jahressummen
    variante: Jahressummen
    variation: Variation
    wetterjahre: tuple[int, ...]


def simuliere_energie(
    konfig: Konfiguration, wetterjahre: Sequence[Wetterjahr], rng: np.random.Generator
) -> Energiesimulation:
    anzahl = konfig.eingaben.montecarlo.parametersaetze
    variation = ziehe_variation(konfig, anzahl, rng)
    haushalt = haushaltslast_kw(
        haushaltsprofil(konfig),
        konfig.eingaben.haushalt.jahresverbrauch_kwh,
        rng,
        konfig.annahmen.haushalt.rauschen_sigma,
        anzahl=anzahl,
    )
    referenz, variante = [], []
    for wetter in wetterjahre:
        for liste, mit_split in ((referenz, False), (variante, True)):
            ergebnis = simuliere(
                konfig, wetter, mit_split=mit_split, haushalt=haushalt, variation=variation
            )
            liste.append(ergebnis.summen())
    return Energiesimulation(
        referenz=_staple(referenz),
        variante=_staple(variante),
        variation=variation,
        wetterjahre=tuple(w.jahr for w in wetterjahre),
    )


@dataclass(frozen=True)
class MonteCarloErgebnis:
    investition: float
    zins: float
    preise: Preispfad
    """Form ``[n, jahre]``"""
    ersparnis: np.ndarray
    """Cashflow je Pfad und Betriebsjahr, Form ``[n, jahre]``"""
    kuehlkosten: np.ndarray
    referenz: Jahressummen
    """Energiemengen je Pfad, summiert über alle Betriebsjahre"""
    variante: Jahressummen

    @property
    def pfade(self) -> int:
        return self.ersparnis.shape[0]

    @property
    def kumulierter_cashflow(self) -> np.ndarray:
        """Form ``[n, jahre + 1]``, Spalte 0 = Investition."""
        start = np.full((self.pfade, 1), -self.investition)
        return np.concatenate([start, start + np.cumsum(self.ersparnis, axis=1)], axis=1)

    @property
    def ersparnis_heizen(self) -> np.ndarray:
        """Ersparnis ohne Kühlkosten (inklusive Wartung), Form ``[n, jahre]``.

        Je Pfad und Jahr gilt ``ersparnis == ersparnis_heizen − kuehlkosten``.
        """
        return self.ersparnis + self.kuehlkosten

    @staticmethod
    def kumuliert(werte: np.ndarray) -> np.ndarray:
        """Kumulierte Summe ab null, Form ``[n, jahre + 1]``, Spalte 0 = 0."""
        return np.concatenate([np.zeros((werte.shape[0], 1)), np.cumsum(werte, axis=1)], axis=1)

    @property
    def barwert(self) -> np.ndarray:
        t = np.arange(1, self.ersparnis.shape[1] + 1)
        return -self.investition + (self.ersparnis / (1 + self.zins) ** t).sum(axis=1)

    @property
    def interner_zinsfuss(self) -> np.ndarray:
        """Je Pfad, ``nan`` wenn es keinen gibt."""
        werte = [interner_zinsfuss(self.investition, cf) for cf in self.ersparnis]
        return np.array([np.nan if w is None else w for w in werte])

    @property
    def amortisation_jahre(self) -> np.ndarray:
        """Je Pfad, ``inf`` wenn keine Amortisation im Betrachtungszeitraum."""
        werte = [amortisationszeit(self.investition, cf) for cf in self.ersparnis]
        return np.array([np.inf if w is None else w for w in werte])

    @property
    def wahrscheinlichkeit_amortisation(self) -> float:
        return float(np.isfinite(self.amortisation_jahre).mean())

    def warnungen(self, konfig: Konfiguration) -> list[str]:
        """Plausibilitätswarnungen für den Median der Pfade."""
        return pruefe_plausibilitaet(konfig, float(np.median(self.variante.jaz_heizen)))

    def kennzahlen(self) -> dict[str, dict[int, float] | float]:
        """Perzentile (P10/P50/P90) der Kern- und Energiekennzahlen.

        Perzentile ohne Interpolation (``inverted_cdf``), damit „keine
        Amortisation“ (``inf``) und „kein Zinsfuß“ (``nan`` → ``-inf``)
        sauber erhalten bleiben.
        """

        def p(werte: np.ndarray) -> dict[int, float]:
            return {q: float(np.percentile(werte, q, method="inverted_cdf")) for q in PERZENTILE}

        irr = np.nan_to_num(self.interner_zinsfuss, nan=-np.inf)
        jahre = self.ersparnis.shape[1]
        v, r = self.variante, self.referenz
        return {
            "barwert_eur": p(self.barwert),
            "interner_zinsfuss": p(irr),
            "amortisation_jahre": p(self.amortisation_jahre),
            "wahrscheinlichkeit_amortisation": self.wahrscheinlichkeit_amortisation,
            "ersparnis_jahr1_eur": p(self.ersparnis[:, 0]),
            "ersparnis_summe_eur": p(self.ersparnis.sum(axis=1)),
            "ersparnis_heizen_summe_eur": p(self.ersparnis_heizen.sum(axis=1)),
            "kuehlkosten_summe_eur": p(self.kuehlkosten.sum(axis=1)),
            "kuehlkosten_eur_jahr": p(self.kuehlkosten.mean(axis=1)),
            "jaz_heizen": p(v.jaz_heizen),
            "pv_anteil_split_heizen": p(v.pv_anteil_split_heizen),
            "eingesparter_brennstoff_kwh_jahr": p((r.brennstoff - v.brennstoff) / jahre),
            "autarkiegrad": p(v.autarkiegrad),
            "eigenverbrauchsquote": p(v.eigenverbrauchsquote),
        }


def rechne(
    konfig: Konfiguration,
    wetterjahre: Sequence[Wetterjahr],
    energie: Energiesimulation | None = None,
) -> MonteCarloErgebnis:
    """Monte-Carlo-Wirtschaftlichkeit; gleicher Seed → identische Ergebnisse.

    Eine vorhandene ``energie`` (z. B. aus einem früheren Lauf mit gleichen
    Eingaben) wird wiederverwendet, dann werden nur Preise und Ziehungen neu
    gerechnet.
    """
    e = konfig.eingaben
    rng = np.random.default_rng(e.montecarlo.seed)
    if energie is None:
        energie = simuliere_energie(konfig, wetterjahre, rng)

    n, jahre = e.montecarlo.pfade, e.rahmen.betrachtungszeitraum_jahre
    satz_idx = rng.integers(0, energie.variation.anzahl, n)
    wetter_idx = rng.integers(0, len(energie.wetterjahre), (n, jahre))
    preise = ziehe_preispfade(konfig, n, rng)

    ref = _auswahl(energie.referenz, wetter_idx, satz_idx)
    var = _auswahl(energie.variante, wetter_idx, satz_idx)
    p = (preise.strom, preise.brennstoff, preise.einspeisung)
    kosten_ref = jahreskosten(ref, *p).gesamt
    kosten_var = jahreskosten(var, *p, wartung=e.kosten.wartung_eur_jahr).gesamt

    return MonteCarloErgebnis(
        investition=e.kosten.investition_gesamt_eur,
        zins=e.rahmen.kalkulationszins,
        preise=preise,
        ersparnis=kosten_ref - kosten_var,
        kuehlkosten=kuehlkosten(var, preise.strom, preise.einspeisung),
        referenz=_summe_ueber_jahre(ref),
        variante=_summe_ueber_jahre(var),
    )


def _format(wert: float) -> str:
    if np.isposinf(wert):
        return "nie"
    if np.isneginf(wert):
        return "–"
    return f"{wert:10.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte-Carlo mit Default-Eingaben")
    parser.add_argument(
        "--synthetisch", action="store_true", help="synthetisches Wetter statt PVGIS"
    )
    args = parser.parse_args()

    konfig = lade_konfiguration()
    if args.synthetisch:
        breite = konfig.eingaben.standort.breitengrad
        wetterjahre = [synthetisches_wetterjahr(2000 + i, breite, seed=i) for i in range(5)]
    else:
        wetterjahre = lade_wetterjahre(konfig)

    ergebnis = rechne(konfig, wetterjahre)
    print(f"{ergebnis.pfade} Pfade, {len(wetterjahre)} Wetterjahre\n")
    for warnung in ergebnis.warnungen(konfig):
        print(f"Warnung: {warnung}\n")
    print(f"  {'Kennzahl':34s} {'P10':>10} {'P50':>10} {'P90':>10}")
    for name, wert in ergebnis.kennzahlen().items():
        if isinstance(wert, dict):
            print(f"  {name:34s} " + " ".join(f"{_format(wert[q]):>10}" for q in PERZENTILE))
        else:
            print(f"  {name:34s} {wert:10.2f}")


if __name__ == "__main__":
    main()
