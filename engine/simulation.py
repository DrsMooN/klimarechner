"""Stündliche Jahressimulation für ein Wetterjahr.

Rechnet stündlich Heiz- und Kühlbedarf, die Aufteilung zwischen Split und
Bestandsheizung, den Strombedarf der Split-Geräte, Haushaltsstrom, PV,
Speicher und die Energiebilanz – jeweils für die Referenz (ohne Split) und
die Variante (mit Split und ggf. neuer PV/Speicher).

Mit einer :class:`Variation` rechnet die Simulation viele Parametersätze
(Sollwerte, Schwelltemperatur, COP-Abschlag) auf einmal; alle Stundenreihen
haben dann die Form ``[K, 8760]``.

Aufruf zum Ausprobieren::

    python -m engine.simulation --synthetisch
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np

from engine.heatpump import erstelle_effizienzmodell
from engine.inputs import Ausbaustatus, Heizstrategie, Konfiguration, lade_konfiguration
from engine.loads import (
    haushaltslast_kw,
    heizbedarf_kw,
    kuehlbedarf_kw,
    lade_lastprofil,
    normiertes_lastprofil,
    waermeverlustkoeffizient_w_k,
)
from engine.pv_battery import Energiebilanz, energiebilanz
from engine.weather import Wetterjahr, lade_wetterjahre, synthetisches_wetterjahr

logger = logging.getLogger(__name__)


def sicherer_anteil(zaehler, nenner):
    """Sicherer Quotient, 0 bei Nenner 0 (Skalar oder Array)."""
    zaehler, nenner = np.asarray(zaehler, float), np.asarray(nenner, float)
    form = np.broadcast(zaehler, nenner).shape
    ergebnis = np.divide(zaehler, nenner, out=np.zeros(form), where=nenner > 0)
    return ergebnis if ergebnis.ndim else float(ergebnis)


@dataclass(frozen=True)
class Variation:
    """Parametersätze für die Monte-Carlo-Simulation, je Feld ein Array der Länge K."""

    heizgrenze_delta_k: np.ndarray
    """Verschiebung der Heizgrenze (entspricht der Abweichung des Heiz-Sollwerts)"""
    t_soll_kuehlen_c: np.ndarray
    schwelltemperatur_c: np.ndarray
    cop_abschlag: np.ndarray

    @property
    def anzahl(self) -> int:
        return len(self.cop_abschlag)

    @classmethod
    def neutral(cls, konfig: Konfiguration, anzahl: int = 1) -> Variation:
        e = konfig.eingaben
        return cls(
            heizgrenze_delta_k=np.zeros(anzahl),
            t_soll_kuehlen_c=np.full(anzahl, e.gebaeude.t_soll_kuehlen_c),
            schwelltemperatur_c=np.full(anzahl, e.nutzung.split_schwelltemperatur_c),
            cop_abschlag=np.zeros(anzahl),
        )


@dataclass(frozen=True)
class Jahressummen:
    """Jahressummen eines Falls in kWh; Skalare oder Arrays der Länge K."""

    heizbedarf: np.ndarray | float
    split_waerme: np.ndarray | float
    bestand_waerme: np.ndarray | float
    unterdeckung: np.ndarray | float
    brennstoff: np.ndarray | float
    kuehlwaerme: np.ndarray | float
    split_strom_heizen: np.ndarray | float
    split_strom_heizen_aus_pv: np.ndarray | float
    """Split-Heizstrom aus PV direkt oder über den Speicher"""
    split_strom_kuehlen: np.ndarray | float
    kuehlstrom_netz: np.ndarray | float
    """Kühlstrom anteilig am Netzbezug je Stunde"""
    haushalt: np.ndarray | float
    pv: np.ndarray | float
    last: np.ndarray | float
    netzbezug: np.ndarray | float
    einspeisung: np.ndarray | float

    @property
    def jaz_heizen(self):
        return sicherer_anteil(self.split_waerme, self.split_strom_heizen)

    @property
    def pv_anteil_split_heizen(self):
        return sicherer_anteil(self.split_strom_heizen_aus_pv, self.split_strom_heizen)

    @property
    def eigenverbrauchsquote(self):
        return sicherer_anteil(np.asarray(self.pv) - self.einspeisung, self.pv)

    @property
    def autarkiegrad(self):
        last = np.asarray(self.last)
        wert = np.where(last > 0, 1.0 - sicherer_anteil(self.netzbezug, last), 0.0)
        return wert if wert.ndim else float(wert)


@dataclass(frozen=True)
class Jahresergebnis:
    """Stündliche Ergebnisse eines Falls (kWh je Stunde, Form ``[..., 8760]``)."""

    wetterjahr: int
    heizbedarf: np.ndarray
    split_waerme: np.ndarray
    bestand_waerme: np.ndarray
    unterdeckung: np.ndarray
    """Nicht gedeckter Wärmebedarf (nur Strategie Vollheizung Teilräume)"""
    kuehlwaerme: np.ndarray
    split_strom_heizen: np.ndarray
    split_strom_kuehlen: np.ndarray
    haushalt: np.ndarray
    bilanz: Energiebilanz
    kesselwirkungsgrad: float

    @property
    def split_strom(self) -> np.ndarray:
        return self.split_strom_heizen + self.split_strom_kuehlen

    @property
    def brennstoff_kwh(self) -> float:
        return float(self.bestand_waerme.sum() / self.kesselwirkungsgrad)

    @property
    def jaz_heizen(self) -> float:
        return float(self.summen().jaz_heizen)

    @property
    def pv_anteil_split_heizen(self) -> float:
        """Anteil des Split-Heizstroms, der aus PV (direkt oder über Speicher) kommt."""
        return float(self.summen().pv_anteil_split_heizen)

    def herkunft_split_strom_heizen(self) -> dict[str, np.ndarray]:
        """Split-Heizstrom je Stunde nach Quelle (PV direkt, Speicher, Netz).

        Jede Quelle wird anteilig nach dem Split-Anteil an der Gesamtlast
        der Stunde zugeordnet.
        """
        b = self.bilanz
        anteil = sicherer_anteil(self.split_strom_heizen, b.last)
        return {
            "pv": anteil * b.direktverbrauch,
            "speicher": anteil * b.entladung,
            "netz": anteil * b.netzbezug,
        }

    def summen(self) -> Jahressummen:
        """Jahressummen; bei ``[K, 8760]``-Reihen je Parametersatz."""
        b = self.bilanz
        herkunft = self.herkunft_split_strom_heizen()
        kuehl_netz = sicherer_anteil(self.split_strom_kuehlen, b.last) * b.netzbezug

        def s(x):
            summe = np.asarray(x).sum(axis=-1)
            return summe if summe.ndim else float(summe)

        return Jahressummen(
            heizbedarf=s(self.heizbedarf),
            split_waerme=s(self.split_waerme),
            bestand_waerme=s(self.bestand_waerme),
            unterdeckung=s(self.unterdeckung),
            brennstoff=s(self.bestand_waerme) / self.kesselwirkungsgrad,
            kuehlwaerme=s(self.kuehlwaerme),
            split_strom_heizen=s(self.split_strom_heizen),
            split_strom_heizen_aus_pv=s(herkunft["pv"] + herkunft["speicher"]),
            split_strom_kuehlen=s(self.split_strom_kuehlen),
            kuehlstrom_netz=s(kuehl_netz),
            haushalt=s(self.haushalt),
            pv=s(b.pv),
            last=s(b.last),
            netzbezug=s(b.netzbezug),
            einspeisung=s(b.einspeisung),
        )

    def kennzahlen(self) -> dict[str, float]:
        s = self.summen()
        return {
            "heizbedarf_kwh": s.heizbedarf,
            "split_waerme_kwh": s.split_waerme,
            "bestand_waerme_kwh": s.bestand_waerme,
            "unterdeckung_kwh": s.unterdeckung,
            "brennstoff_kwh": s.brennstoff,
            "kuehlwaerme_kwh": s.kuehlwaerme,
            "split_strom_heizen_kwh": s.split_strom_heizen,
            "split_strom_kuehlen_kwh": s.split_strom_kuehlen,
            "jaz_heizen": s.jaz_heizen,
            "pv_anteil_split_heizen": s.pv_anteil_split_heizen,
            "haushalt_kwh": s.haushalt,
            "pv_kwh": s.pv,
            "netzbezug_kwh": s.netzbezug,
            "einspeisung_kwh": s.einspeisung,
            "eigenverbrauchsquote": s.eigenverbrauchsquote,
            "autarkiegrad": s.autarkiegrad,
        }


def pruefe_plausibilitaet(konfig: Konfiguration, jaz_heizen: float) -> list[str]:
    """Prüft Kennzahlen gegen die Plausibilitätsbereiche und loggt Abweichungen.

    Gibt die Warnungen zurück (leer, wenn alles plausibel ist oder der Split
    gar nicht heizt).
    """
    warnungen = []
    unten, oben = konfig.annahmen.plausibilitaet.jaz_heizen
    if jaz_heizen > 0 and not unten <= jaz_heizen <= oben:
        warnungen.append(
            f"JAZ Heizen {jaz_heizen:.2f} liegt außerhalb des plausiblen Bereichs "
            f"{unten:.1f}–{oben:.1f} für Luft-Luft-Geräte".replace(".", ",")
        )
    for warnung in warnungen:
        logger.warning(warnung)
    return warnungen


def haushaltsprofil(konfig: Konfiguration) -> np.ndarray:
    """Normiertes Standardlastprofil (Summe 1) aus der Konfiguration."""
    return normiertes_lastprofil(lade_lastprofil(konfig.annahmen.haushalt.lastprofil))


def simuliere(
    konfig: Konfiguration,
    wetter: Wetterjahr,
    *,
    mit_split: bool,
    haushalt: np.ndarray,
    variation: Variation | None = None,
) -> Jahresergebnis:
    """Kern der Simulation mit vorgegebener Haushaltslast.

    Ohne ``variation`` sind alle Reihen eindimensional (8.760), mit
    ``variation`` haben sie die Form ``[K, 8760]``; ``haushalt`` muss dazu
    broadcastbar sein.
    """
    e, a = konfig.eingaben, konfig.annahmen
    t = wetter.t_aussen_c

    if variation is None:
        heizgrenze = e.gebaeude.heizgrenztemperatur_c
        t_soll_kuehlen = e.gebaeude.t_soll_kuehlen_c
        schwelle = e.nutzung.split_schwelltemperatur_c
        cop_abschlag = 0.0
        form = t.shape
    else:
        heizgrenze = e.gebaeude.heizgrenztemperatur_c + variation.heizgrenze_delta_k[:, None]
        t_soll_kuehlen = variation.t_soll_kuehlen_c[:, None]
        schwelle = variation.schwelltemperatur_c[:, None]
        cop_abschlag = variation.cop_abschlag[:, None]
        form = (variation.anzahl, len(t))

    h = waermeverlustkoeffizient_w_k(konfig)
    bedarf = np.broadcast_to(heizbedarf_kw(t, h, heizgrenze), form)

    null = np.zeros(form)
    split_waerme, unterdeckung, strom_heizen = null, null, null
    kuehlwaerme, strom_kuehlen = null, null
    if mit_split and e.geraete.anzahl_innengeraete > 0:
        modell = erstelle_effizienzmodell(e.geraete, a.waermepumpe)
        deckbar = bedarf * e.gebaeude.split_anteil
        split_waerme = np.minimum(deckbar, modell.max_heizleistung_kw(t))
        strategie = e.nutzung.heizstrategie
        if strategie == Heizstrategie.UEBERGANGSZEIT:
            split_waerme = np.where(t >= schwelle, split_waerme, 0.0)
        if strategie == Heizstrategie.VOLLHEIZUNG_TEILRAEUME:
            unterdeckung = deckbar - split_waerme
        strom_heizen = split_waerme / modell.cop(t, cop_abschlag)

        if e.nutzung.kuehlen_aktiv:
            kuehl = kuehlbedarf_kw(
                t,
                wetter.einstrahlung_w_m2,
                h * e.gebaeude.split_anteil,
                t_soll_kuehlen,
                a.kuehlen.kuehlgrenztemperatur_c,
                a.kuehlen.solarfaktor_m2,
                a.kuehlen.interne_lasten_w,
                a.kuehlen.zeitkonstante_solar_h,
            )
            kuehlwaerme = np.broadcast_to(np.minimum(kuehl, modell.max_kuehlleistung_kw(t)), form)
            strom_kuehlen = kuehlwaerme / modell.eer(t, cop_abschlag)

    bestand_waerme = bedarf - split_waerme - unterdeckung

    pv_dabei = mit_split or e.pv.status == Ausbaustatus.VORHANDEN
    speicher_dabei = mit_split or e.speicher.status == Ausbaustatus.VORHANDEN
    pv = np.broadcast_to(wetter.pv_kw(e.pv), form) if pv_dabei else null
    speicher = (
        e.speicher
        if speicher_dabei
        else e.speicher.model_copy(update={"status": Ausbaustatus.KEINE})
    )
    haushalt = np.broadcast_to(haushalt, form)
    bilanz = energiebilanz(pv, haushalt + strom_heizen + strom_kuehlen, speicher)

    return Jahresergebnis(
        wetterjahr=wetter.jahr,
        heizbedarf=bedarf,
        split_waerme=split_waerme,
        bestand_waerme=bestand_waerme,
        unterdeckung=unterdeckung,
        kuehlwaerme=kuehlwaerme,
        split_strom_heizen=strom_heizen,
        split_strom_kuehlen=strom_kuehlen,
        haushalt=haushalt,
        bilanz=bilanz,
        kesselwirkungsgrad=e.bestand.kesselwirkungsgrad,
    )


def simuliere_jahr(
    konfig: Konfiguration,
    wetter: Wetterjahr,
    *,
    mit_split: bool,
    normprofil: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> Jahresergebnis:
    """Ein Jahr für Referenz (``mit_split=False``) oder Variante (``True``).

    PV und Speicher mit Status "neu" gehören nur zur Variante. Mit ``rng``
    bekommt der Haushaltsstrom Rauschen, ohne bleibt alles deterministisch.
    """
    if normprofil is None:
        normprofil = haushaltsprofil(konfig)
    haushalt = haushaltslast_kw(
        normprofil,
        konfig.eingaben.haushalt.jahresverbrauch_kwh,
        rng,
        konfig.annahmen.haushalt.rauschen_sigma,
    )
    return simuliere(konfig, wetter, mit_split=mit_split, haushalt=haushalt)


def _ausgabe(name: str, ergebnis: Jahresergebnis) -> None:
    print(f"\n{name} (Wetterjahr {ergebnis.wetterjahr})")
    for schluessel, wert in ergebnis.kennzahlen().items():
        print(f"  {schluessel:26s} {wert:12.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Jahressimulation mit Default-Eingaben")
    parser.add_argument(
        "--synthetisch", action="store_true", help="synthetisches Wetter statt PVGIS"
    )
    args = parser.parse_args()

    konfig = lade_konfiguration()
    if args.synthetisch:
        wetter = synthetisches_wetterjahr(breitengrad=konfig.eingaben.standort.breitengrad)
    else:
        wetter = lade_wetterjahre(konfig)[-1]
    _ausgabe("Referenz", simuliere_jahr(konfig, wetter, mit_split=False))
    variante = simuliere_jahr(konfig, wetter, mit_split=True)
    _ausgabe("Variante", variante)
    for warnung in pruefe_plausibilitaet(konfig, variante.jaz_heizen):
        print(f"\nWarnung: {warnung}")


if __name__ == "__main__":
    main()
