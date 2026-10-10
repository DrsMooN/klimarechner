"""Deterministische Jahressimulation für ein Wetterjahr.

Rechnet stündlich Heiz- und Kühlbedarf, die Aufteilung zwischen Split und
Bestandsheizung, den Strombedarf der Split-Geräte, Haushaltsstrom, PV,
Speicher und die Energiebilanz – jeweils für die Referenz (ohne Split) und
die Variante (mit Split und ggf. neuer PV/Speicher).

Aufruf zum Ausprobieren::

    python -m engine.simulation --synthetisch
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from engine.heatpump import Effizienzmodell, erstelle_effizienzmodell
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


@dataclass(frozen=True)
class Jahresergebnis:
    """Stündliche Ergebnisse eines Falls (alle Werte kWh je Stunde)."""

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
        strom = self.split_strom_heizen.sum()
        return float(self.split_waerme.sum() / strom) if strom > 0 else 0.0

    def herkunft_split_strom_heizen(self) -> dict[str, np.ndarray]:
        """Split-Heizstrom je Stunde nach Quelle (PV direkt, Speicher, Netz).

        Jede Quelle wird anteilig nach dem Split-Anteil an der Gesamtlast
        der Stunde zugeordnet.
        """
        b = self.bilanz
        anteil = np.divide(
            self.split_strom_heizen, b.last, out=np.zeros_like(b.last), where=b.last > 0
        )
        return {
            "pv": anteil * b.direktverbrauch,
            "speicher": anteil * b.entladung,
            "netz": anteil * b.netzbezug,
        }

    @property
    def pv_anteil_split_heizen(self) -> float:
        """Anteil des Split-Heizstroms, der aus PV (direkt oder über Speicher) kommt."""
        strom = self.split_strom_heizen.sum()
        if strom == 0:
            return 0.0
        herkunft = self.herkunft_split_strom_heizen()
        return float((herkunft["pv"].sum() + herkunft["speicher"].sum()) / strom)

    def kennzahlen(self) -> dict[str, float]:
        b = self.bilanz
        return {
            "heizbedarf_kwh": float(self.heizbedarf.sum()),
            "split_waerme_kwh": float(self.split_waerme.sum()),
            "bestand_waerme_kwh": float(self.bestand_waerme.sum()),
            "unterdeckung_kwh": float(self.unterdeckung.sum()),
            "brennstoff_kwh": self.brennstoff_kwh,
            "kuehlwaerme_kwh": float(self.kuehlwaerme.sum()),
            "split_strom_heizen_kwh": float(self.split_strom_heizen.sum()),
            "split_strom_kuehlen_kwh": float(self.split_strom_kuehlen.sum()),
            "jaz_heizen": self.jaz_heizen,
            "pv_anteil_split_heizen": self.pv_anteil_split_heizen,
            "haushalt_kwh": float(self.haushalt.sum()),
            "pv_kwh": float(b.pv.sum()),
            "netzbezug_kwh": float(b.netzbezug.sum()),
            "einspeisung_kwh": float(b.einspeisung.sum()),
            "eigenverbrauchsquote": b.eigenverbrauchsquote,
            "autarkiegrad": b.autarkiegrad,
        }


def _split_heizen(
    konfig: Konfiguration, wetter: Wetterjahr, bedarf: np.ndarray, modell: Effizienzmodell
) -> tuple[np.ndarray, np.ndarray]:
    """Wärme vom Split und nicht gedeckter Bedarf je Stunde."""
    e = konfig.eingaben
    deckbar = bedarf * e.gebaeude.split_anteil
    split = np.minimum(deckbar, modell.max_heizleistung_kw(wetter.t_aussen_c))
    strategie = e.nutzung.heizstrategie
    if strategie == Heizstrategie.UEBERGANGSZEIT:
        split = np.where(wetter.t_aussen_c >= e.nutzung.split_schwelltemperatur_c, split, 0.0)
    unterdeckung = np.zeros_like(split)
    if strategie == Heizstrategie.VOLLHEIZUNG_TEILRAEUME:
        unterdeckung = deckbar - split
    return split, unterdeckung


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
    e, a = konfig.eingaben, konfig.annahmen
    t = wetter.t_aussen_c

    h = waermeverlustkoeffizient_w_k(konfig)
    bedarf = heizbedarf_kw(t, h, e.gebaeude.heizgrenztemperatur_c)

    null = np.zeros_like(t)
    if mit_split and e.geraete.anzahl_innengeraete > 0:
        modell = erstelle_effizienzmodell(e.geraete, a.waermepumpe)
        split_waerme, unterdeckung = _split_heizen(konfig, wetter, bedarf, modell)
        strom_heizen = split_waerme / modell.cop(t)
        if e.nutzung.kuehlen_aktiv:
            kuehl = kuehlbedarf_kw(
                t,
                wetter.einstrahlung_w_m2,
                h * e.gebaeude.split_anteil,
                e.gebaeude.t_soll_kuehlen_c,
                a.kuehlen.kuehlgrenztemperatur_c,
                a.kuehlen.solarfaktor_m2,
                a.kuehlen.interne_lasten_w,
            )
            kuehlwaerme = np.minimum(kuehl, modell.max_kuehlleistung_kw(t))
            strom_kuehlen = kuehlwaerme / modell.eer(t)
        else:
            kuehlwaerme, strom_kuehlen = null, null
    else:
        split_waerme, unterdeckung, strom_heizen = null, null, null
        kuehlwaerme, strom_kuehlen = null, null

    bestand_waerme = bedarf - split_waerme - unterdeckung

    if normprofil is None:
        normprofil = normiertes_lastprofil(lade_lastprofil(a.haushalt.lastprofil))
    haushalt = haushaltslast_kw(
        normprofil, e.haushalt.jahresverbrauch_kwh, rng, a.haushalt.rauschen_sigma
    )

    pv_dabei = mit_split or e.pv.status == Ausbaustatus.VORHANDEN
    speicher_dabei = mit_split or e.speicher.status == Ausbaustatus.VORHANDEN
    pv = wetter.pv_kw(e.pv) if pv_dabei else null
    speicher = (
        e.speicher
        if speicher_dabei
        else e.speicher.model_copy(update={"status": Ausbaustatus.KEINE})
    )
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
    _ausgabe("Variante", simuliere_jahr(konfig, wetter, mit_split=True))


if __name__ == "__main__":
    main()
