"""Eingabemodelle des Klimarechners.

Alle Eingaben werden mit pydantic validiert. Die Defaults stehen zentral in
``config/defaults.yaml``; :func:`lade_konfiguration` liest sie ein und legt
optionale Überschreibungen (z. B. aus der Eingabemaske) darüber.

Die Datei trennt zwei Ebenen:

* :class:`Eingaben` – was der Installateur in der Maske angibt.
* :class:`Annahmen` – Tabellen, Szenarien und Modellfaktoren ("Erweitert").
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULTS_PFAD = Path(__file__).resolve().parent.parent / "config" / "defaults.yaml"


class Baualtersklasse(StrEnum):
    VOR_1949 = "vor_1949"
    B1949_1978 = "1949_1978"
    B1979_1994 = "1979_1994"
    B1995_2009 = "1995_2009"
    AB_2010 = "ab_2010"
    EFFIZIENZHAUS = "effizienzhaus"


class Heizstrategie(StrEnum):
    UEBERGANGSZEIT = "uebergangszeit"
    """Split heizt nur oberhalb der Schwelltemperatur, darunter die Bestandsheizung."""
    BEVORZUGT_SPLIT = "bevorzugt_split"
    """Split deckt so viel wie möglich, Bestandsheizung übernimmt den Rest."""
    VOLLHEIZUNG_TEILRAEUME = "vollheizung_teilraeume"
    """Split-Räume werden ausschließlich über Split beheizt."""


class Ausbaustatus(StrEnum):
    KEINE = "keine"
    VORHANDEN = "vorhanden"
    NEU = "neu"


class Heiztraeger(StrEnum):
    GAS = "gas"
    OEL = "oel"


class _Basis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _pruefe_kennlinie(punkte: dict[float, float] | None, name: str) -> dict[float, float] | None:
    if punkte is None:
        return None
    if len(punkte) < 2:
        raise ValueError(f"{name}: mindestens zwei Punkte nötig")
    if any(wert <= 0 for wert in punkte.values()):
        raise ValueError(f"{name}: alle Werte müssen größer 0 sein")
    return dict(sorted(punkte.items()))


# --- Eingaben ---------------------------------------------------------------


class Standort(_Basis):
    breitengrad: float = Field(ge=-90, le=90)
    laengengrad: float = Field(ge=-180, le=180)


class Gebaeude(_Basis):
    beheizte_flaeche_m2: float = Field(gt=0)
    split_flaeche_m2: float = Field(ge=0)
    baualtersklasse: Baualtersklasse
    t_soll_heizen_c: float = Field(ge=15, le=26)
    t_soll_kuehlen_c: float = Field(ge=20, le=30)
    heizgrenztemperatur_c: float = Field(ge=5, le=22)

    @model_validator(mode="after")
    def _plausibel(self) -> Self:
        if self.split_flaeche_m2 > self.beheizte_flaeche_m2:
            raise ValueError("Split-Fläche darf nicht größer als die beheizte Fläche sein")
        if self.t_soll_kuehlen_c < self.t_soll_heizen_c:
            raise ValueError("Sollwert Kühlen muss mindestens so hoch sein wie Sollwert Heizen")
        if self.heizgrenztemperatur_c > self.t_soll_heizen_c:
            raise ValueError("Heizgrenztemperatur darf nicht über dem Sollwert Heizen liegen")
        return self

    @property
    def split_anteil(self) -> float:
        """Anteil der beheizten Fläche, den die Split-Geräte versorgen können."""
        return self.split_flaeche_m2 / self.beheizte_flaeche_m2


class Geraete(_Basis):
    anzahl_innengeraete: int = Field(ge=0, le=12)
    nennleistung_heizen_kw: float = Field(ge=0, description="je Innengerät")
    nennleistung_kuehlen_kw: float = Field(ge=0, description="je Innengerät")
    scop: float = Field(gt=1, le=10)
    seer: float = Field(gt=1, le=15)
    cop_pruefpunkte: dict[float, float] | None = None
    eer_pruefpunkte: dict[float, float] | None = None

    @field_validator("cop_pruefpunkte")
    @classmethod
    def _cop(cls, v: dict[float, float] | None) -> dict[float, float] | None:
        return _pruefe_kennlinie(v, "COP-Prüfpunkte")

    @field_validator("eer_pruefpunkte")
    @classmethod
    def _eer(cls, v: dict[float, float] | None) -> dict[float, float] | None:
        return _pruefe_kennlinie(v, "EER-Prüfpunkte")

    @property
    def gesamtleistung_heizen_kw(self) -> float:
        return self.anzahl_innengeraete * self.nennleistung_heizen_kw

    @property
    def gesamtleistung_kuehlen_kw(self) -> float:
        return self.anzahl_innengeraete * self.nennleistung_kuehlen_kw


class Nutzung(_Basis):
    heizstrategie: Heizstrategie
    split_schwelltemperatur_c: float = Field(ge=-20, le=20)
    kuehlen_aktiv: bool


class PV(_Basis):
    status: Ausbaustatus
    leistung_kwp: float = Field(ge=0, le=100)
    azimut_grad: float = Field(ge=-180, le=180, description="0 = Süd, -90 = Ost, 90 = West")
    neigung_grad: float = Field(ge=0, le=90)
    systemverluste_prozent: float = Field(ge=0, le=50)

    @property
    def aktiv(self) -> bool:
        return self.status != Ausbaustatus.KEINE and self.leistung_kwp > 0


class Speicher(_Basis):
    status: Ausbaustatus
    kapazitaet_kwh: float = Field(ge=0, le=200)
    ladeleistung_kw: float = Field(ge=0)
    entladeleistung_kw: float = Field(ge=0)
    wirkungsgrad: float = Field(gt=0, le=1, description="je Richtung (Laden bzw. Entladen)")
    soc_min: float = Field(ge=0, le=1)
    soc_max: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _soc(self) -> Self:
        if self.soc_min >= self.soc_max:
            raise ValueError("soc_min muss kleiner als soc_max sein")
        return self

    @property
    def aktiv(self) -> bool:
        return self.status != Ausbaustatus.KEINE and self.kapazitaet_kwh > 0


class Haushalt(_Basis):
    jahresverbrauch_kwh: float = Field(ge=0, le=50_000, description="ohne Klimaanlage")


class Bestand(_Basis):
    heiztraeger: Heiztraeger
    kesselwirkungsgrad: float = Field(gt=0, le=1.1)


class Preise(_Basis):
    strom_bezug_eur_kwh: float = Field(ge=0, le=2)
    brennstoff_eur_kwh: float = Field(ge=0, le=1)
    einspeiseverguetung_eur_kwh: float = Field(ge=0, le=1)
    szenario: str


class Kosten(_Basis):
    investition_split_eur: float = Field(ge=0)
    investition_pv_eur: float = Field(ge=0)
    investition_speicher_eur: float = Field(ge=0)
    wartung_eur_jahr: float = Field(ge=0)

    @property
    def investition_gesamt_eur(self) -> float:
        return self.investition_split_eur + self.investition_pv_eur + self.investition_speicher_eur


class Rahmen(_Basis):
    betrachtungszeitraum_jahre: int = Field(ge=1, le=40)
    kalkulationszins: float = Field(ge=-0.05, le=0.2)


class MonteCarlo(_Basis):
    pfade: int = Field(ge=1, le=100_000)
    seed: int | None = None
    parametersaetze: int = Field(default=100, ge=1, le=2000)


class Eingaben(_Basis):
    standort: Standort
    gebaeude: Gebaeude
    geraete: Geraete
    nutzung: Nutzung
    pv: PV
    speicher: Speicher
    haushalt: Haushalt
    bestand: Bestand
    preise: Preise
    kosten: Kosten
    rahmen: Rahmen
    montecarlo: MonteCarlo

    @model_validator(mode="after")
    def _investitionen_passen_zum_status(self) -> Self:
        if self.kosten.investition_pv_eur > 0 and self.pv.status != Ausbaustatus.NEU:
            raise ValueError("Investition PV nur angeben, wenn die PV-Anlage neu ist")
        if self.kosten.investition_speicher_eur > 0 and self.speicher.status != Ausbaustatus.NEU:
            raise ValueError("Investition Speicher nur angeben, wenn der Speicher neu ist")
        return self


# --- Annahmen ---------------------------------------------------------------


class Kuehlannahmen(_Basis):
    kuehlgrenztemperatur_c: float
    solarfaktor_m2: float = Field(ge=0)
    interne_lasten_w: float = Field(ge=0)


class Carnotannahmen(_Basis):
    kondensation_heizen_c: float
    verdampfer_abstand_k: float = Field(ge=0)
    verdampfung_kuehlen_c: float
    kondensator_abstand_k: float = Field(ge=0)


class Referenzklima(_Basis):
    """Stunden je Temperaturklasse zur Kalibrierung auf SCOP bzw. SEER."""

    auslegungstemperatur_c: float
    stunden: dict[float, float]
    heizgrenze_c: float | None = None
    kuehlgrenze_c: float | None = None


class Waermepumpenannahmen(_Basis):
    pruefpunkte_heizen_c: list[float]
    cop_min: float = Field(gt=0)
    cop_max: float = Field(gt=0)
    abtau_band_c: tuple[float, float]
    abtau_abschlag: float = Field(ge=0, lt=1)
    teillast_abschlag: float = Field(ge=0, lt=1)
    praxisabschlag: float = Field(ge=0, lt=1)
    leistungsfaktor_heizen: dict[float, float]
    carnot: Carnotannahmen
    referenzklima_heizen: Referenzklima
    referenzklima_kuehlen: Referenzklima

    @field_validator("leistungsfaktor_heizen")
    @classmethod
    def _leistung(cls, v: dict[float, float]) -> dict[float, float]:
        return _pruefe_kennlinie(v, "Leistungsfaktor Heizen")

    @model_validator(mode="after")
    def _grenzen(self) -> Self:
        if self.cop_min >= self.cop_max:
            raise ValueError("cop_min muss kleiner als cop_max sein")
        if self.abtau_band_c[0] >= self.abtau_band_c[1]:
            raise ValueError("abtau_band_c muss aufsteigend sein")
        return self


class Haushaltsannahmen(_Basis):
    lastprofil: str
    rauschen_sigma: float = Field(ge=0, le=1)


class PVGISAnnahmen(_Basis):
    api_url: str
    startjahr: int = Field(ge=2005)
    endjahr: int
    zeitversatz_h: int = Field(ge=-12, le=12)

    @model_validator(mode="after")
    def _jahre(self) -> Self:
        if self.startjahr > self.endjahr:
            raise ValueError("PVGIS: startjahr darf nicht nach endjahr liegen")
        return self


class Brennstoffannahmen(_Basis):
    co2_kg_kwh: dict[Heiztraeger, float]


class Preisszenario(_Basis):
    strom_drift: float
    strom_volatilitaet: float = Field(ge=0)
    brennstoff_drift: float
    brennstoff_volatilitaet: float = Field(ge=0)
    korrelation: float = Field(ge=-1, le=1)
    co2_aufschlag_eur_t: float = Field(ge=0)


class Plausibilitaet(_Basis):
    jaz_heizen: tuple[float, float]

    @model_validator(mode="after")
    def _bereich(self) -> Self:
        if self.jaz_heizen[0] >= self.jaz_heizen[1]:
            raise ValueError("Plausibilitätsbereich JAZ muss aufsteigend sein")
        return self


class Verhaltensannahmen(_Basis):
    sigma_t_soll_k: float = Field(ge=0)
    sigma_schwelle_k: float = Field(ge=0)


class Technikannahmen(_Basis):
    cop_abschlag_min: float = Field(ge=0, lt=1)
    cop_abschlag_max: float = Field(ge=0, lt=1)

    @model_validator(mode="after")
    def _bereich(self) -> Self:
        if self.cop_abschlag_min > self.cop_abschlag_max:
            raise ValueError("cop_abschlag_min darf nicht größer als cop_abschlag_max sein")
        return self


class Annahmen(_Basis):
    spezifische_heizlast_w_m2: dict[Baualtersklasse, float]
    norm_aussentemperatur_c: float
    kuehlen: Kuehlannahmen
    waermepumpe: Waermepumpenannahmen
    haushalt: Haushaltsannahmen
    pvgis: PVGISAnnahmen
    brennstoffe: Brennstoffannahmen
    preisszenarien: dict[str, Preisszenario]
    plausibilitaet: Plausibilitaet
    verhalten: Verhaltensannahmen
    technik: Technikannahmen

    @field_validator("spezifische_heizlast_w_m2")
    @classmethod
    def _alle_klassen(cls, v: dict[Baualtersklasse, float]) -> dict[Baualtersklasse, float]:
        fehlend = set(Baualtersklasse) - set(v)
        if fehlend:
            raise ValueError(f"Heizlast fehlt für: {', '.join(sorted(fehlend))}")
        return v

    @field_validator("brennstoffe")
    @classmethod
    def _alle_traeger(cls, v: Brennstoffannahmen) -> Brennstoffannahmen:
        fehlend = set(Heiztraeger) - set(v.co2_kg_kwh)
        if fehlend:
            raise ValueError(f"CO2-Faktor fehlt für: {', '.join(sorted(fehlend))}")
        return v


# --- Gesamtkonfiguration ----------------------------------------------------


class Meta(_Basis):
    platzhalter: bool
    stand: str


class Konfiguration(_Basis):
    meta: Meta
    eingaben: Eingaben
    annahmen: Annahmen

    @model_validator(mode="after")
    def _szenario_bekannt(self) -> Self:
        szenario = self.eingaben.preise.szenario
        if szenario not in self.annahmen.preisszenarien:
            bekannt = ", ".join(sorted(self.annahmen.preisszenarien))
            raise ValueError(f"Unbekanntes Preisszenario '{szenario}' (bekannt: {bekannt})")
        return self

    @property
    def preisszenario(self) -> Preisszenario:
        return self.annahmen.preisszenarien[self.eingaben.preise.szenario]

    @property
    def spezifische_heizlast_w_m2(self) -> float:
        return self.annahmen.spezifische_heizlast_w_m2[self.eingaben.gebaeude.baualtersklasse]


def _ist_tabelle(d: dict[Any, Any]) -> bool:
    """Kennlinien (Zahl -> Wert) werden als Ganzes ersetzt, nicht gemischt."""
    return bool(d) and all(isinstance(k, int | float) for k in d)


def _zusammenfuehren(basis: dict[str, Any], neu: dict[str, Any]) -> dict[str, Any]:
    ergebnis = dict(basis)
    for schluessel, wert in neu.items():
        alt = ergebnis.get(schluessel)
        if (
            isinstance(alt, dict)
            and isinstance(wert, dict)
            and not _ist_tabelle(alt)
            and not _ist_tabelle(wert)
        ):
            ergebnis[schluessel] = _zusammenfuehren(alt, wert)
        else:
            ergebnis[schluessel] = wert
    return ergebnis


def lade_defaults(pfad: Path = DEFAULTS_PFAD) -> dict[str, Any]:
    """Liest die Defaults-Datei als verschachteltes Dictionary."""
    with pfad.open(encoding="utf-8") as datei:
        return yaml.safe_load(datei)


def lade_konfiguration(
    ueberschreibungen: dict[str, Any] | None = None, pfad: Path = DEFAULTS_PFAD
) -> Konfiguration:
    """Lädt die Defaults und legt Überschreibungen darüber.

    Überschreibungen haben dieselbe Struktur wie die YAML-Datei, z. B.
    ``{"eingaben": {"gebaeude": {"split_flaeche_m2": 40}}}``. Verschachtelte
    Gruppen werden zusammengeführt, Kennlinien (z. B. COP-Prüfpunkte) als
    Ganzes ersetzt.
    """
    daten = lade_defaults(pfad)
    if ueberschreibungen:
        daten = _zusammenfuehren(daten, ueberschreibungen)
    return Konfiguration.model_validate(daten)
