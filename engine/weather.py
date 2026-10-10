"""Wetter- und PV-Zeitreihen.

Quelle sind stündliche Zeitreihen von PVGIS (EU JRC, ``seriescalc``):
Außentemperatur, Einstrahlung auf die Modulebene und PV-Ertrag. PVGIS wird
immer mit 1 kWp abgefragt, damit der Cache unabhängig von der Anlagengröße
ist; skaliert wird in :meth:`Wetterjahr.pv_kw`.

Für Tests und Offline-Betrieb gibt es :func:`synthetisches_wetterjahr`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

from engine.inputs import PV, Konfiguration, PVGISAnnahmen

STUNDEN_PRO_JAHR = 8760
CACHE_VERZEICHNIS = Path(__file__).resolve().parent.parent / "data" / "cache" / "pvgis"


@dataclass(frozen=True)
class Wetterjahr:
    """Ein Kalenderjahr mit 8.760 Stundenwerten (Ortszeit ohne Sommerzeit)."""

    jahr: int
    t_aussen_c: np.ndarray
    """Außentemperatur in 2 m Höhe [°C]"""
    einstrahlung_w_m2: np.ndarray
    """Globalstrahlung auf die Modulebene [W/m²]"""
    pv_kw_pro_kwp: np.ndarray
    """PV-Ertrag (AC) je kWp [kW]"""

    def __post_init__(self) -> None:
        for name in ("t_aussen_c", "einstrahlung_w_m2", "pv_kw_pro_kwp"):
            werte = getattr(self, name)
            if werte.shape != (STUNDEN_PRO_JAHR,):
                raise ValueError(
                    f"{name}: erwartet {STUNDEN_PRO_JAHR} Werte, erhalten {werte.shape}"
                )

    def pv_kw(self, pv: PV) -> np.ndarray:
        """PV-Erzeugung der Anlage [kW]; Nullen, wenn keine PV aktiv ist."""
        if not pv.aktiv:
            return np.zeros(STUNDEN_PRO_JAHR)
        return self.pv_kw_pro_kwp * pv.leistung_kwp


# --- PVGIS ------------------------------------------------------------------


def _ohne_schalttag(zeiten: list[str], werte: np.ndarray) -> np.ndarray:
    maske = np.array([not z.startswith(z[:4] + "0229") for z in zeiten])
    return werte[maske]


def parse_pvgis(antwort: dict, zeitversatz_h: int) -> list[Wetterjahr]:
    """Zerlegt eine ``seriescalc``-Antwort (JSON) in Wetterjahre.

    Schalttage werden entfernt, die UTC-Zeitreihe wird um ``zeitversatz_h``
    Stunden in Ortszeit verschoben (innerhalb des Jahres rotiert).
    """
    stunden = antwort["outputs"]["hourly"]
    nach_jahr: dict[int, list[dict]] = {}
    for eintrag in stunden:
        nach_jahr.setdefault(int(eintrag["time"][:4]), []).append(eintrag)

    jahre = []
    for jahr, eintraege in sorted(nach_jahr.items()):
        zeiten = [e["time"] for e in eintraege]

        def spalte(schluessel: str, eintraege=eintraege, zeiten=zeiten) -> np.ndarray:
            werte = np.array([float(e[schluessel]) for e in eintraege])
            return np.roll(_ohne_schalttag(zeiten, werte), zeitversatz_h)

        jahre.append(
            Wetterjahr(
                jahr=jahr,
                t_aussen_c=spalte("T2m"),
                einstrahlung_w_m2=spalte("G(i)"),
                pv_kw_pro_kwp=spalte("P") / 1000.0,
            )
        )
    return jahre


@dataclass(frozen=True)
class PVGISAnfrage:
    breitengrad: float
    laengengrad: float
    neigung_grad: float
    azimut_grad: float
    systemverluste_prozent: float
    startjahr: int
    endjahr: int

    @classmethod
    def aus_konfiguration(cls, konfig: Konfiguration) -> PVGISAnfrage:
        e = konfig.eingaben
        return cls(
            breitengrad=round(e.standort.breitengrad, 3),
            laengengrad=round(e.standort.laengengrad, 3),
            neigung_grad=e.pv.neigung_grad,
            azimut_grad=e.pv.azimut_grad,
            systemverluste_prozent=e.pv.systemverluste_prozent,
            startjahr=konfig.annahmen.pvgis.startjahr,
            endjahr=konfig.annahmen.pvgis.endjahr,
        )

    def parameter(self) -> dict[str, str | float | int]:
        return {
            "lat": self.breitengrad,
            "lon": self.laengengrad,
            "startyear": self.startjahr,
            "endyear": self.endjahr,
            "pvcalculation": 1,
            "peakpower": 1,
            "loss": self.systemverluste_prozent,
            "angle": self.neigung_grad,
            "aspect": self.azimut_grad,
            "outputformat": "json",
        }

    def cache_schluessel(self) -> str:
        text = json.dumps(self.parameter(), sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()[:16]


class PVGISClient:
    """Ruft PVGIS ab und speichert die Wetterjahre als ``.npz`` im Cache."""

    def __init__(
        self,
        annahmen: PVGISAnnahmen,
        cache_verzeichnis: Path = CACHE_VERZEICHNIS,
        http: httpx.Client | None = None,
    ) -> None:
        self.annahmen = annahmen
        self.cache_verzeichnis = cache_verzeichnis
        self._http = http

    def _cache_pfad(self, anfrage: PVGISAnfrage) -> Path:
        return self.cache_verzeichnis / f"{anfrage.cache_schluessel()}.npz"

    def _abrufen(self, anfrage: PVGISAnfrage) -> dict:
        http = self._http or httpx.Client(timeout=120)
        try:
            antwort = http.get(self.annahmen.api_url, params=anfrage.parameter())
            antwort.raise_for_status()
            return antwort.json()
        finally:
            if self._http is None:
                http.close()

    def wetterjahre(self, anfrage: PVGISAnfrage) -> list[Wetterjahr]:
        pfad = self._cache_pfad(anfrage)
        if pfad.exists():
            return _lade_cache(pfad)
        jahre = parse_pvgis(self._abrufen(anfrage), self.annahmen.zeitversatz_h)
        _speichere_cache(pfad, jahre)
        return jahre


def _speichere_cache(pfad: Path, jahre: list[Wetterjahr]) -> None:
    pfad.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pfad,
        jahre=np.array([j.jahr for j in jahre]),
        t_aussen_c=np.stack([j.t_aussen_c for j in jahre]),
        einstrahlung_w_m2=np.stack([j.einstrahlung_w_m2 for j in jahre]),
        pv_kw_pro_kwp=np.stack([j.pv_kw_pro_kwp for j in jahre]),
    )


def _lade_cache(pfad: Path) -> list[Wetterjahr]:
    with np.load(pfad) as daten:
        return [
            Wetterjahr(
                jahr=int(jahr),
                t_aussen_c=daten["t_aussen_c"][i],
                einstrahlung_w_m2=daten["einstrahlung_w_m2"][i],
                pv_kw_pro_kwp=daten["pv_kw_pro_kwp"][i],
            )
            for i, jahr in enumerate(daten["jahre"])
        ]


def lade_wetterjahre(konfig: Konfiguration, client: PVGISClient | None = None) -> list[Wetterjahr]:
    """Alle verfügbaren Wetterjahre für Standort und PV-Ausrichtung der Konfiguration."""
    client = client or PVGISClient(konfig.annahmen.pvgis)
    return client.wetterjahre(PVGISAnfrage.aus_konfiguration(konfig))


# --- Synthetische Daten -----------------------------------------------------


def synthetisches_wetterjahr(
    jahr: int = 2000, breitengrad: float = 51.0, seed: int = 0
) -> Wetterjahr:
    """Grob mitteleuropäisches Wetterjahr für Tests und Offline-Betrieb.

    Keine Messdaten: Jahres- und Tagesgang der Temperatur mit Rauschen,
    Einstrahlung aus Sonnenhöhe mal zufälliger Tagesbewölkung.
    """
    rng = np.random.default_rng(seed)
    stunde = np.arange(STUNDEN_PRO_JAHR)
    tag = stunde // 24
    tagesstunde = stunde % 24

    # Temperatur: Jahresgang (Min. Mitte Januar), Tagesgang (Max. 15 Uhr),
    # träges Rauschen auf Tagesbasis
    rauschen_tag = np.zeros(365)
    for d in range(1, 365):
        rauschen_tag[d] = 0.8 * rauschen_tag[d - 1] + rng.normal(0, 1.8)
    t_aussen = (
        9.5
        - 9.0 * np.cos(2 * np.pi * (tag - 15) / 365)
        + 4.0 * np.cos(2 * np.pi * (tagesstunde - 15) / 24)
        + rauschen_tag[tag]
    )

    # Einstrahlung: Sonnenhöhe (Mittagszeit 12 Uhr) mal Klarheitsindex
    deklination = np.radians(23.44) * np.sin(2 * np.pi * (tag - 80) / 365)
    stundenwinkel = np.radians(15.0 * (tagesstunde + 0.5 - 12))
    phi = np.radians(breitengrad)
    sin_hoehe = np.sin(phi) * np.sin(deklination) + np.cos(phi) * np.cos(deklination) * np.cos(
        stundenwinkel
    )
    klarheit = rng.uniform(0.25, 0.85, size=365)[tag]
    einstrahlung = np.clip(1100.0 * sin_hoehe, 0, None) * klarheit * 1.15

    pv = einstrahlung / 1000.0 * 0.86
    return Wetterjahr(
        jahr=jahr, t_aussen_c=t_aussen, einstrahlung_w_m2=einstrahlung, pv_kw_pro_kwp=pv
    )
