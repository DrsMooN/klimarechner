import os
from dataclasses import replace
from datetime import datetime, timedelta

import httpx
import numpy as np
import pytest

from engine.inputs import PV, lade_konfiguration
from engine.weather import (
    STUNDEN_PRO_JAHR,
    PVGISAnfrage,
    PVGISClient,
    PVGISFehler,
    Wetterjahr,
    lade_wetterjahre,
    parse_pvgis,
    synthetisches_wetterjahr,
)


def _pvgis_antwort(jahre):
    """Antwort im Format von PVGIS seriescalc; T2m kodiert die Stunde des Jahres."""
    stunden = []
    for jahr in jahre:
        zeit = datetime(jahr, 1, 1, 0, 10)
        i = 0
        while zeit.year == jahr:
            stunden.append(
                {
                    "time": zeit.strftime("%Y%m%d:%H%M"),
                    "P": 500.0,
                    "G(i)": 600.0,
                    "H_sun": 10.0,
                    "T2m": float(i),
                    "WS10m": 1.0,
                    "Int": 0.0,
                }
            )
            zeit += timedelta(hours=1)
            i += 1
    return {"inputs": {}, "outputs": {"hourly": stunden}, "meta": {}}


def test_parse_entfernt_schalttag_und_verschiebt_zeit():
    jahre = parse_pvgis(_pvgis_antwort([2019, 2020]), zeitversatz_h=1)
    assert [j.jahr for j in jahre] == [2019, 2020]
    for j in jahre:
        assert j.t_aussen_c.shape == (STUNDEN_PRO_JAHR,)
    # Ortszeit = UTC + 1: Stunde 1 Ortszeit ist UTC-Stunde 0
    assert jahre[0].t_aussen_c[1] == 0.0
    # 2020: nach dem 28.2. fehlen 24 Stunden (29.2.)
    index_1_maerz = (31 + 28) * 24
    assert jahre[1].t_aussen_c[index_1_maerz + 1] == index_1_maerz + 24
    assert jahre[0].pv_kw_pro_kwp[0] == pytest.approx(0.5)


def test_client_nutzt_cache(tmp_path):
    aufrufe = []

    def handler(anfrage: httpx.Request) -> httpx.Response:
        aufrufe.append(anfrage)
        return httpx.Response(200, json=_pvgis_antwort([2021]))

    konfig = lade_konfiguration()
    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = PVGISClient(konfig.annahmen.pvgis, cache_verzeichnis=tmp_path, http=http)

    erste = lade_wetterjahre(konfig, client)
    zweite = lade_wetterjahre(konfig, client)

    assert len(aufrufe) == 1
    assert aufrufe[0].url.params["peakpower"] == "1"
    assert aufrufe[0].url.params["aspect"] == "0.0"
    assert len(list(tmp_path.glob("*.npz"))) == 1
    np.testing.assert_array_equal(erste[0].t_aussen_c, zweite[0].t_aussen_c)


def test_client_meldet_pvgis_fehlertext(tmp_path):
    def handler(anfrage: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, json={"message": "Location over the sea. Please, select another location"}
        )

    konfig = lade_konfiguration()
    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = PVGISClient(konfig.annahmen.pvgis, cache_verzeichnis=tmp_path, http=http)

    with pytest.raises(PVGISFehler, match="HTTP 400: Location over the sea"):
        lade_wetterjahre(konfig, client)
    assert not list(tmp_path.glob("*.npz"))


@pytest.mark.skipif(not os.environ.get("PVGIS_LIVE"), reason="nur mit PVGIS_LIVE=1 (Netzzugriff)")
def test_pvgis_live(tmp_path):
    """Echter Abruf für Mitte Deutschland, 1 kWp Süd 30°, ein Jahr."""
    konfig = lade_konfiguration()
    anfrage = replace(PVGISAnfrage.aus_konfiguration(konfig), startjahr=2023, endjahr=2023)
    client = PVGISClient(konfig.annahmen.pvgis, cache_verzeichnis=tmp_path)

    (jahr,) = client.wetterjahre(anfrage)

    assert jahr.jahr == 2023
    assert 900 < jahr.pv_kw_pro_kwp.sum() < 1250
    assert 7 < jahr.t_aussen_c.mean() < 12
    # Ortszeit: PV-Maximum um die Mittagsstunde, Temperaturmaximum am Nachmittag
    tagesgang_pv = jahr.pv_kw_pro_kwp.reshape(365, 24).mean(axis=0)
    tagesgang_t = jahr.t_aussen_c.reshape(365, 24).mean(axis=0)
    assert 11 <= int(np.argmax(tagesgang_pv)) <= 13
    assert 14 <= int(np.argmax(tagesgang_t)) <= 16


def test_cache_schluessel_haengt_von_ausrichtung_ab():
    konfig = lade_konfiguration()
    sued = PVGISAnfrage.aus_konfiguration(konfig)
    west = PVGISAnfrage.aus_konfiguration(
        lade_konfiguration({"eingaben": {"pv": {"azimut_grad": 90}}})
    )
    assert sued.cache_schluessel() != west.cache_schluessel()


def test_pv_skaliert_mit_kwp_und_status():
    wetter = synthetisches_wetterjahr()
    konfig = lade_konfiguration()
    pv = konfig.eingaben.pv
    np.testing.assert_allclose(wetter.pv_kw(pv), wetter.pv_kw_pro_kwp * pv.leistung_kwp)
    keine = PV.model_validate({**pv.model_dump(), "status": "keine"})
    assert wetter.pv_kw(keine).sum() == 0


def test_synthetisches_wetter_ist_plausibel():
    wetter = synthetisches_wetterjahr(seed=3)
    assert 5 < wetter.t_aussen_c.mean() < 14
    assert wetter.einstrahlung_w_m2.min() >= 0
    # Jahresertrag je kWp grob im mitteleuropäischen Bereich
    assert 700 < wetter.pv_kw_pro_kwp.sum() < 1300


def test_wetterjahr_prueft_laenge():
    with pytest.raises(ValueError, match="8760"):
        Wetterjahr(2020, np.zeros(10), np.zeros(10), np.zeros(10))
