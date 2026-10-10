import io
from datetime import date

import pytest

pytest.importorskip("reportlab")
pytest.importorskip("matplotlib")
pypdf = pytest.importorskip("pypdf")

from engine import auswertung  # noqa: E402
from engine.inputs import lade_konfiguration  # noqa: E402
from engine.montecarlo import rechne  # noqa: E402
from engine.report import HAFTUNG, eingabezeilen, erstelle_bericht  # noqa: E402
from engine.weather import synthetisches_wetterjahr  # noqa: E402

WETTER = [synthetisches_wetterjahr(2000 + i, seed=i) for i in range(3)]


def _text(pdf: bytes) -> str:
    seiten = pypdf.PdfReader(io.BytesIO(pdf)).pages
    return " ".join(" ".join(s.extract_text().split()) for s in seiten)


@pytest.fixture(scope="module")
def bericht():
    konfig = lade_konfiguration(
        {"eingaben": {"montecarlo": {"pfade": 200, "seed": 3, "parametersaetze": 5}}}
    )
    ergebnis = rechne(konfig, WETTER)
    detail = auswertung.detailjahr(konfig, WETTER)
    pdf = erstelle_bericht(konfig, ergebnis, detail, len(WETTER), erstellt_am=date(2026, 10, 10))
    return konfig, ergebnis, pdf


def test_bericht_ist_pdf(bericht):
    _, _, pdf = bericht
    assert pdf.startswith(b"%PDF")
    seiten = len(pypdf.PdfReader(io.BytesIO(pdf)).pages)
    assert 1 <= seiten <= 3


def test_bericht_enthaelt_kernaussagen_und_hinweis(bericht):
    konfig, ergebnis, pdf = bericht
    text = _text(pdf)
    assert "Wirtschaftlichkeit Ihrer Split-Klimaanlage" in text
    assert "Berechnung vom 10.10.2026" in text
    for abschnitt in ("Das Wichtigste", "Ihre Angaben", "Ergebnisse im Überblick", "Annahmen"):
        assert abschnitt in text
    for aussage in auswertung.kernaussagen(ergebnis, konfig):
        assert aussage in text
    assert " ".join(HAFTUNG.split()) in text
    assert "Schätzung, keine Garantie." in text


def test_eingabezeilen_ohne_pv_und_speicher():
    konfig = lade_konfiguration(
        {
            "eingaben": {
                "pv": {"status": "keine"},
                "speicher": {"status": "keine"},
                "kosten": {"investition_pv_eur": 0, "investition_speicher_eur": 0},
            }
        }
    )
    zeilen = dict(eingabezeilen(konfig))
    assert zeilen["PV / Speicher"] == "keine / keiner"
