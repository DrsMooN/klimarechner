from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def _app():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    return at


def _rechnen(at):
    at.radio(key="wetterquelle").set_value("Synthetisches Testwetter (offline, nur Demo)")
    at.button[0].click().run()
    return at


def test_app_startet_mit_defaults():
    at = _app()
    assert at.title[0].value == "Wirtschaftlichkeit Split-Klimaanlage"
    assert len(at.button) == 1


def test_app_rechnet_mit_synthetischem_wetter():
    at = _rechnen(_app())
    assert not at.exception, at.exception
    assert not at.error
    assert at.header[0].value == "Ergebnis"
    texte = " ".join(m.value for m in at.markdown)
    assert "Ersparnis im ersten Jahr" in texte
    assert len(at.metric) == 4
    knopf = at.get("download_button")
    assert len(knopf) == 1
    assert knopf[0].proto.label == "PDF-Bericht herunterladen"


def test_app_zeigt_eingabefehler():
    at = _app()
    flaeche = next(n for n in at.number_input if n.label.startswith("Fläche der Räume"))
    flaeche.set_value(500.0)
    at.button[0].click().run()
    assert not at.exception
    assert "Split-Fläche" in at.error[0].value
