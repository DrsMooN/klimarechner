"""PDF-Bericht für den Endkunden.

Inhalt: Kernaussagen, wichtigste Eingaben, Kennzahlen (P10/P50/P90), zwei
Diagramme, Annahmen und Haftungshinweis. Gesetzt mit ReportLab; die
Diagramme werden mit Matplotlib statisch gezeichnet, damit kein Browser für
den Export nötig ist. Benötigt das Extra ``app``.
"""

from __future__ import annotations

import io
from datetime import date

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from engine import auswertung  # noqa: E402
from engine.inputs import Konfiguration  # noqa: E402
from engine.montecarlo import MonteCarloErgebnis  # noqa: E402
from engine.simulation import Jahresergebnis  # noqa: E402

BLAU, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, TEXT_2, LINIE = "#0b0b0b", "#52514e", "#d9d8d4"

HAFTUNG = (
    "Diese Berechnung ist eine Schätzung auf Basis vereinfachter Modelle. Sie ist keine "
    "normgerechte Heiz- oder Kühllastberechnung (DIN EN 12831, VDI 2078) und keine "
    "Garantie für tatsächliche Verbräuche, Kosten oder Ersparnisse. Wetter, Energiepreise "
    "und Nutzerverhalten können erheblich von den Annahmen abweichen."
)

BAUALTER = {
    "vor_1949": "vor 1949",
    "1949_1978": "1949–1978",
    "1979_1994": "1979–1994",
    "1995_2009": "1995–2009",
    "ab_2010": "ab 2010",
    "effizienzhaus": "Effizienzhaus",
}
STRATEGIE = {
    "uebergangszeit": "Übergangszeit (Split ab {schwelle} °C)",
    "bevorzugt_split": "bevorzugt Split",
    "vollheizung_teilraeume": "Vollheizung der Split-Räume",
}
AUSBAU = {"keine": "keine", "vorhanden": "vorhanden", "neu": "neu"}


# --- Diagramme --------------------------------------------------------------


def _euro_achse(wert: float, _pos) -> str:
    return auswertung.zahl(wert)


def _stil(ax: plt.Axes, titel: str, y_titel: str) -> None:
    ax.set_title(titel, loc="left", fontsize=10, color=TEXT, fontweight="bold", pad=14)
    ax.set_ylabel(y_titel, fontsize=8, color=TEXT_2)
    ax.tick_params(labelsize=8, colors=TEXT_2, length=0)
    ax.grid(axis="y", color=LINIE, linewidth=0.6)
    ax.set_axisbelow(True)
    for seite in ("top", "right", "left"):
        ax.spines[seite].set_visible(False)
    ax.spines["bottom"].set_color(LINIE)


def _legende(ax: plt.Axes) -> None:
    """Legende rechts über der Zeichenfläche, damit sie keine Daten verdeckt."""
    ax.legend(
        fontsize=8,
        frameon=False,
        ncols=3,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        borderaxespad=0.2,
        labelcolor=TEXT_2,
    )


def _png(fig: plt.Figure) -> io.BytesIO:
    puffer = io.BytesIO()
    fig.savefig(puffer, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    puffer.seek(0)
    return puffer


def diagramm_cashflow(ergebnis: MonteCarloErgebnis) -> io.BytesIO:
    kum = ergebnis.kumulierter_cashflow
    jahre = np.arange(kum.shape[1])
    p10, p50, p90 = (np.percentile(kum, q, axis=0) for q in (10, 50, 90))
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.fill_between(
        jahre, p10, p90, color=BLAU, alpha=0.18, linewidth=0, label="Bandbreite (80 % der Fälle)"
    )
    ax.plot(jahre, p50, color=BLAU, linewidth=2, label="Mittlerer Fall")
    ax.axhline(0, color=TEXT_2, linewidth=0.8)
    ax.yaxis.set_major_formatter(FuncFormatter(_euro_achse))
    ax.set_xlim(0, jahre[-1])
    ax.set_xlabel("Jahr", fontsize=8, color=TEXT_2)
    _stil(ax, "Kumulierter Cashflow", "€")
    _legende(ax)
    return _png(fig)


def diagramm_monatsbilanz(tabelle: pd.DataFrame) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    unten = np.zeros(len(tabelle))
    for quelle, farbe in zip(tabelle.columns, (BLAU, ORANGE, AQUA), strict=False):
        werte = tabelle[quelle].to_numpy()
        if werte.sum() <= 0:
            continue
        ax.bar(
            tabelle.index,
            werte,
            bottom=unten,
            color=farbe,
            width=0.7,
            edgecolor="white",
            linewidth=0.8,
            label=quelle,
        )
        unten += werte
    _stil(ax, "Strom für das Split-Gerät je Monat", "kWh")
    _legende(ax)
    return _png(fig)


# --- Bericht ----------------------------------------------------------------


def _stile() -> dict[str, ParagraphStyle]:
    basis = getSampleStyleSheet()
    return {
        "titel": ParagraphStyle(
            "titel",
            parent=basis["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            alignment=TA_LEFT,
            spaceAfter=2 * mm,
            textColor=TEXT,
        ),
        "unter": ParagraphStyle(
            "unter", parent=basis["Normal"], fontSize=9, textColor=TEXT_2, spaceAfter=6 * mm
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=basis["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            spaceBefore=5 * mm,
            spaceAfter=2 * mm,
            textColor=TEXT,
        ),
        "text": ParagraphStyle(
            "text", parent=basis["Normal"], fontSize=9.5, leading=13, textColor=TEXT
        ),
        "aussage": ParagraphStyle(
            "aussage",
            parent=basis["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=14,
            leftIndent=4 * mm,
            bulletIndent=0,
            spaceAfter=1.5 * mm,
            textColor=TEXT,
        ),
        "klein": ParagraphStyle(
            "klein", parent=basis["Normal"], fontSize=8, leading=10.5, textColor=TEXT_2
        ),
    }


def _tabelle(zeilen: list[list[str]], spalten_mm: list[float], kopf: bool) -> Table:
    tabelle = Table(zeilen, colWidths=[b * mm for b in spalten_mm], hAlign="LEFT")
    stil = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(TEXT)),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor(LINIE)),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT" if kopf else "LEFT"),
    ]
    if kopf:
        stil += [
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(TEXT_2)),
        ]
    tabelle.setStyle(TableStyle(stil))
    return tabelle


def eingabezeilen(konfig: Konfiguration) -> list[list[str]]:
    e = konfig.eingaben
    z = auswertung.zahl
    g, r, n = e.gebaeude, e.geraete, e.nutzung
    strategie = STRATEGIE[n.heizstrategie].format(schwelle=z(n.split_schwelltemperatur_c, 1))
    pv = "keine"
    if e.pv.status != "keine":
        pv = f"{z(e.pv.leistung_kwp, 1)} kWp ({AUSBAU[e.pv.status]})"
    speicher = "keiner"
    if e.speicher.status != "keine":
        speicher = f"{z(e.speicher.kapazitaet_kwh, 1)} kWh ({AUSBAU[e.speicher.status]})"
    return [
        [
            "Gebäude",
            f"{z(g.beheizte_flaeche_m2)} m² beheizt, davon {z(g.split_flaeche_m2)} m² "
            f"mit Split-Gerät, Baujahr {BAUALTER[g.baualtersklasse]}",
        ],
        [
            "Split-Geräte",
            f"{r.anzahl_innengeraete} × {z(r.nennleistung_heizen_kw, 1)} kW "
            f"Heizen, SCOP {z(r.scop, 1)}, SEER {z(r.seer, 1)}",
        ],
        ["Betrieb", f"{strategie}, Kühlen {'aktiv' if n.kuehlen_aktiv else 'aus'}"],
        [
            "Bestandsheizung",
            f"{'Gas' if e.bestand.heiztraeger == 'gas' else 'Öl'}, "
            f"Wirkungsgrad {z(e.bestand.kesselwirkungsgrad * 100)} %",
        ],
        ["PV / Speicher", f"{pv} / {speicher}"],
        ["Haushaltsstrom", f"{z(e.haushalt.jahresverbrauch_kwh)} kWh pro Jahr"],
        [
            "Preise heute",
            f"Strom {z(e.preise.strom_bezug_eur_kwh, 2)} €/kWh, Brennstoff "
            f"{z(e.preise.brennstoff_eur_kwh, 2)} €/kWh, Einspeisung "
            f"{z(e.preise.einspeiseverguetung_eur_kwh, 3)} €/kWh",
        ],
        [
            "Investition",
            f"{auswertung.euro(e.kosten.investition_gesamt_eur)}, Wartung "
            f"{auswertung.euro(e.kosten.wartung_eur_jahr)} pro Jahr",
        ],
    ]


def annahmen_text(konfig: Konfiguration, ergebnis: MonteCarloErgebnis, wetterjahre: int) -> str:
    e, a = konfig.eingaben, konfig.annahmen
    z = auswertung.zahl
    s = konfig.preisszenario
    return (
        f"Betrachtungszeitraum {e.rahmen.betrachtungszeitraum_jahre} Jahre, Kalkulationszins "
        f"{z(e.rahmen.kalkulationszins * 100, 1)} %. Preisentwicklung „{e.preise.szenario}“: "
        f"Strom im Mittel {z(s.strom_drift * 100, 1)} % pro Jahr, Brennstoff "
        f"{z(s.brennstoff_drift * 100, 1)} % pro Jahr, mit Schwankungen. Stündliche "
        f"Simulation über {wetterjahre} Wetterjahre, {z(ergebnis.pfade)} berechnete Fälle "
        f"mit unterschiedlichem Wetter, Preisen und Nutzerverhalten. Gerätewerte mit "
        f"{z(a.waermepumpe.praxisabschlag * 100)} % Abschlag gegenüber dem Datenblatt "
        f"(Praxisbetrieb). Kühlen ist als zusätzliche Stromkosten berücksichtigt, nicht als "
        f"Ersparnis. Angegebene Bandbreiten umfassen 80 % der berechneten Fälle."
    )


def erstelle_bericht(
    konfig: Konfiguration,
    ergebnis: MonteCarloErgebnis,
    detail: Jahresergebnis,
    wetterjahre: int,
    erstellt_am: date | None = None,
) -> bytes:
    """PDF-Bericht als Bytes (A4)."""
    st = _stile()
    erstellt_am = erstellt_am or date.today()
    puffer = io.BytesIO()
    doc = SimpleDocTemplate(
        puffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="Wirtschaftlichkeit Split-Klimaanlage",
        author="Klimarechner",
    )
    breite = A4[0] - 36 * mm

    inhalt = [
        Paragraph("Wirtschaftlichkeit Ihrer Split-Klimaanlage", st["titel"]),
        Paragraph(f"Berechnung vom {erstellt_am:%d.%m.%Y}", st["unter"]),
        Paragraph("Das Wichtigste", st["h2"]),
    ]
    inhalt += [
        Paragraph(aussage, st["aussage"], bulletText="•")
        for aussage in auswertung.kernaussagen(ergebnis, konfig)
    ]

    inhalt += [
        Paragraph("Ihre Angaben", st["h2"]),
        _tabelle(eingabezeilen(konfig), [32, breite / mm - 32], kopf=False),
    ]

    kennzahlen = auswertung.kennzahlentabelle(ergebnis)
    zeilen = [["", "ungünstig (P10)", "mittel (P50)", "günstig (P90)"]]
    zeilen += [
        [name, *werte]
        for name, werte in zip(kennzahlen.index, kennzahlen.to_numpy().tolist(), strict=True)
    ]
    inhalt += [
        KeepTogether(
            [
                Paragraph("Ergebnisse im Überblick", st["h2"]),
                _tabelle(zeilen, [70, 34, 34, 34], kopf=True),
                Spacer(1, 1.5 * mm),
                Paragraph(
                    "P10/P50/P90: In 10 % der berechneten Fälle ist das Ergebnis schlechter als "
                    "„ungünstig“, in 10 % besser als „günstig“.",
                    st["klein"],
                ),
            ]
        )
    ]

    bild_breite = breite
    cashflow = diagramm_cashflow(ergebnis)
    monat = diagramm_monatsbilanz(auswertung.monatsbilanz(detail))
    inhalt += [
        KeepTogether(
            [
                Paragraph("Verlauf und Energiebilanz", st["h2"]),
                Image(cashflow, width=bild_breite, height=bild_breite * 2.6 / 6.4 * 1.08),
                Spacer(1, 3 * mm),
                Image(monat, width=bild_breite, height=bild_breite * 2.4 / 6.4 * 1.08),
                Paragraph(
                    f"Monatsbilanz für ein Jahr mit mittlerem Heizbedarf (Wetterjahr "
                    f"{detail.wetterjahr}).",
                    st["klein"],
                ),
            ]
        )
    ]

    inhalt += [
        KeepTogether(
            [
                Paragraph("Annahmen", st["h2"]),
                Paragraph(annahmen_text(konfig, ergebnis, wetterjahre), st["text"]),
            ]
        ),
        KeepTogether(
            [
                Paragraph("Hinweis", st["h2"]),
                Paragraph(HAFTUNG, st["text"]),
            ]
        ),
    ]

    def fusszeile(canvas, dokument) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor(TEXT_2))
        canvas.drawString(18 * mm, 9 * mm, "Schätzung, keine Garantie.")
        canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"Seite {dokument.page}")
        canvas.restoreState()

    doc.build(inhalt, onFirstPage=fusszeile, onLaterPages=fusszeile)
    return puffer.getvalue()
