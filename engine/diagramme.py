"""Plotly-Diagramme für die Oberfläche (der PDF-Bericht zeichnet mit Matplotlib).

Benötigt das Extra ``app`` (plotly). Farben: feste Reihenfolge einer
farbfehlsichtigkeits-geprüften Kategorienpalette; Werte und Beschriftungen
bleiben in Textfarbe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engine.montecarlo import MonteCarloErgebnis

BLAU, ORANGE, AQUA, GELB = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BLAU_BAND = "rgba(42, 120, 214, 0.18)"
ORANGE_BAND = "rgba(235, 104, 52, 0.18)"
AQUA_BAND = "rgba(27, 175, 122, 0.18)"
SCHRIFT = dict(family="Inter, system-ui, sans-serif", size=13)


def _layout(fig: go.Figure, titel: str, y_titel: str, x_titel: str = "") -> go.Figure:
    fig.update_layout(
        title=dict(
            text=titel,
            x=0,
            xanchor="left",
            yref="container",
            y=1,
            yanchor="top",
            pad=dict(t=10),
            font=dict(size=15),
        ),
        font=SCHRIFT,
        # Platz für Titel und eine zweizeilige Legende zwischen Titel und Zeichenfläche
        margin=dict(l=10, r=10, t=95, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        separators=",.",
    )
    fig.update_yaxes(title_text=y_titel, zeroline=False, gridcolor="rgba(128,128,128,0.18)")
    fig.update_xaxes(title_text=x_titel, showgrid=False)
    return fig


def _faecher(kum: np.ndarray, titel: str, farbe: str, band: str) -> go.Figure:
    """Kumulierte Werte je Pfad ``[n, jahre + 1]``: P10–P90-Band und P50-Linie."""
    jahre = np.arange(kum.shape[1])
    p10, p50, p90 = (np.percentile(kum, q, axis=0) for q in (10, 50, 90))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=jahre,
            y=p90,
            mode="lines",
            line=dict(width=0),
            name="P90",
            hovertemplate="P90: %{y:,.0f} €<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=jahre,
            y=p10,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=band,
            name="Bandbreite P10–P90",
            hovertemplate="P10: %{y:,.0f} €<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=jahre,
            y=p50,
            mode="lines",
            line=dict(color=farbe, width=2),
            name="Mittlerer Fall (P50)",
            hovertemplate="P50: %{y:,.0f} €<extra></extra>",
        )
    )
    fig.add_hline(y=0, line=dict(color="rgba(128,128,128,0.7)", width=1))
    fig = _layout(fig, titel, "€", "Jahr")
    fig.update_yaxes(tickformat=",.0f")
    return fig


def faecher_ersparnis(ergebnis: MonteCarloErgebnis) -> go.Figure:
    """Kumulierte Gesamtersparnis ab null (ohne Investition)."""
    kum = ergebnis.kumuliert(ergebnis.ersparnis)
    return _faecher(kum, "Kumulierte Ersparnis gesamt", BLAU, BLAU_BAND)


def faecher_heizen(ergebnis: MonteCarloErgebnis) -> go.Figure:
    """Kumulierte Ersparnis ohne Kühlkosten (Heizen, nach Wartung)."""
    kum = ergebnis.kumuliert(ergebnis.ersparnis_heizen)
    return _faecher(kum, "Kumulierte Ersparnis Heizen (nach Wartung)", AQUA, AQUA_BAND)


def faecher_kuehlen(ergebnis: MonteCarloErgebnis) -> go.Figure:
    """Kumulierte Kühlkosten als negativer Beitrag zur Gesamtersparnis."""
    kum = ergebnis.kumuliert(-ergebnis.kuehlkosten)
    return _faecher(kum, "Kumulierte Kosten Kühlen", ORANGE, ORANGE_BAND)


def investition_ersparnis(ergebnis: MonteCarloErgebnis) -> go.Figure:
    """Investition und Gesamtersparnis im Betrachtungszeitraum (nicht abgezinst)."""
    summe = ergebnis.ersparnis.sum(axis=1)
    p10, p50, p90 = (float(np.percentile(summe, q, method="inverted_cdf")) for q in (10, 50, 90))
    jahre = ergebnis.ersparnis.shape[1]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=["Investition"],
            y=[ergebnis.investition],
            marker_color=ORANGE,
            name="Investition",
            hovertemplate="Investition: %{y:,.0f} €<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=[f"Ersparnis über {jahre} Jahre"],
            y=[p50],
            marker_color=BLAU,
            error_y=dict(
                type="data",
                symmetric=False,
                array=[p90 - p50],
                arrayminus=[p50 - p10],
                color="rgba(80,80,80,0.8)",
                thickness=1.5,
                width=10,
            ),
            name="Ersparnis (P50, Fehlerbalken P10–P90)",
            customdata=[[p10, p90]],
            hovertemplate=(
                "P50: %{y:,.0f} €<br>P10: %{customdata[0]:,.0f} €<br>"
                "P90: %{customdata[1]:,.0f} €<extra></extra>"
            ),
        )
    )
    fig = _layout(fig, "Investition und Ersparnis", "€")
    fig.update_layout(hovermode="closest", bargap=0.45, showlegend=False)
    fig.update_yaxes(tickformat=",.0f")
    return fig


def monatsbilanz(tabelle: pd.DataFrame) -> go.Figure:
    """Gestapelte Monatsbilanz des Split-Stroms nach Quelle."""
    fig = go.Figure()
    for quelle, farbe in zip(tabelle.columns, (BLAU, ORANGE, AQUA), strict=False):
        if tabelle[quelle].sum() <= 0:
            continue
        fig.add_trace(
            go.Bar(
                x=tabelle.index,
                y=tabelle[quelle],
                name=quelle,
                marker_color=farbe,
                marker_line=dict(width=1, color="rgba(255,255,255,0.9)"),
                hovertemplate=f"{quelle}: %{{y:,.0f}} kWh<extra></extra>",
            )
        )
    fig.update_layout(barmode="stack", bargap=0.25, legend_traceorder="normal")
    return _layout(fig, "Strom für das Split-Gerät je Monat", "kWh")


def beispielwoche(woche: pd.DataFrame, mit_speicher: bool) -> go.Figure:
    """Stundenwerte einer Woche; Speicherinhalt in eigenem Feld (andere Einheit)."""
    zeilen = 2 if mit_speicher else 1
    fig = make_subplots(
        rows=zeilen,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3] if mit_speicher else [1.0],
    )
    for spalte, farbe in zip(
        ("PV-Erzeugung", "Gesamtlast", "Split-Leistung"), (BLAU, ORANGE, AQUA), strict=True
    ):
        fig.add_trace(
            go.Scatter(
                x=woche.index,
                y=woche[spalte],
                name=spalte,
                mode="lines",
                line=dict(color=farbe, width=2),
                hovertemplate=f"{spalte}: %{{y:.2f}} kW<extra></extra>",
            ),
            row=1,
            col=1,
        )
    if mit_speicher:
        fig.add_trace(
            go.Scatter(
                x=woche.index,
                y=woche["Speicherinhalt"],
                name="Speicherinhalt",
                mode="lines",
                line=dict(color=GELB, width=2),
                hovertemplate="Speicher: %{y:.1f} kWh<extra></extra>",
            ),
            row=2,
            col=1,
        )
    fig = _layout(fig, "Beispielwoche in der Übergangszeit", "kW")
    if mit_speicher:
        fig.update_yaxes(title_text="kWh", row=2, col=1)
    fig.update_xaxes(tickformat="%d.%m.")
    return fig
