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

from engine.auswertung import zahl
from engine.montecarlo import MonteCarloErgebnis

BLAU, ORANGE, AQUA, GELB = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BLAU_BAND = "rgba(42, 120, 214, 0.18)"
SCHRIFT = dict(family="Inter, system-ui, sans-serif", size=13)


def _layout(fig: go.Figure, titel: str, y_titel: str, x_titel: str = "") -> go.Figure:
    fig.update_layout(
        title=dict(text=titel, x=0, xanchor="left", font=dict(size=15)),
        font=SCHRIFT,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        hovermode="x unified",
        separators=",.",
    )
    fig.update_yaxes(title_text=y_titel, zeroline=False, gridcolor="rgba(128,128,128,0.18)")
    fig.update_xaxes(title_text=x_titel, showgrid=False)
    return fig


def faecher_cashflow(ergebnis: MonteCarloErgebnis) -> go.Figure:
    """Kumulierter Cashflow über die Jahre: P10–P90-Band und P50-Linie."""
    kum = ergebnis.kumulierter_cashflow
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
            fillcolor=BLAU_BAND,
            name="Bandbreite P10–P90",
            hovertemplate="P10: %{y:,.0f} €<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=jahre,
            y=p50,
            mode="lines",
            line=dict(color=BLAU, width=2),
            name="Mittlerer Fall (P50)",
            hovertemplate="P50: %{y:,.0f} €<extra></extra>",
        )
    )
    fig.add_hline(y=0, line=dict(color="rgba(128,128,128,0.7)", width=1))
    fig = _layout(fig, "Kumulierter Cashflow", "€", "Jahr")
    fig.update_yaxes(tickformat=",.0f")
    return fig


def histogramm_amortisation(ergebnis: MonteCarloErgebnis) -> go.Figure:
    """Verteilung der Amortisationszeit; nicht amortisierte Pfade im Titel."""
    amort = ergebnis.amortisation_jahre
    endlich = amort[np.isfinite(amort)]
    nicht = 1 - ergebnis.wahrscheinlichkeit_amortisation
    jahre = ergebnis.ersparnis.shape[1]

    fig = go.Figure(
        go.Histogram(
            x=endlich,
            xbins=dict(start=0, end=jahre, size=1),
            marker=dict(color=BLAU, line=dict(color="rgba(0,0,0,0)", width=2)),
            name="Pfade",
            hovertemplate="%{x} Jahre: %{y} Pfade<extra></extra>",
        )
    )
    titel = "Amortisationszeit"
    if nicht > 0:
        titel += f" ({zahl(nicht * 100)} % der Pfade amortisieren nicht in {jahre} Jahren)"
    fig = _layout(fig, titel, "Anzahl Pfade", "Jahre")
    fig.update_layout(hovermode="closest", bargap=0.08, showlegend=False)
    fig.update_xaxes(range=[0, jahre])
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
