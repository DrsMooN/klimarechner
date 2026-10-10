# klimarechner

Prototyp eines Wirtschaftlichkeitsrechners für Split-Klimaanlagen (Luft-Luft-Wärmepumpen) in Kombination mit PV und optionalem Batteriespeicher. Zielgruppe ist der Installateur im Kundengespräch; das Ergebnis ist ein verständlicher Bericht für den Endkunden.

Kernideen:

- Stündliche Simulation über ein ganzes Jahr (8.760 h), wiederholt über mehrere historische Wetterjahre (PVGIS).
- Temperaturabhängige Effizienz (COP/EER) statt pauschaler Jahresarbeitszahl.
- Monte-Carlo über Wetter, Energiepreise und Nutzerverhalten, Ergebnisse als Bandbreiten (P10/P50/P90).
- Split als Teil- oder Zusatzheizung neben bestehender Gas-/Ölheizung als Standardfall.

Technik: Python 3.12, NumPy/pandas/SciPy, pydantic, Streamlit, Plotly, pytest, ruff.

## Status

Noch keine Implementierung. Die vollständige Spezifikation steht in [docs/SPEC.md](docs/SPEC.md), inklusive Projektstruktur und Umsetzungsreihenfolge.
