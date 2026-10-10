# klimarechner

Prototyp eines Wirtschaftlichkeitsrechners für Split-Klimaanlagen (Luft-Luft-Wärmepumpen) in Kombination mit PV und optionalem Batteriespeicher. Zielgruppe ist der Installateur im Kundengespräch; das Ergebnis ist ein verständlicher Bericht für den Endkunden.

Kernideen:

- Stündliche Simulation über ein ganzes Jahr (8.760 h), wiederholt über mehrere historische Wetterjahre (PVGIS).
- Temperaturabhängige Effizienz (COP/EER) statt pauschaler Jahresarbeitszahl.
- Monte-Carlo über Wetter, Energiepreise und Nutzerverhalten, Ergebnisse als Bandbreiten (P10/P50/P90).
- Split als Teil- oder Zusatzheizung neben bestehender Gas-/Ölheizung als Standardfall.

Technik: Python 3.12, NumPy/pandas/SciPy, pydantic, Streamlit, Plotly, pytest, ruff.

## Status

Die vollständige Spezifikation steht in [docs/SPEC.md](docs/SPEC.md), inklusive Projektstruktur und Umsetzungsreihenfolge.

Umgesetzt (Schritt 1 der Reihenfolge):

- `config/defaults.yaml`: alle Defaults, Tabellen und Preisszenarien. Die Zahlenwerte sind **Platzhalter und vor Nutzung zu prüfen**.
- `engine/inputs.py`: pydantic-Modelle für Eingaben und Annahmen mit Plausibilitätsprüfungen; `lade_konfiguration()` lädt die Defaults und legt Überschreibungen darüber.

Umgesetzt (Schritt 2):

- `engine/weather.py`: PVGIS-Abruf (`seriescalc`, alle Jahre, 1 kWp) mit Dateicache unter `data/cache/pvgis/`, plus synthetisches Wetterjahr für Tests und Offline-Betrieb.
- `engine/loads.py`: Heizbedarf, Kühlbedarf der Split-Räume, Haushaltsstrom aus Standardlastprofil (BDEW-Verfahren mit Dynamisierung).
- `engine/heatpump.py`: COP/EER aus Prüfpunkten oder Carnot-Modell, kalibriert auf SCOP/SEER; Abtau- und Teillastabschläge; Leistungsgrenzen.
- `engine/pv_battery.py`: regelbasierter Eigenverbrauch mit Speicher und stündlicher Energiebilanz (vektorisiert über Pfade).
- `engine/simulation.py`: ein Wetterjahr für Referenz und Variante mit Kennzahlen (JAZ, PV-Anteil am Split-Heizstrom, Brennstoff, Autarkie, Eigenverbrauch).

Umgesetzt (Schritt 3):

- `engine/economics.py`: deterministische Preispfade je Szenario (Drift, CO2-Aufschlag), Jahreskosten für Referenz und Variante, Ersparnis, kumulierter Cashflow, Barwert, interner Zinsfuß, statische und dynamische Amortisation. Kühlen wird als Mehrkosten der Variante gerechnet und separat ausgewiesen.

`data/lastprofile/h0_naeherung.csv` ist eine **Näherung**, nicht das offizielle BDEW-H0-Profil (siehe README dort).

## Entwicklung

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest
```

Beispiel:

```python
from engine.inputs import lade_konfiguration

konfig = lade_konfiguration({"eingaben": {"gebaeude": {"split_flaeche_m2": 40}}})
print(konfig.eingaben.gebaeude.split_anteil, konfig.preisszenario)
```

Jahressimulation mit Default-Eingaben (`--synthetisch` ohne Internetzugang, sonst PVGIS):

```bash
python -m engine.simulation --synthetisch
python -m engine.economics --synthetisch
```
