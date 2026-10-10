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
