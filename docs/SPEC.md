# Prototyp: Wirtschaftlichkeitsrechner Split-Klima + PV

## 1. Ziel

Ein Rechner, mit dem ein Klima-Installateur im Kundengespräch zeigen kann, was eine Split-Klimaanlage (Luft-Luft-Wärmepumpe) in Kombination mit PV und optionalem Batteriespeicher wirtschaftlich bringt – beim Kühlen und vor allem beim (Zu-)Heizen in der Übergangszeit.

Der Unterschied zu bestehenden Gratis-Rechnern:

- **Stündliche Simulation (8.760 h/Jahr)** statt Monats- oder Jahresbilanz – entscheidend für die Gleichzeitigkeit von PV-Ertrag und Heiz-/Kühlbedarf.
- **Temperaturabhängige Effizienz (COP/EER)** statt pauschaler Jahresarbeitszahl.
- **Bandbreiten statt Einzelwerte**: Monte-Carlo über Wetterjahre, Energiepreispfade und Nutzerverhalten. Ergebnis z. B. „Amortisation mit 80 % Wahrscheinlichkeit in 7–11 Jahren“.
- **Luft-Luft als Teil- oder Zusatzheizung** neben einer bestehenden Gas-/Ölheizung ist der Standardfall, nicht die Ausnahme.

Zielgruppe der Oberfläche: Installateur (nicht Endkunde). Ergebnis: ein verständlicher Bericht für den Endkunden.

## 2. Scope des Prototyps

### Drin

- Eingabemaske für Standort, Gebäude, Geräte, PV, Speicher, Haushaltsstrom, Bestandsheizung, Preise.
- Stündliche Energiesimulation für ein Jahr, wiederholt über mehrere Wetterjahre.
- Monte-Carlo-Wirtschaftlichkeit über die Nutzungsdauer.
- Ergebnisseite mit Kennzahlen und Diagrammen.
- Export eines einfachen PDF-Berichts.

### Nicht drin (bewusst)

- Normgerechte Heizlast (DIN EN 12831) oder Kühllast (VDI 2078) – nur vereinfachte Schätzung.
- Förderprogramme, Hydraulik, Rohrleitungsplanung.
- Benutzerkonten, Mandanten, Datenbank, Zahlungen.
- Echte Kundendaten – nur Test- und Beispieldaten.
- Optimierte Speichersteuerung für dynamische Tarife (spätere Ausbaustufe, siehe Abschnitt 9).

## 3. Eingaben

Alle Eingaben mit sinnvollen Defaults, damit ein Durchlauf mit wenigen Angaben möglich ist. Defaults zentral in einer Konfigurationsdatei (`config/defaults.yaml`); Zahlenwerte dort sind **Platzhalter und vor Nutzung zu prüfen**.

| Gruppe | Eingabe | Einheit / Typ | Hinweis |
| --- | --- | --- | --- |
| Standort | PLZ oder Koordinaten | lat/lon | Für Wetter- und PV-Daten |
| Gebäude | Beheizte Fläche gesamt | m² | |
| Gebäude | Fläche der Räume mit Split-Gerät | m² | Anteil, den Split versorgen kann |
| Gebäude | Baualtersklasse / Dämmstandard | Auswahl | Mappt auf spezifische Heizlast W/m² |
| Gebäude | Raumsolltemperatur Heizen / Kühlen | °C | z. B. 21 / 25 |
| Gebäude | Heizgrenztemperatur | °C | z. B. 15 |
| Geräte | Anzahl Innengeräte, Nennleistung Heizen/Kühlen | kW | |
| Geräte | COP bei Prüfpunkten (EN 14825: −7, +2, +7, +12 °C) oder SCOP | – | Fallback: Default-Kurve aus SCOP |
| Geräte | SEER / EER | – | Für Kühlbetrieb |
| Nutzung | Heizstrategie | Auswahl | „Übergangszeit“ (Split nur oberhalb Schwelltemperatur), „bevorzugt Split“, „Vollheizung Teilräume“ |
| Nutzung | Kühlen aktiv | bool + Sollwert | |
| PV | Leistung, Ausrichtung, Neigung | kWp, °, ° | Optional: vorhanden / neu |
| Speicher | Kapazität, Lade-/Entladeleistung, Wirkungsgrad | kWh, kW, % | Optional |
| Haushalt | Jahresstromverbrauch ohne Klima | kWh/a | Skaliert Standardlastprofil |
| Bestand | Heizträger, Kesselwirkungsgrad | Gas/Öl, % | Für eingesparte Brennstoffkosten |
| Preise | Strompreis Bezug, Brennstoffpreis, Einspeisevergütung | €/kWh | Startwerte der Preispfade |
| Preise | Annahmen Preisentwicklung | Szenario-Auswahl | Siehe Abschnitt 5 |
| Kosten | Investition Split (und ggf. PV/Speicher), jährliche Wartung | € | |
| Rahmen | Betrachtungszeitraum, Kalkulationszins | Jahre, % | z. B. 15 Jahre |

## 4. Physikalisches Modell (stündlich)

Alle Größen als NumPy-Arrays der Länge 8.760, vektorisiert über Monte-Carlo-Pfade (Form `[n_paths, 8760]`).

### 4.1 Wetter und PV

- Stündliche Außentemperatur und Einstrahlung pro Standort über **PVGIS** (EU JRC, kostenlose API): `seriescalc` liefert stündliche Zeitreihen für mehrere Kalenderjahre, inkl. PV-Ertrag für gegebene kWp, Ausrichtung und Neigung.
- Mehrere historische Jahre abrufen (z. B. alle verfügbaren) und **lokal cachen** (Datei pro Standort/Parameter), damit die API nicht bei jedem Lauf angefragt wird.
- Wetterjahre sind die Quelle für Wetterunsicherheit (Ziehen mit Zurücklegen).

### 4.2 Heizbedarf (vereinfacht)

- Wärmeverlustkoeffizient H [W/K] aus spezifischer Heizlast (Default-Tabelle nach Baualtersklasse) × Fläche, bezogen auf die Differenz Innen- zu Norm-Außentemperatur.
- Stündlich: `Q_heat(t) = H * max(0, T_heizgrenze - T_aussen(t))` (Gradtagzahl-Prinzip; die Differenz zwischen Raumsoll- und Heizgrenztemperatur deckt pauschal solare und interne Gewinne ab). Entscheidung vom 2026-10-10; ursprünglich `H * max(0, T_soll - T_aussen(t))`, was ohne Gewinne rund 300 kWh/m²a ergab.
- Aufteilung: Anteil der Split-Räume an der Fläche → maximal von Split deckbarer Bedarf. Begrenzung durch die temperaturabhängige maximale Heizleistung der Geräte. Rest übernimmt die Bestandsheizung.
- Heizstrategie bestimmt, wann Split heizt (z. B. nur oberhalb einer Außentemperaturschwelle).

### 4.3 Kühlbedarf (vereinfacht)

- `Q_cool(t) = max(0, H * (T_aussen(t) - T_soll_kuehl) + a_solar * G(t) + Q_intern)`, nur in Stunden mit aktivem Kühlen.
- `a_solar` als einfacher Faktor nach Fensteranteil/Ausrichtung (Default konservativ), interne Lasten als Konstante.

### 4.4 Effizienz der Split-Geräte

- COP(T) durch lineare Interpolation zwischen den Prüfpunkten (EN 14825), außerhalb linear extrapoliert und begrenzt.
- Fallback ohne Prüfpunkte: Carnot-Gütegrad-Modell `COP = eta * T_kond / (T_kond - T_verd)`, `eta` so kalibriert, dass die Saisonzahl dem angegebenen SCOP entspricht.
- Analog EER(T) für Kühlen.
- Abschlag für Abtauen bei Temperaturen um 0 °C und Teillast (Faktoren in Config).
- Elektrische Leistung: `P_el(t) = Q(t) / COP(t)`.

### 4.5 Haushaltsstrom

- Standardlastprofil (BDEW H0 oder Nachfolger, als CSV im Repo, Quelle dokumentieren), skaliert auf Jahresverbrauch.
- Pro Pfad multiplikatives Rauschen, damit Eigenverbrauch nicht künstlich glatt ist.

### 4.6 Speicher und Energiebilanz

- Einfache regelbasierte Eigenverbrauchsstrategie: PV-Überschuss lädt, Defizit entlädt; SoC-Grenzen, Lade-/Entladeleistung, Wirkungsgrad.
- Pro Stunde: Gesamtlast = Haushalt + Split. Ergebnis: Eigenverbrauch PV, Einspeisung, Netzbezug, SoC.
- **Invariante (Test!)**: Energieerhaltung je Stunde – PV + Netzbezug + Speicherentladung = Last + Einspeisung + Speicherladung + Verluste.

## 5. Wirtschaftlichkeit und Unsicherheit

### 5.1 Vergleichsfälle

Immer zwei Fälle mit identischen Wetter- und Preispfaden rechnen:

- **Referenz**: ohne Split (Heizen komplett über Bestand, kein Kühlen bzw. Kühlen als Komfortgewinn nicht monetarisiert).
- **Variante**: mit Split (und ggf. neuer PV/Speicher).

Ersparnis = Kosten Referenz − Kosten Variante. Kühlen wird als zusätzliche Stromkosten ausgewiesen, nicht als Ersparnis.

### 5.2 Kosten je Jahr

- Strom: Netzbezug × Strompreis − Einspeisung × Vergütung.
- Brennstoff: Wärme aus Bestandsheizung / Kesselwirkungsgrad × Brennstoffpreis.
- Wartung, optional CO2-Preis-Aufschlag auf fossile Brennstoffe als Szenario.

### 5.3 Monte-Carlo

Pro Pfad (Default 1.000 Pfade, Seed fixierbar):

- **Wetter**: pro Simulationsjahr ein historisches Wetterjahr ziehen.
- **Preise**: Strom- und Brennstoffpreis als korrelierte log-normale Pfade (Drift + Volatilität + Korrelation je Szenario aus Config, z. B. „moderat“, „steigend“, „volatil“).
- **Verhalten**: Sollwerte und Heizstrategie-Schwelle leicht variieren (Normalverteilung um Eingabe).
- **Technik**: COP-Abschlag (z. B. gleichverteilt 0–10 %).

Energiesimulation nur einmal je Wetterjahr und Parametersatz rechnen und cachen; Preise erst danach anwenden (Performance).

### 5.4 Kennzahlen

- Barwert (NPV), interner Zinsfuß, Amortisationszeit – jeweils als P10 / P50 / P90.
- Wahrscheinlichkeit, dass sich die Investition innerhalb der Nutzungsdauer amortisiert.
- Energiekennzahlen (P50 + Bandbreite): Anteil des Split-Heizbedarfs, der durch PV gedeckt wird; Jahresarbeitszahl Heizen; eingesparte Brennstoffmenge; Autarkiegrad; Eigenverbrauchsquote.

## 6. Ausgaben / UI

- **Eingabeseite**: kompakt, Defaults vorbelegt, „Erweitert“-Bereich für Prüfpunkte, Szenarien, Monte-Carlo-Parameter.
- **Ergebnisseite**:
  - Kopfzeile mit 3–4 Kernaussagen in Klartext, z. B. „Amortisation in 7–11 Jahren (80 % Wahrscheinlichkeit)“.
  - Fächerdiagramm kumulierter Cashflow über die Jahre (P10–P90-Band, P50-Linie).
  - Histogramm der Amortisationszeit.
  - Monatliche Energiebilanz (gestapelt): Split-Strom aus PV / Speicher / Netz.
  - Beispielwoche stündlich (Übergangszeit): PV, Last, Split-Leistung, SoC.
- **PDF-Bericht** für den Endkunden: Eingaben, Kernaussagen, zwei bis drei Diagramme, Annahmen und Haftungshinweis („Schätzung auf Basis vereinfachter Modelle, keine normgerechte Auslegung, keine Garantie“).

## 7. Technik

- Python 3.12, NumPy, pandas, SciPy; `pydantic` für Eingabe-Validierung; `httpx` + Dateicache für PVGIS.
- **UI für den Prototyp: Streamlit** (schnell iterierbar). Die Rechen-Engine bleibt vollständig UI-unabhängig, damit später ein FastAPI-Backend mit eigenem Web-Frontend möglich ist.
- Diagramme: Plotly (interaktiv in Streamlit, statisch exportierbar für PDF).
- PDF: z. B. WeasyPrint oder ReportLab.
- Tests: pytest. Linting/Format: ruff.

### Projektstruktur (Vorschlag)

```
klimarechner/
  config/defaults.yaml        # alle Defaults und Szenarien, als Platzhalter markiert
  data/lastprofile/           # Standardlastprofil-CSV + Quellenangabe
  engine/
    inputs.py                 # pydantic-Modelle
    weather.py                # PVGIS-Abruf, Cache, Wetterjahre
    loads.py                  # Heiz-, Kühl-, Haushaltslast
    heatpump.py               # COP/EER-Modelle, Leistungsgrenzen
    pv_battery.py             # Speicherlogik, Energiebilanz
    economics.py              # Kosten, Cashflows, NPV, IRR, Amortisation
    montecarlo.py             # Pfade, Szenarien, Aggregation
    report.py                 # PDF
  app/streamlit_app.py
  tests/
```

## 8. Qualität und Tests

- Energieerhaltung pro Stunde (Toleranz 1e-6).
- Grenzfälle: keine PV, kein Speicher, kein Kühlen, Split-Fläche 0 → Variante = Referenz.
- Plausibilität: simulierte Jahresarbeitszahl Heizen für typische Geräte im Bereich realistischer Werte für Luft-Luft; Abweichungen loggen.
- Reproduzierbarkeit: gleicher Seed → identische Ergebnisse.
- Laufzeit: 1.000 Pfade × 15 Jahre auf einem Laptop in wenigen Sekunden (Energiesimulation je Wetterjahr cachen).

## 9. Umsetzungsreihenfolge

1. Eingabemodelle + Defaults-Config.
2. PVGIS-Abruf mit Cache; ein Wetterjahr deterministisch durchrechnen (Heizlast, COP, Haushalt, PV, Speicher, Energiebilanz) + Tests.
3. Wirtschaftlichkeit deterministisch (ein Preis-Szenario) + Vergleich Referenz/Variante.
4. Monte-Carlo über Wetterjahre, Preise, Verhalten; Kennzahlen als Perzentile.
5. Streamlit-Oberfläche mit Ergebnisseite.
6. PDF-Bericht.
7. Später: Optimierung der Speicher- und Split-Steuerung bei dynamischen Stromtarifen (LP je Tag mit Day-Ahead-Preisen), White-Label (Logo/Farben des Betriebs), Web-Widget zur Anfragequalifizierung.

## 10. Offene Fragen (mit Installateur klären)

- Welche Eingaben hat er im Kundengespräch realistisch zur Hand?
- Welche Gerätedaten (Prüfpunkte, SCOP) nutzt er, und welche Hersteller dominieren?
- Wie wichtig ist Kühlen vs. Heizen in seinen Gesprächen?
- Welche Form des Ergebnisses überzeugt Kunden: eine Zahl, eine Bandbreite, ein Diagramm?
- Wie oft spielen PV, Speicher und dynamische Tarife eine Rolle?
