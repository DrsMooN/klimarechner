# Standardlastprofile

Jede CSV enthält 24 Stundenwerte (Spalte `stunde` = Beginn der Stunde, 0–23) je
Typtag: `<saison>_<tagtyp>` mit Saison `winter`, `sommer`, `uebergang` und Tagtyp
`werktag`, `samstag`, `sonntag`. Die absolute Höhe ist egal, das Profil wird auf
den Jahresverbrauch skaliert (siehe `engine/loads.py`).

Saisons und Dynamisierung folgen dem Verfahren der BDEW-Standardlastprofile:

- Winter: 1.11.–20.3., Sommer: 15.5.–14.9., Übergang: übrige Tage.
- Dynamisierungsfaktor je Tag t (1–365):
  F(t) = −3,92·10⁻¹⁰·t⁴ + 3,2·10⁻⁷·t³ − 7,02·10⁻⁵·t² + 2,1·10⁻³·t + 1,24
- Feiertage werden im Prototyp nicht gesondert behandelt.

## h0_naeherung.csv

**Platzhalter.** Händisch angenäherte Tagesgänge in Anlehnung an das BDEW-Profil H0
(Haushalt), Werte grob in W je 1.000 kWh/a. Es sind **keine offiziellen BDEW-Daten**.
Vor produktiver Nutzung durch das offizielle H0-Profil (BDEW, Viertelstundenwerte,
zu Stundenmitteln aggregiert) oder dessen Nachfolger ersetzen und die Quelle hier
dokumentieren. Der Dateiname wird in `config/defaults.yaml` unter
`annahmen.haushalt.lastprofil` gewählt.
