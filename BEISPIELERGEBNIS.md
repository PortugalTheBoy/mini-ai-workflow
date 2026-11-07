# Lokal geprüftes Beispielergebnis

Stand: 7. November 2026. Ausgeführt mit Python 3.12.14 und den fünf in `requirements.txt` festgelegten Paketversionen.

## Trainings- und Testlauf

| Kennzahl | Ergebnis |
| --- | ---: |
| Künstliche Beispiele insgesamt | 90 |
| Trainingsbeispiele | 72 |
| Separat zurückgehaltene Testbeispiele | 18 |
| Korrekt klassifizierte Testbeispiele | 16 von 18 |
| Accuracy | 88.9% |
| Macro-F1 | 0.889 |
| Accuracy der Mehrheitsklassen-Baseline | 33.3% |
| Automatisierte Tests | 11 bestanden |

Die Evaluation beruht ausschließlich auf synthetischen Demo-Daten und ist kein Nachweis für die Leistungsfähigkeit auf realen Kundendaten.

## Konfusionsmatrix

Zeilen sind die tatsächlichen Kategorien; Spalten sind die Vorhersagen.

| Tatsächlich / Vorhergesagt | Abrechnung | Technik | Vertrag |
| --- | ---: | ---: | ---: |
| Abrechnung | 5 | 0 | 1 |
| Technik | 0 | 6 | 0 |
| Vertrag | 1 | 0 | 5 |

## Beispielvorhersage

Eingabe: `Meine Rechnung wurde doppelt abgebucht`

Ausgabe des tatsächlichen lokalen Laufs:

```json
{
  "category": "Abrechnung",
  "model_version": "v1-ecc3bf85245a-sklearn1.8.0",
  "timestamp_utc": "2026-11-07T11:03:35.455008+00:00",
  "runtime_ms": 2.571,
  "text_length": 38
}
```

Zeitstempel und Laufzeit variieren bei erneuter Ausführung. Das Laufzeitprotokoll enthält den Rohtext nicht.

## Umfang der Prüfung

- Tests für Training und gespeichertes Modell, neue Vorhersagen, leere oder ungültige Eingaben, fehlende Modelldatei, CSV-Spalten, doppelte Texte, unbekannte Kategorien, zu wenige Beispiele, JSON-Protokollierung und Kommandozeile bestanden.
- Training und die oben gezeigte Vorhersage zusätzlich über die Kommandozeile ausgeführt.
- Alle Paketversionen mit `requirements.txt` abgeglichen.
- GitHub-Actions-Datei als YAML eingelesen; Start-Ereignisse und Leseberechtigung geprüft.

Der GitHub-Actions-Lauf selbst kann erst nach dem Hochladen in ein GitHub-Repository ausgeführt werden.
