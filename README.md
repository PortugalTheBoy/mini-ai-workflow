# Mini-KI-Workflow: Supportanfragen klassifizieren

Ein kleines Python-Projekt, das deutschsprachige Supporttexte automatisch den Kategorien **Technik**, **Abrechnung** oder **Vertrag** zuordnet. Enthalten sind Datenprüfung, Modelltraining, Evaluation, Vorhersage, Protokollierung und automatisierte Tests mit GitHub Actions.

**Beispiel:** „Meine Rechnung wurde doppelt abgebucht“ → **Abrechnung**.

Das Modell lernt aus Beispielen: TF-IDF wandelt Texte in numerische Merkmale um; eine logistische Regression ordnet sie einer Kategorie zu. Es werden weder ein Sprachmodell noch ein kostenpflichtiger KI-Dienst benötigt. Nach Installation der Python-Pakete läuft die Anwendung lokal ohne externe API.

## Ablauf

1. **Daten prüfen:** CSV laden und ungültige Zeilen sowie doppelte Texte zurückweisen.
2. **Daten teilen:** 80 % Training und 20 % Test; jede Kategorie ist in beiden Teilen vertreten. Der Zufallsstartwert ist festgelegt.
3. **Modell trainieren:** Textvektorisierung und Klassifikator lernen ausschließlich aus den Trainingsdaten.
4. **Bewerten:** Accuracy, Präzision, Recall, F1 und Konfusionsmatrix auf den zurückgehaltenen Testdaten berechnen. Eine Mehrheitsklassen-Baseline dient als Vergleich.
5. **Vorhersagen:** Neue Supporttexte mit dem gespeicherten Modell klassifizieren.
6. **Protokollieren:** Kategorie, Modellversion, UTC-Zeitstempel, Laufzeit und Textlänge als JSON-Zeile speichern.

Das gespeicherte Modell entspricht genau dem auf den Testdaten bewerteten Modell. Es wird anschließend nicht noch einmal auf dem gesamten Datensatz trainiert.

## Schnellstart

Voraussetzung: **Python 3.12**, für den späteren Upload zusätzlich **Git**. Alle Befehle im entpackten Projektordner `mini-ai-workflow` ausführen.

### Windows – PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe workflow.py train
.\.venv\Scripts\python.exe workflow.py predict --text "Meine Rechnung wurde doppelt abgebucht"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Die virtuelle Umgebung muss hierbei nicht aktiviert werden; damit ist auch keine Änderung der PowerShell-Ausführungsrichtlinie erforderlich.

### macOS / Linux

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python workflow.py train
.venv/bin/python workflow.py predict --text "Meine Rechnung wurde doppelt abgebucht"
.venv/bin/python -m unittest discover -s tests -v
```

| Befehl | Zweck |
| --- | --- |
| `-m venv .venv` | Isolierte Python-Umgebung anlegen |
| `-m pip install -r requirements.txt` | Festgelegte Paketversionen installieren |
| `workflow.py train` | CSV prüfen, Modell trainieren und Evaluation speichern |
| `workflow.py predict --text "..."` | Neuen Text klassifizieren und Ergebnis protokollieren |
| `-m unittest discover -s tests -v` | Automatisierte Tests ausführen |

## Dateien und Ergebnisse

| Datei | Inhalt |
| --- | --- |
| `workflow.py` | Trainings- und Vorhersagelogik mit Kommandozeilenschnittstelle |
| `data/tickets.csv` | 90 künstlich formulierte Supportanfragen, je 30 pro Kategorie |
| `tests/test_workflow.py` | Tests für Training, Vorhersage, Datenfehler, Logging und Kommandozeile |
| `requirements.txt` | Paketversionen für Python 3.12 |
| `.github/workflows/ci.yml` | Automatischer Ablauf bei Push, Pull Request oder manuellem Start |
| `LICENSE` | MIT-Lizenz für das Projekt |
| `BEISPIELERGEBNIS.md` | Ergebnis des lokal ausgeführten Beispiels |

Beim Training entstehen `artifacts/model.joblib` und `artifacts/metrics.json`. Vorhersagen werden an `logs/predictions.jsonl` angehängt. Diese Laufzeitdateien sind über `.gitignore` vom Git-Upload ausgeschlossen; auf GitHub werden der Evaluationsbericht und das Beispielprotokoll separat als Workflow-Artefakt gespeichert.

Die Laufzeit umfasst Laden und Ausführen des Modells, jedoch nicht das Schreiben der Protokolldatei. Rohtexte werden nicht in die Protokolldatei geschrieben. Das Protokoll ermöglicht eine einfache Sicht auf technische Kennzahlen; zur Überwachung der tatsächlichen Klassifikationsgüte wären zusätzlich nachträglich bestätigte Kategorien notwendig.

## Auf GitHub hochladen

1. Auf [GitHub](https://github.com/new) ein neues Repository namens `mini-ai-workflow` erstellen. **README, Lizenz und .gitignore dort nicht vorab erzeugen**, weil diese Dateien bereits enthalten sind.
2. Im entpackten Projektordner ein Terminal öffnen und die folgenden Befehle ausführen. `DEIN-BENUTZERNAME` durch deinen GitHub-Namen ersetzen. Git muss installiert und die Anmeldung bei GitHub eingerichtet sein.

```bash
git init -b main
git add .
git commit -m "Add mini AI workflow for support ticket classification"
git remote add origin https://github.com/DEIN-BENUTZERNAME/mini-ai-workflow.git
git push -u origin main
```

| Befehl | Zweck |
| --- | --- |
| `git init -b main` | Lokales Repository mit Hauptbranch `main` anlegen |
| `git add .` | Projektdateien für den ersten Commit vormerken |
| `git commit ...` | Ersten lokalen Versionsstand speichern |
| `git remote add origin ...` | Lokales Projekt mit dem GitHub-Repository verbinden |
| `git push -u origin main` | Commit auf GitHub hochladen |

3. Anschließend auf GitHub den Reiter **Actions** öffnen. Der Workflow installiert die Abhängigkeiten, führt die Tests aus, trainiert das Modell und verarbeitet eine Beispielanfrage. Unter dem jeweiligen Lauf steht `ki-workflow-ergebnisse` als Download bereit. Der Ablauf kann dort auch manuell über **Run workflow** gestartet werden.

Bei einem ZIP-Download wird die ZIP-Datei zuerst entpackt. Auf GitHub gehören die **Projektdateien samt `.github/workflows/ci.yml` in das Stammverzeichnis des Repositorys**, nicht das ungeöffnete ZIP-Archiv oder ein zusätzlicher übergeordneter Ordner.

## Eigene Daten verwenden

Die CSV enthält genau zwei Spalten: `text,label`. Zulässige Kategorien sind `Technik`, `Abrechnung` und `Vertrag`; jede benötigt mindestens fünf unterschiedliche Texte. Für eine aussagekräftige Anwendung sind wesentlich mehr repräsentative und unabhängig gelabelte Beispiele erforderlich.

```csv
text,label
Die App startet nicht,Technik
Meine Rechnung ist falsch,Abrechnung
Ich möchte meinen Vertrag kündigen,Vertrag
```

Dieser Ausschnitt zeigt nur das Format und ist allein zu klein zum Trainieren. Enthält ein Text ein Komma, muss er gemäß CSV-Format in Anführungszeichen stehen. Eine andere Datei kann über `workflow.py train --data data/eigene_tickets.csv` verwendet werden. Die drei Kategorien bleiben dabei unverändert; für neue Kategorien muss auch die Konstante `LABELS` angepasst werden.

## Grenzen des Beispiels

Dies ist eine Lern- und Portfolio-Demo mit synthetischen Daten. Die Evaluation zeigt nur die Leistung auf dem kleinen Demo-Testbestand; daraus lässt sich keine belastbare Aussage über reale Supportanfragen oder Fairness ableiten. Das Modell wählt immer eine der drei Kategorien, auch bei fachfremden oder mehrdeutigen Texten. Eingaben sollten für eine praktische Nutzung geprüft und die Datenbasis erweitert werden. `joblib`-Modelle nur aus vertrauenswürdigen Quellen laden.

## Quellen

Die Beispieldaten wurden für dieses Projekt künstlich formuliert; es wurden keine Kundendaten oder externen Textdatensätze verwendet. Technische Grundlagen und Dokumentation:

- [scikit-learn: Textklassifikation mit TF-IDF](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html)
- [scikit-learn: Pipeline und Vermeidung von Data Leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
- [GitHub: Python mit Actions erstellen und testen](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)
- [GitHub: Lokalen Quellcode hochladen](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)
