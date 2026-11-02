# Übergabe — Micro AI

Stand: 02.11.2026.
Ausgangscommit: a11678f1dd60685a05457781c0184e97f5c7a1c2
Arbeitsbranch: codex/portfolio-20261102

## Ausgangszustand und Ergebnis
Kleines lokales KI-Lernexperiment: Bigram plus Kontextnetz, Training, numerische JSON-Checkpoints und faire Held-out-Auswertung. Kein Chatbot-Pflichtziel.

12 Tests bestanden: alle Kontext-Parameter per finiten Differenzen geprüft, Lernfähigkeit, korrekte Checkpoints/Abweisungen. CLI-Train/Load/Generate-Smoketest und fairer Baselinevergleich ausgeführt.

## Änderungen dieser Arbeitsrunde

- `AGENTS.md`
- `HANDOFF.md`
- `README.md`
- `TESTING.md`
- `context_cli.py`
- `context_model.py`
- `evaluate.py`
- `micro_ai.py`
- `test_context_model.py`
- `test_micro_ai.py`

Nur einzeln geprüfte Quell-/Dokumentationsdateien. Keine generierten Medien, Testdaten oder Zugangsdaten werden aufgenommen.

## Offen und nächste Etappe
Lokaler Datenaufbereitungs-/Trainings-/Inferenzablauf ist implementiert und geprüft. Mehrfach-Seed-Auswertung (1/42/2026) wurde ausgeführt. Als Nächstes den kleinen Lernumfang mit einer Lernperson abnehmen; kein Cloudmodell nötig.

Keine Pickle-Checkpoints, fremden Korpora oder Modellgewichte ungeprüft übernehmen. Generierte Zeichenfolgen nicht als verlässliche Antworten ausgeben.

Kein Nachweis realer Nutzer, Einnahmen oder universeller Plattformunterstützung. Dokumentation ist kein Ersatz für fehlende Hardware-, Netzwerk- oder Produktabnahmen.

## Ergänzung: lokale Datenvorbereitung

Neu/angepasst: dataset.py, prepare_corpus.py, test_dataset.py, micro_ai.py, context_cli.py, evaluate.py, .gitignore und README.md. Zwölf Tests bestanden; Prepare → Context-Train → Generate sowie Bigram-Train mit vorbereiteten Daten tatsächlich ausgeführt. Neue Ausgaben liegen ausschließlich in isolierten Arbeitsordnern.

Mehrfach-Seed-Auswertung mit 1/42/2026 und 250 Schritten ausgeführt: mittlere Held-out-Verluste 2,714 (Häufigkeit), 1,974 (Bigram), 1,508 (Kontextnetz). Gleiche 72 Zielzeichen, kein allgemeiner Sprachbenchmark. Zusätzlich benchmark.py.
