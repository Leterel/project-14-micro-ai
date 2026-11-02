# Übergabe — Micro AI

Stand: 02.11.2026.
Ausgangscommit: a11678f1dd60685a05457781c0184e97f5c7a1c2
Arbeitsbranch: codex/portfolio-20261102

## Ausgangszustand und Ergebnis
Kleines lokales KI-Lernexperiment: Bigram plus Kontextnetz, Training, numerische JSON-Checkpoints und faire Held-out-Auswertung. Kein Chatbot-Pflichtziel.

9 Tests bestanden: alle Kontext-Parameter per finiten Differenzen geprüft, Lernfähigkeit, korrekte Checkpoints/Abweisungen. CLI-Train/Load/Generate-Smoketest und fairer Baselinevergleich ausgeführt.

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
Didaktischen Daten-/Tokenisierungsablauf kompakt erklären und reproduzierbare Mehrfach-Seed-Auswertung ergänzen; kein Cloudmodell nötig.

Keine Pickle-Checkpoints, fremden Korpora oder Modellgewichte ungeprüft übernehmen. Generierte Zeichenfolgen nicht als verlässliche Antworten ausgeben.

Kein Nachweis realer Nutzer, Einnahmen oder universeller Plattformunterstützung. Dokumentation ist kein Ersatz für fehlende Hardware-, Netzwerk- oder Produktabnahmen.
