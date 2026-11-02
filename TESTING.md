# Tests und Reproduzierbarkeit — Micro AI

## Voraussetzungen
Python 3.12+; ausschließlich Standardbibliothek.

## Befehle
```text
python -m unittest -v
```

Start/Build: python evaluate.py --steps 250; oder context_cli.py train/generate gemäß README.

## Tatsächlicher Befund vom 02.11.2026
9 Tests bestanden: alle Kontext-Parameter per finiten Differenzen geprüft, Lernfähigkeit, korrekte Checkpoints/Abweisungen. CLI-Train/Load/Generate-Smoketest und fairer Baselinevergleich ausgeführt.

Bei fehlenden Paketen/Hardware/Betriebssystemen ist das Ergebnis „nicht ausgeführt“, kein Pass. Plattformnachweise aus älteren Detailberichten gelten nur für die dort genannten Versionen. Android-Automatisierung ausschließlich mit expliziter emulator-Seriennummer und bestätigtem ro.kernel.qemu=1. Physische Geräte, persönliche Medien und Produktionskonten sind keine Testfixtures.

## Vor Merge
`git diff --check` ausführen; relevante Funktionstests wiederholen, wenn Codeänderungen ihren Befund entwerten. Speicherdaten/alte Versionen erhalten. Fehlende externe Abnahmen in HANDOFF.md benennen.
