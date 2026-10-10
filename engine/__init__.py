"""Rechen-Engine des Klimarechners (UI-unabhängig)."""

import logging

# Bibliothek: Warnungen nur ausgeben, wenn die Anwendung Logging konfiguriert
logging.getLogger(__name__).addHandler(logging.NullHandler())
