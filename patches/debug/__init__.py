"""Env-gated diagnostic probes.

Modules here are imported lazily and only when their controlling env var is
set. Keep this package's ``__init__`` side-effect free so that importing a
single probe module never drags the others (each parses its own env spec at
import time).
"""
