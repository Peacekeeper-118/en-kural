"""Pluggable voice-conversion engines.

An engine implements :class:`enkural.vc.base.VoiceConverter`.  Engines are
registered in :mod:`enkural.vc.registry`; the rest of the application only
talks to the abstract interface, so a better model can be dropped in later
without touching the API, the pipeline or the UI.
"""
