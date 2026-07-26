"""FastAPI + WebSocket front door for the KANCHA pipeline.

This package is the transport layer that lets jarvis_frontend (Electron/React)
talk to the existing EventBus-driven pipeline in core/, memory/, nlu/,
reasoning/, tasks/, input/, output/. It does not contain any assistant logic
itself — see core/pipeline.py for pipeline construction and answers/guide.md
for how to extend this when adding new backend features.
"""
