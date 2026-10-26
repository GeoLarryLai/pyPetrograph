# ppweb progress notes
- Stage 0: cloned at 42d4ffb, branch webapp. /tmp/pwvenv missing on box.
- env: /workspace/ppenv (py3.12, -e ppweb[test] + PySide6-Essentials, fastapi, uvicorn, playwright)
- reference notebook-4 10B outputs (at 42d4ffb): /workspace/ppq/outputs/amics/10B (incl. grains.geojson). Box has no SEG -> equivalence uses those grains as "saved grains".
- pyLogDigitizer shell copied to /workspace/plshell (read-only template)
- Stage 1: shell copied into pyPetrograph/webapp/shell
- Stage 2: webapp/state.py written (Qt-free step logic)
- Stage 3: routes.py written + imports OK
- Stage 4: index.html, app.css, viewer.js written; next app.js
- Stage 5: app.js, __init__ main, pyproject web extra + script
