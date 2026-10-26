# pypetrograph-web vs pypetrograph-app (Qt): feature parity

Branch `webapp` (from main 42d4ffb). Web = `pyPetrograph/webapp/` (FastAPI + vanilla JS canvas, no CDN).
Science: both call the same `pyPetrograph.app.amics_pipeline.AmicsSession` + `backend.SegBackend`; nothing duplicated.
✓ = same function, ≈ = same function, different widget, ✗ = gap.

| Step | Control (Qt) | Web | |
|---|---|---|---|
| 1 Files | BSE / Mineral map / Legend fields + "…" browse, companion auto-fill | text fields + "…" server-side file browser, same `file_match.complete` | ≈ |
| | Output folder, Sample | same | ✓ |
| | Open (primary) | Open (primary, top of panel) | ✓ |
| 2 Legend | Show combo (legend image / as used / mineral map) | Show select, same 3 views | ✓ |
| | Table: Use tick, RGB, Mineral, mark (✎ edited, + added, italic) | same | ✓ |
| | Rename in place, Revert to OCR, right-click menu | same; right-click opens a small menu dialog | ≈ |
| | + Add legend entry dialog: RGB spins, hex, palette, eyedropper, name, live errors | same (native colour picker + 36 swatches; eyedropper on legend image or mineral map) | ✓ |
| | Delete entry, Undo, Reset all to OCR, Load saved, Read again (OCR), Clear saved… (confirm) | same | ✓ |
| | Keys Del, Ctrl+Z | Del, Z / Ctrl+Z | ✓ |
| | Use this legend (primary) | same, n / N progress | ✓ |
| 3 BSE & footer | Footer row spin, Use detected, red dashed line | same | ✓ |
| | Use this footer | same | ✓ |
| 4 Non-grain | click pick / right-click remove / Z / C / T / Enter | same | ✓ |
| | Show blue tint (T), Gray level threshold, Undo last pick, Reset picks, Load saved, Clear saved… | same | ✓ |
| | Apply non-grain regions | same | ✓ |
| 5 Scale | two-click line, right-click remove, B whole/footer strip | same | ✓ |
| | Bar length, Unit, µm/px, Undo last point, Reset line, Check footer text (OCR), Clear saved… | same | ✓ |
| | Apply scale; skip when leaving | Apply scale + explicit "Skip (no new scale)" button (segment step also auto-skips like Qt) | ≈ |
| | "(optional)" marker in the step list | lead text says Optional; rail has no "(optional)" | ✗ minor |
| 6 Segmentation | note "SEG finds the grains; SAM 2.1 refine is an optional extra pass, not an alternative" | same | ✓ |
| | SEG missing → Run segmentation off + link to SEG repo | same | ✓ |
| | SAM missing → SAM refine checkbox and Add grain (A) off | same | ✓ |
| | Min grain (px), Non-grain drop fraction, Load saved grains, Clear saved grains… | same | ✓ |
| | click select, Delete (D), Merge (M), Undo (Z), Deselect (Esc), Reset edits (confirm) | same | ✓ |
| | Add grain (A): points / negative points / box, Accept (Enter), Clear prompts (Esc) | same (box = drag in add mode) | ✓ untested here (no SAM on box) |
| | Run segmentation (n / N, patch) | same progress mapping | ✓ untested here (no SEG on box) |
| | Use this grain map (precomputes minerals, votes, figures) | same | ✓ |
| 7 Minerals | Figure combo (Composition / Overlay / Mixed / Unmapped), status table ✓ ▲ ✗ | same | ✓ |
| | + Add map colour not in legend, Remove added colour, Clear saved… | same (dialog with suggestion, palette, eyedropper on the mineral map) | ✓ |
| | (no primary button) | "Next: grain size" primary | ≈ |
| 8 Grain size | Figure combo, Replot grain size curves | same (+ "Next: outputs") | ✓ |
| 9 Outputs | Export CSV data sheets, Export all figures, Replot knolling, Open export folder, exported-files list (click to view) | same | ✓ |
| Global | zoom / pan canvas | wheel zoom, drag pan, F / double-click fit, +/− buttons; preview ≤1600 px, full image loaded when zoomed in | ✓ |
| | Controls box per step | collapsible Controls box per step + help bar | ✓ |
| | Log dock | Log toggle (top bar) | ≈ |
| | Progress "stage: done / total (pct%) · patch · elapsed · ETA" | "n / N (pct%) · stage · patch"; total seconds when done; no live elapsed/ETA | ✗ minor |
| | Quit | Quit (stops the server) | ✓ |
| | native OS look, single window | runs in any browser tab on 127.0.0.1 | ≈ |
