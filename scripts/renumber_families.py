"""Renumber the MNAR missingness families in a sweep JSON written before 2026-10-08:
C2 (cadence) -> C1, C3 (low-value) -> C2, C4 (reflex) -> C3. A file is old iff it contains family C4
(new files never do), so the script is safe to run repeatedly. Usage: renumber_families.py <json>..."""
import json, sys
NEW = {"C2": "C1", "C3": "C2", "C4": "C3"}
for p in sys.argv[1:]:
    rows = json.load(open(p))
    if not any(r.get("family") == "C4" for r in rows): continue
    for r in rows:
        if r.get("family") in NEW: r["family"] = NEW[r["family"]]
    json.dump(rows, open(p, "w"), indent=2); print("renumbered", p)
