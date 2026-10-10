"""Human audit of 60 oracle corrections (post hoc, reviews 3-5).

Reads the filled audit sheet (three questions: Q1 information, Q2 scope,
Q3 valid), the hidden mechanical flags and the audit items, and reports the
shares overall, by domain, by mechanical-flag stratum and for tau2
corrections that change state. `--flip` lists items whose Q3 answer is read
as "no" because the auditor's own written reason says the correction is not
valid (the sheet's raw answers are always reported too).

    python -m attrib.audit_human --sheet SHEET --key KEY --items ITEMS \
        [--flip A03,A08,...] --out results/attrib/audit_human.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

TAU2_WRITES = re.compile(r'"name":\s*"(modify_|cancel_|book_|update_|exchange_|return_|send_certificate)')


def corrections(items: Path) -> dict:
    """Item id -> text of the oracle's correction."""
    out, cur, buf, on = {}, None, [], False
    for line in items.read_text().splitlines():
        m = re.match(r"^## (A\d+) ", line)
        if m:
            if cur:
                out[cur] = "\n".join(buf)
            cur, buf, on = m.group(1), [], False
        elif line.startswith("**Oracle's correction**"):
            on = True
        elif on and line.startswith("**") and "correction" not in line:
            on = False
        elif on:
            buf.append(line)
    if cur:
        out[cur] = "\n".join(buf)
    return out


def shares(rows: list[dict]) -> dict:
    n = len(rows)
    c = {q: sum(r[q] for r in rows) for q in ("info", "scope", "valid", "clean")}
    return {"n": n, **c}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--flip", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    key = {k["item"]: k for k in json.load(open(a.key))}
    corr = corrections(Path(a.items))
    sheet = list(csv.DictReader(open(a.sheet, encoding="utf-8-sig")))
    flip = {x for x in a.flip.split(",") if x}
    res = {"flip": sorted(flip)}
    for reading, fl in (("as_filled", set()), ("reasons", flip)):
        rows = []
        for s in sheet:
            k = key[s["id"]]
            assert int(s["step"]) == k["k"] and s["domain"] == k["domain"], s["id"]
            info = s["Q1_information"] == "yes"
            scope = s["Q2_scope"] == "yes"
            valid = s["Q3_valid"] == "yes" and s["id"] not in fl
            rows.append({"id": s["id"], "domain": k["domain"], "kind": k["kind"],
                         "flagged": k["flags"]["unseen_id_or_expanded"],
                         "tau2_write": k["domain"] != "appworld" and bool(TAU2_WRITES.search(corr[s["id"]])),
                         "info": info, "scope": scope, "valid": valid,
                         "clean": valid and not info and not scope})
        r = {"all": shares(rows)}
        for g, f in (("domain", lambda x: x["domain"]), ("flagged", lambda x: str(x["flagged"])),
                     ("tau2_write", lambda x: str(x["tau2_write"]))):
            by = defaultdict(list)
            for x in rows:
                by[f(x)].append(x)
            r["by_" + g] = {kk: shares(v) for kk, v in sorted(by.items())}
        r["tau2_write_items"] = [x["id"] for x in rows if x["tau2_write"]]
        res[reading] = r
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k]["all"] if k != "flip" else res[k] for k in res}, indent=1))
    for k in ("as_filled", "reasons"):
        print(k, "flagged", res[k]["by_flagged"], "tau2_write", res[k]["by_tau2_write"])


if __name__ == "__main__":
    main()
