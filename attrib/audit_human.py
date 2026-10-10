"""Human audit of 60 oracle corrections (post hoc, reviews 3-5).

Two auditors labelled the same 60 corrections independently (three questions:
Q1 information the agent had not seen, Q2 more than one careful step, Q3 valid
given what the agent could see and the policy). Auditor 1 first filed labels
that contradicted their own written reasons on 15 rows and then made every
label follow its reason (auditor1_filed, auditor1); auditor 2 never saw
auditor 1's sheet. Reports each sheet's shares overall, by domain, by
mechanical-flag stratum and for tau2 corrections that change state, and the
agreement (Cohen's kappa) of the first two sheets. The released labels omit
the auditors' free-text notes.

    python -m attrib.audit_human --sheets auditor1=attrib/audit/auditor1.csv \
        auditor2=attrib/audit/auditor2.csv auditor1_filed=attrib/audit/auditor1_filed.csv \
        --key KEY --items ITEMS --out results/attrib/audit_human.json
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


def _kappa(x: list, y: list) -> tuple[float, float]:
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    pe = sum((x.count(c) / n) * (y.count(c) / n) for c in set(x) | set(y))
    return round(po, 4), round((po - pe) / (1 - pe), 4) if pe < 1 else None


def label_rows(sheet: list[dict], key: dict, corr: dict) -> dict:
    rows = {}
    for s in sheet:
        sid = s.get("id") or s["item_id"]
        k = key[sid]
        if "step" in s:
            assert int(s["step"]) == k["k"] and s["domain"] == k["domain"], sid
        info, scope, valid = (s[q] == "yes" for q in ("Q1_information", "Q2_scope", "Q3_valid"))
        rows[sid] = {"id": sid, "domain": k["domain"], "kind": k["kind"],
                     "flagged": k["flags"]["unseen_id_or_expanded"],
                     "tau2_write": k["domain"] != "appworld" and bool(TAU2_WRITES.search(corr[sid])),
                     "info": info, "scope": scope, "valid": valid,
                     "clean": valid and not info and not scope}
    assert len(rows) == 60
    return rows


def summary(rows: list[dict]) -> dict:
    r = {"all": shares(rows)}
    for g, f in (("domain", lambda x: x["domain"]), ("flagged", lambda x: str(x["flagged"])),
                 ("tau2_write", lambda x: str(x["tau2_write"]))):
        by = defaultdict(list)
        for x in rows:
            by[f(x)].append(x)
        r["by_" + g] = {kk: shares(v) for kk, v in sorted(by.items())}
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", nargs="+", required=True, help="name=path; the first two are compared")
    ap.add_argument("--key", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    key = {k["item"]: k for k in json.load(open(a.key))}
    corr = corrections(Path(a.items))
    labs = {}
    for spec in a.sheets:
        name, path = spec.split("=", 1)
        labs[name] = label_rows(list(csv.DictReader(open(path, encoding="utf-8-sig"))), key, corr)
    res = {name: summary(list(r.values())) for name, r in labs.items()}
    names = list(labs)
    res["tau2_write_items"] = sorted(i for i, x in labs[names[0]].items() if x["tau2_write"])
    if len(names) >= 2:
        x, y = labs[names[0]], labs[names[1]]
        ids = sorted(x)
        agree = {}
        for q in ("info", "scope", "valid"):
            po, kap = _kappa([x[i][q] for i in ids], [y[i][q] for i in ids])
            agree[q] = {"agreement": po, "kappa": kap, "disagree": [i for i in ids if x[i][q] != y[i][q]]}
        both = [dict(x[i], valid=x[i]["valid"] and y[i]["valid"]) for i in ids]
        either = [dict(x[i], valid=x[i]["valid"] or y[i]["valid"]) for i in ids]
        res["agreement"] = {"pair": names[:2], **agree}
        res["valid_both"] = summary(both)
        res["valid_either"] = summary(either)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for name in names:
        print(name, res[name]["all"], "unflagged", res[name]["by_flagged"]["False"],
              "tau2_write", res[name]["by_tau2_write"]["True"])
    if "agreement" in res:
        print({q: (v["agreement"], v["kappa"]) for q, v in res["agreement"].items() if q != "pair"})
        print("both", res["valid_both"]["all"]["valid"], "either", res["valid_either"]["all"]["valid"],
              "tau2 write both", res["valid_both"]["by_tau2_write"]["True"]["valid"],
              "either", res["valid_either"]["by_tau2_write"]["True"]["valid"])


if __name__ == "__main__":
    main()
