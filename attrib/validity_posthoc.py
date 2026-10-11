"""Post hoc analyses of the registered validity check (attrib/PREREGISTRATION_V.md).

Not registered; written after the scores of results/attrib/validity_analysis.json.
Uses the two label files as returned (attrib/validity_labels), under the two
bounds of the main analysis: strict (valid only if both accept) and lenient
(valid if either accepts).

  paired_change   change of (first write - m) from raw to valid score on the same
                  failures, (FW - m)_valid - (FW - m)_raw, cluster bootstrap over tasks
  validity_rate   first write's validity rate minus m's, same bootstrap
  minus_stl       valid score of each method minus the second-to-last step's
  by_kind         distinct items by domain and kind of the observed step, valid per auditor
  unapplied       corrections (samples) not applied at each method's steps, and all
                  scores without the failures that have one
  no_scope        valid scores with the scope criterion left out

  python -m attrib.validity_posthoc [--root R] --out results/attrib/validity_posthoc.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attrib import validity as V  # noqa: E402
from attrib.analyze import _boot, _mean  # noqa: E402

LABELS = ROOT / "attrib" / "validity_labels"
FILES = ("auditor_codex_lenient_A.csv", "labels_C.csv")


def _kind(rec: dict, k: int) -> str:
    from domains.tau2.common import READ_PREFIXES
    names = [b["name"] for b in rec["steps"][k]["assistant"] if b["type"] == "tool_use"]
    if not names:
        return "utterance"
    if any(n.startswith("transfer") for n in names):
        return "transfer"
    if any(not n.startswith(READ_PREFIXES) for n in names):
        return "write"
    return "read"


def _bound(rows: list[dict], rng: random.Random) -> dict:
    out = {"n": len(rows), "methods": {}, "paired_change": {}, "validity_rate": {}, "minus_stl": {},
           "first_write_minus": {}}
    for m in V.METHODS:
        out["methods"][m] = {k: _boot(rows, lambda xs, m=m, s=s: _mean([x[m + s] for x in xs]), rng)
                             for k, s in (("raw", ""), ("valid", "@valid"))}
    for m in V.METHODS[1:]:
        out["paired_change"][m] = _boot(
            [{"d": (r["first_write@valid"] - r[m + "@valid"]) - (r["first_write"] - r[m]), "cluster": r["cluster"]}
             for r in rows], lambda xs: _mean([x["d"] for x in xs]), rng)

        def rate(xs, m=m):
            a = sum(x["first_write@n_valid"] for x in xs) / max(1, sum(x["first_write@n_lab"] for x in xs))
            b = sum(x[m + "@n_valid"] for x in xs) / max(1, sum(x[m + "@n_lab"] for x in xs))
            return a - b
        out["validity_rate"][m] = _boot(rows, rate, rng)
        out["first_write_minus"][m] = {
            k: _boot([{"d": r["first_write" + s] - r[m + s], "cluster": r["cluster"]} for r in rows],
                     lambda xs: _mean([x["d"] for x in xs]), rng)
            for k, s in (("raw", ""), ("valid", "@valid"))}
    for m in V.METHODS:
        if m != "second_to_last":
            out["minus_stl"][m] = _boot(
                [{"d": r[m + "@valid"] - r["second_to_last@valid"], "cluster": r["cluster"]} for r in rows],
                lambda xs: _mean([x["d"] for x in xs]), rng)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(V.VROOT))
    ap.add_argument("--out")
    a = ap.parse_args()
    root = Path(a.root)
    key = json.loads((root / "audit" / "key.json").read_text())
    A, C = (V._labels(LABELS / f) for f in FILES)

    def expand(f):
        return {(v["fid"], v["k"], i): f(iid) for iid, v in key.items() for i in v["samples"]}
    bounds = {"strict": expand(lambda i: int(A[i]["valid"] and C[i]["valid"])),
              "lenient": expand(lambda i: int(A[i]["valid"] or C[i]["valid"])),
              "strict_no_scope": expand(lambda i: int(all(A[i][c] == "yes" and C[i][c] == "yes"
                                                          for c in V.CRITERIA if c != "scope")))}
    cs = V.corrections(root)
    fails = {d: json.loads((root / d / "failures.json").read_text()) for d in V.DOMAINS}
    pk = {d: V.picks(root, d) for d in V.DOMAINS}
    at = defaultdict(list)
    for c in cs:
        at[(c["fid"], c["k"])].append(c)

    unapplied = {}
    bad = set()
    for m in V.METHODS:
        tot = na = 0
        hit = set()
        for d in V.DOMAINS:
            for f in fails[d]:
                for c in at.get((f["fid"], pk[d][m].get(f["fid"])), []):
                    tot += 1
                    if not c["applied"]:
                        na += 1
                        hit.add(f["fid"])
        bad |= hit
        unapplied[m] = {"samples": tot, "not_applied": na, "failures": len(hit)}
    unapplied["all_union_steps"] = {"samples": len(cs), "not_applied": sum(not c["applied"] for c in cs)}

    trace = {f["fid"]: f["trace"] for d in V.DOMAINS for f in fails[d]}
    by_kind = defaultdict(lambda: {"items": 0, "valid_A": 0, "valid_C": 0})
    for iid, v in key.items():
        rec = json.loads(Path(trace[v["fid"]]).read_text())
        cell = by_kind[f"{v['domain']}|{_kind(rec, v['k'])}"]
        cell["items"] += 1
        cell["valid_A"] += A[iid]["valid"]
        cell["valid_C"] += C[iid]["valid"]

    patterns = {n: Counter("".join("y" if L[i][c] == "yes" else "n" for c in V.CRITERIA)
                           for i in key if not L[i]["valid"])
                for n, L in (("A", A), ("C", C))}

    rng = random.Random(0)
    res = {"note": __doc__.split("\n\n")[0] + " Auditor A = " + FILES[0] + ", C = " + FILES[1] + ".",
           "unapplied": unapplied, "by_kind": dict(sorted(by_kind.items())),
           "invalid_patterns_info_policy_confirmation_scope": {n: dict(p) for n, p in patterns.items()}}
    for b, lab in bounds.items():
        rows = V.scores(root, lab)
        res[b] = _bound(rows, rng)
        if b != "strict_no_scope":
            res[b + "_without_unapplied"] = _bound([r for r in rows if r["fid"] not in bad], rng)
    res["failures_with_unapplied"] = len(bad)
    print(json.dumps({k: res[k] for k in ("unapplied", "by_kind", "failures_with_unapplied")}, indent=1))
    for b in ("strict", "lenient"):
        print(b, "paired change", res[b]["paired_change"])
        print(b, "validity rate", res[b]["validity_rate"])
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
