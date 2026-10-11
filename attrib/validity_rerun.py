"""Addendum 1 of attrib/PREREGISTRATION_V.md: the validity check with the replays whose
restoration diverged rerun under the recording's check decisions
(TAU2_REPLAY_RECORDED_CHECKS=1 in domains/tau2/driver.py).

  python -m attrib.validity_rerun compare  [--root R] [--run1 R1]
  python -m attrib.validity_rerun addsheet [--root R]
  python -m attrib.validity_rerun analyze  [--root R] [--run1 R1] [--add-labels A.csv C.csv] [--out J]
  python -m attrib.validity_rerun posthoc  [--root R] --add-labels A.csv C.csv [--out J]

compare   what the rerun changed: each method's picks, corrections not applied, gains, and
          the outcome-changing corrections whose item (domain, failure, step, normalized
          action; validity.corrections item_key) the auditors labelled already or not; spend.
addsheet  the new items for the same two auditors, blinded and shuffled as `validity sheet`,
          with every tool result in full: R/audit/add/{items_add.md, labels_template_add.csv};
          the key R/audit/key_add.json is withheld from them.
analyze   the registered analysis (validity._summary) and the post hoc paired changes
          (validity_posthoc._bound) under the strict and lenient bounds; until the new items
          are labelled, each bound twice, with them counted invalid and counted valid. The
          first run (R1, with its labels) is reported beside it.
posthoc   on the rerun, with every item labelled: agreement, the registered secondary results
          (rescuable failures, domains, each auditor's labels) and attrib/validity_posthoc.py's
          analyses.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attrib import validity as V  # noqa: E402
from attrib import validity_posthoc as P  # noqa: E402

RUN1 = Path("/home/user/attrib_runs/validity_run1")


def _item_key(root: Path, d: str, f: str, k: int, i: int) -> str:
    v = json.loads((root / d / "gt" / "oracle" / f"{f}_k{k}_o{i}.json").read_text())
    return f"{d}|{f}|{k}|{json.dumps(v.get('action'), sort_keys=True, ensure_ascii=False)}"


def _keys(root: Path) -> dict:
    """item id -> (item_key, domain) for the first sheet and the added one."""
    out = {}
    for name in ("key.json", "key_add.json"):
        p = root / "audit" / name
        if p.exists():
            for iid, v in json.loads(p.read_text()).items():
                out[iid] = _item_key(root, v["domain"], v["fid"], v["k"], v["samples"][0])
    return out


def _changing(root: Path) -> list[dict]:
    return [c for c in V.corrections(root) if c["applied"] and abs(c["gain"]) > 1e-9]


def _labels(add: list[str] | None) -> tuple[dict, dict]:
    A, C = (V._labels(P.LABELS / f) for f in P.FILES)
    if add:
        a2, c2 = (V._labels(Path(p)) for p in add)
        A, C = {**A, **a2}, {**C, **c2}
    return A, C


def _valid(root: Path, A: dict, C: dict, bound: str, new: int) -> tuple[dict, int]:
    """{(fid, k, i): 0/1} for every outcome-changing correction; unlabelled items count `new`."""
    ik2id = {ik: iid for iid, ik in _keys(root).items()}
    out, unl = {}, set()
    for c in _changing(root):
        iid = ik2id.get(c["item_key"])
        if iid is None or iid not in A or iid not in C:
            out[(c["fid"], c["k"], c["i"])] = new
            unl.add(c["item_key"])
            continue
        a, b = A[iid], C[iid]
        out[(c["fid"], c["k"], c["i"])] = int(a["valid"] and b["valid"]) if bound == "strict" \
            else int(a["valid"] or b["valid"])
    return out, len(unl)


def _spend(root: Path, run1: Path) -> dict:
    from attrib.pilot_report import PRICES, replay_dollars
    rep = 0.0
    n = 0
    for d in V.DOMAINS:
        for rd in (root / d / "gt" / "replays", root / d / "search_maxgain" / "aligned" / "replays"):
            for p in rd.glob("*.json"):
                if p.name.startswith("refs_"):
                    continue
                x = json.loads(p.read_text())
                if not (x.get("replay") or {}).get("recorded_checks"):
                    continue
                rep += replay_dollars(x)
                n += 1
    llm = 0.0
    for d in V.DOMAINS:
        for rel in ("gt/usage.jsonl", "search_maxgain/aligned/usage.jsonl"):
            new, old = root / d / rel, run1 / d / rel
            lines = new.read_text().splitlines() if new.exists() else []
            skip = len(old.read_text().splitlines()) if old.exists() else 0
            for line in lines[skip:]:
                u = json.loads(line)
                pi, pc, po = PRICES.get(u.get("model"), PRICES["deepseek-v4-pro"])
                llm += ((u.get("in") or 0) * pi + (u.get("cache_read") or 0) * pc + (u.get("out") or 0) * po) / 1e6
    return {"new_replays": n, "replay_dollars": round(rep, 3), "oracle_and_search_dollars": round(llm, 3),
            "total_dollars": round(rep + llm, 3)}


def compare(args) -> dict:
    root, run1 = Path(args.root), Path(args.run1)
    out = {"picks_changed": {}, "unapplied": {}, "restoration": Counter()}
    for d in V.DOMAINS:
        p0, p1 = V.picks(run1, d), V.picks(root, d)
        out["picks_changed"][d] = {m: sorted((f, p0[m].get(f), p1[m].get(f)) for f in p1[m]
                                             if p0[m].get(f) != p1[m].get(f)) for m in V.METHODS}
    for name, r in (("run1", run1), ("rerun", root)):
        cs = V.corrections(r)
        out["unapplied"][name] = sum(not c["applied"] for c in cs)
        out[f"corrections_{name}"] = len(cs)
    for d in V.DOMAINS:
        for rd in (root / d / "gt" / "replays", root / d / "search_maxgain" / "aligned" / "replays"):
            for p in rd.glob("*.json"):
                if p.name.startswith("refs_"):
                    continue
                rp = json.loads(p.read_text()).get("replay") or {}
                kind = "null" if re.search(r"_n\d+$", p.stem) else "corrected"
                if rp.get("recorded_checks"):
                    out["restoration"][f"{d}|{rd.parent.name}|rerun|{kind}"] += 1
                    if kind == "corrected":
                        out["restoration"][f"{d}|{rd.parent.name}|rerun|corrected_applied"] += bool(rp.get("forced"))
                for dv in rp.get("diverged") or []:
                    out["restoration"][f"{d}|{rd.parent.name}|diverged|{kind}|{dv['kind']}"] += 1
    out["restoration"] = dict(sorted(out["restoration"].items()))
    c0 = {(c["fid"], c["k"], c["i"]): c for c in V.corrections(run1)}
    c1 = {(c["fid"], c["k"], c["i"]): c for c in V.corrections(root)}
    out["gain_changed"] = sum(1 for s in c1 if s in c0 and abs(c1[s]["gain"] - c0[s]["gain"]) > 1e-9)
    out["samples_new_steps"] = len(set(c1) - set(c0))
    labelled = set(_keys(root).values())
    ch1, ch0 = _changing(root), _changing(run1)
    ik1, ik0 = {c["item_key"] for c in ch1}, {c["item_key"] for c in ch0}
    out["outcome_changing"] = {
        "run1_samples": len(ch0), "run1_items": len(ik0),
        "rerun_samples": len(ch1), "rerun_items": len(ik1),
        "rerun_items_labelled": len(ik1 & labelled), "rerun_items_new": len(ik1 - labelled),
        "labelled_items_no_longer_changing": len(ik0 - ik1),
        "new_items_by_domain": dict(Counter(k.split("|")[0] for k in ik1 - labelled))}
    out["spend"] = _spend(root, run1)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def addsheet(args) -> None:
    import attrib.groundtruth as gtm
    from attrib.review4_runs import blind_prompt_tau2
    root = Path(args.root)
    adir = root / "audit" / "add"
    adir.mkdir(parents=True, exist_ok=True)
    labelled = set(_keys(root).values())
    if (root / "audit" / "key_add.json").exists():
        raise SystemExit("key_add.json exists; the added sheet is written once")
    items = {}
    for c in _changing(root):
        if c["item_key"] not in labelled:
            items.setdefault(c["item_key"], []).append(c)
    keys = sorted(items)
    random.Random(0).shuffle(keys)
    first = 1 + max(int(i[1:]) for i in json.loads((root / "audit" / "key.json").read_text()))
    fails = {d: {f["fid"]: f for f in json.loads((root / d / "failures.json").read_text())} for d in V.DOMAINS}
    md = ["# 补充审核条目", "",
          "重跑一部分重放后，新出现了一些改变结果的纠正，需要两位按同样的方法标注。格式、四个问题和标准都与第一批相同"
          "（criteria.md、policy_<domain>.md）。这一批的工具结果是完整的，没有截断。编号接着第一批往下排。", ""]
    key, cap = {}, gtm.RESULT_CAP
    try:
        gtm.RESULT_CAP = 10 ** 9
        for n, ik in enumerate(keys, first):
            c = items[ik][0]
            iid = f"V{n:03d}"
            rec = json.loads(Path(fails[c["domain"]][c["fid"]]["trace"]).read_text())
            os.environ["TAU2_DOMAIN"] = c["domain"].split("_", 1)[1]
            ctx = blind_prompt_tau2(rec, c["k"])
            md += [f"## {iid}（{'零售 retail' if c['domain'] == 'tau2_retail' else '航空 airline'}，第 {c['k']} 步）", "",
                   "```", ctx.rstrip(), "", f"=== CORRECTED ACTION AT STEP {c['k']} ===", V._action_text(c["action"]),
                   "```", ""]
            key[iid] = {"domain": c["domain"], "fid": c["fid"], "k": c["k"],
                        "samples": [x["i"] for x in items[ik]], "gains": [x["gain"] for x in items[ik]]}
    finally:
        gtm.RESULT_CAP = cap
    (adir / "items_add.md").write_text("\n".join(md))
    (root / "audit" / "key_add.json").write_text(json.dumps(key, indent=1))
    with (adir / "labels_template_add.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", *V.CRITERIA, "note"])
        for iid in key:
            w.writerow([iid, "", "", "", "", ""])
    print(f"{len(keys)} new items ({sum(items[k][0]['domain'] == 'tau2_retail' for k in keys)} retail), "
          f"{', '.join(list(key)[:1] + list(key)[-1:])}")


def rerun_key(root: Path) -> dict:
    """The rerun's outcome-changing items with their samples, under the ids they were labelled
    with (key.json, key_add.json); the form validity_posthoc.analyse takes."""
    ids = {ik: iid for iid, ik in _keys(root).items()}
    out = {}
    for c in _changing(root):
        v = out.setdefault(ids[c["item_key"]], {"domain": c["domain"], "fid": c["fid"], "k": c["k"],
                                                 "samples": [], "gains": []})
        v["samples"].append(c["i"])
        v["gains"].append(c["gain"])
    return dict(sorted(out.items()))


def posthoc(args) -> None:
    """The registered secondary results and attrib/validity_posthoc.py's analyses on the rerun,
    with both label files (first sheet and added sheet)."""
    root = Path(args.root)
    A, C = _labels(args.add_labels)
    key = rerun_key(root)
    res = {"note": "validity_posthoc.analyse on the rerun (Addendum 1); items = the rerun's outcome-changing "
                   "corrections under their labelled ids", "items": len(key),
           "items_new": sum(iid not in json.loads((root / "audit" / "key.json").read_text()) for iid in key),
           "valid_A": sum(A[i]["valid"] for i in key), "valid_C": sum(C[i]["valid"] for i in key),
           "agreement": {c: {"agree": sum(A[i][c] == C[i][c] for i in key),
                             "percent": round(sum(A[i][c] == C[i][c] for i in key) / len(key), 3),
                             "kappa": V._kappa([A[i][c] for i in key], [C[i][c] for i in key])}
                         for c in V.CRITERIA + ("valid",)}}
    lab = {"strict": lambda a, c: int(a["valid"] and c["valid"]), "lenient": lambda a, c: int(a["valid"] or c["valid"]),
           "auditor_A": lambda a, c: a["valid"], "auditor_C": lambda a, c: c["valid"]}
    res["secondary"] = {}
    for name, f in lab.items():
        valid = {(v["fid"], v["k"], i): f(A[iid], C[iid]) for iid, v in key.items() for i in v["samples"]}
        rows = V.scores(root, valid)
        rng = random.Random(0)
        res["secondary"][name] = {"all": V._summary(rows, rng),
                                  "rescuable_union": V._summary([r for r in rows if r["max_raw"] >= 0.5], rng),
                                  **{d: V._summary([r for r in rows if r["domain"] == d], rng) for d in V.DOMAINS}}
    res.update(P.analyse(root, key, A, C))
    print(json.dumps({k: res[k] for k in ("items", "items_new", "valid_A", "valid_C", "agreement", "by_kind",
                                          "invalid_only", "unapplied")}, indent=1))
    for b in ("strict", "lenient"):
        print(b, "validity_rate", res[b]["validity_rate"])
        print(b, "minus_stl", res[b]["minus_stl"])
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))


def _block(root: Path, valid: dict) -> dict:
    rows = V.scores(root, valid)
    return {"registered": V._summary(rows, random.Random(0)), "post_hoc": P._bound(rows, random.Random(0))}


def analyze(args) -> None:
    root, run1 = Path(args.root), Path(args.run1)
    A, C = _labels(args.add_labels)
    res = {"note": __doc__.split("\n\n")[0], "compare": compare(args), "rerun": {}, "run1": {}}
    for b in ("strict", "lenient"):
        news = (("labelled", 0),) if args.add_labels else (("new_invalid", 0), ("new_valid", 1))
        for tag, new in news:
            valid, unl = _valid(root, A, C, b, new)
            res["rerun"][f"{b}_{tag}"] = {"unlabelled_items": unl, **_block(root, valid)}
        valid, unl = _valid(run1, A, C, b, 0)
        assert unl == 0
        res["run1"][b] = _block(run1, valid)
    if args.add_labels:
        ka = json.loads((root / "audit" / "key_add.json").read_text())
        res["agreement_add"] = {c: {"percent": round(sum(A[i][c] == C[i][c] for i in ka) / max(1, len(ka)), 3),
                                    "kappa": V._kappa([A[i][c] for i in ka], [C[i][c] for i in ka])}
                                for c in V.CRITERIA + ("valid",)}
    for k, v in res["rerun"].items():
        reg, ph = v["registered"], v["post_hoc"]
        print(k, "unlabelled", v["unlabelled_items"])
        print("  valid", {m: [round(x, 3) for x in reg["methods"][m]["valid"]] for m in V.METHODS})
        print("  raw  ", {m: [round(x, 3) for x in reg["methods"][m]["raw"]] for m in V.METHODS})
        print("  FW-m valid", {m: [round(x, 3) for x in reg["first_write_minus"][m]["valid"]] for m in V.METHODS[1:]})
        print("  paired", {m: [round(x, 3) for x in ph["paired_change"][m]] for m in V.METHODS[1:]})
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("compare", "addsheet", "analyze", "posthoc"))
    ap.add_argument("--root", default=str(V.VROOT))
    ap.add_argument("--run1", default=str(RUN1))
    ap.add_argument("--add-labels", nargs=2)
    ap.add_argument("--out")
    a = ap.parse_args()
    {"compare": compare, "addsheet": addsheet, "analyze": analyze, "posthoc": posthoc}[a.cmd](a)


if __name__ == "__main__":
    main()
