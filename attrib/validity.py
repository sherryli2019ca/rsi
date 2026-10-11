"""Are rescuing corrections valid? Fresh tau2 failures, registered in
attrib/PREREGISTRATION_V.md.

  python -m attrib.validity sample  [--root R] [--runs /home/user/e1r2 /home/user/e1r3]
  python -m attrib.validity harness [--root R] [--bundle archive/rsi-local-branches.bundle]
  (methods: attrib.methods run --methods first_write,all_at_once_gain:pro,binary_search_gain:pro
            --failures R/<domain>/failures.json --out R/<domain>/methods;
   search:  attrib.search_maxgain run --variant aligned --main R)
  python -m attrib.validity oracle  --domain tau2_retail|tau2_airline [--root R] [--workers 8]
  python -m attrib.validity sheet   [--root R]
  python -m attrib.validity analyze --labels <adjudicated.csv> [--auditors a1.csv a2.csv] [--root R] [--out <json>]

sample   draws the fresh failures (R/<domain>/failures.json; failures_rescuable.json
         is the same list, which search_maxgain reads; no failure is selected
         by its gains).
harness  restores each failure's harness at its commit from the bundle into
         R/wt/<commit>/domains/<domain>/harness (git archive, no worktree).
oracle   the registered oracle of the main set at the union of the five methods'
         steps: K = 4 samples, 2 corrected replays per applicable correction,
         4 null replays per step with one; R/<domain>/gt/{oracle,replays,result.json}.
sheet    the blinded audit items: distinct corrections that change the outcome,
         shuffled; R/audit/{items.md, labels_template.csv, policy_<domain>.md}
         and the hidden key R/audit/key.json.
analyze  raw and valid scores per method over all failures, validity rates,
         paired differences with cluster-bootstrap intervals, agreement.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VROOT = Path("/home/user/attrib_runs/validity")
MAIN = Path("/home/user/attrib_runs/main")
DOMAINS = ("tau2_retail", "tau2_airline")
N = {"tau2_retail": 60, "tau2_airline": 20}
NS = {"e1r2": "r2", "e1r3": "r3"}
# evolve jobs of the eight repair-experiment states (attrib/PREREGISTRATION_B.md)
PB_JOBS = {("r2", "tau2_retail", "r0A"), ("r2", "tau2_retail", "r10B"), ("r2", "tau2_airline", "r4A"),
           ("r2", "tau2_airline", "r12A"), ("r3", "tau2_retail", "r2B"), ("r3", "tau2_retail", "r10A"),
           ("r3", "tau2_airline", "r2A"), ("r3", "tau2_airline", "r11A")}
METHODS = ("first_write", "second_to_last", "all_at_once_gain_pro", "binary_search_gain_pro", "aligned_search")
K, R_C, N_NULL = 4, 2, 4
CRITERIA = ("information", "policy", "confirmation", "scope")


# ------------------------------------------------------------------ sample --
def _failed(p: Path):
    r = json.loads(p.read_text())
    if r.get("reward", 0) < 1 and "harness_error" not in r and r.get("steps"):
        return r
    return None


def pool(domain: str, root: Path, exclude_tasks: set) -> list[dict]:
    from verify.state import rounds
    ns = NS[root.name]
    rdir = root / "runs" / "rrsi" / domain
    jobs = {}
    for r in rounds(rdir):
        jobs.setdefault(r.inc_job, r.inc_commit)
        for c in r.cands:
            if c.job:
                jobs.setdefault(c.job, c.commit)
    srcs = [("evolve", job, commit, rdir / "jobs" / job) for job, commit in jobs.items()
            if (ns, domain, job) not in PB_JOBS]
    hd = root / "runs" / "verify" / domain / "jobs" / "heldout"
    srcs += [("heldout", d.name, d.name, d) for d in sorted(hd.iterdir()) if d.is_dir()]
    rows = []
    for src, job, commit, d in srcs:
        for p in sorted(d.glob("s[0-9]*/*.json")):
            if p.stem in exclude_tasks or not p.stem.isdigit():
                continue
            r = _failed(p)
            if r is None:
                continue
            rows.append({"commit": commit, "source": src, "job": f"{ns}/{job}", "task_id": str(r["task_id"]),
                         "trial": p.parent.name, "trace": str(p), "n_steps": r["n_steps"],
                         "fid": f"{ns}_{commit}_{src[0]}_{r['task_id']}_{p.parent.name}"})
    return rows


def sample(args) -> None:
    from attrib.pool import sample as draw
    root = Path(args.root)
    for d in DOMAINS:
        excl = {str(f["task_id"]) for f in json.loads((MAIN / d / "failures.json").read_text())}
        rows = [r for run in args.runs for r in pool(d, Path(run), excl)]
        out = draw(rows, N[d], 0, set())
        for r in out:
            r["harness"] = str(root / "wt" / r["commit"] / "domains" / d / "harness")
        (root / d).mkdir(parents=True, exist_ok=True)
        for name in ("failures.json", "failures_rescuable.json"):
            (root / d / name).write_text(json.dumps(out, indent=1))
        print(f"{d}: pool {len(rows)} over {len({r['task_id'] for r in rows})} tasks; sampled {len(out)} over "
              f"{len({r['task_id'] for r in out})} tasks, {len({r['commit'] for r in out})} harnesses, "
              f"{sum(r['source'] == 'heldout' for r in out)} held-out")


def harness(args) -> None:
    root = Path(args.root)
    for d in DOMAINS:
        for f in json.loads((root / d / "failures.json").read_text()):
            dst = root / "wt" / f["commit"]
            if (dst / "domains" / d / "harness").is_dir():
                continue
            full = subprocess.run(["git", "-C", str(ROOT), "rev-parse", f"{f['commit']}^{{commit}}"],
                                  capture_output=True, text=True)
            if full.returncode:
                subprocess.run(["git", "-C", str(ROOT), "fetch", "-q", args.bundle, "refs/heads/*:refs/bundle/*"],
                               check=True)
            dst.mkdir(parents=True, exist_ok=True)
            tar = subprocess.run(["git", "-C", str(ROOT), "archive", f["commit"], "domains"], check=True,
                                 capture_output=True).stdout
            subprocess.run(["tar", "-x", "-C", str(dst)], input=tar, check=True)
    print("harnesses:", sorted(p.name for p in (root / "wt").iterdir()))


# ------------------------------------------------------------------ picks --
def picks(root: Path, d: str) -> dict:
    """{method: {fid: step or None}}."""
    fails = json.loads((root / d / "failures.json").read_text())
    out = {m: {} for m in METHODS}
    for f in fails:
        out["second_to_last"][f["fid"]] = max(f["n_steps"] - 2, 0)
        for m, sub in (("first_write", "first_write"), ("all_at_once_gain_pro", "all_at_once_gain_pro"),
                       ("binary_search_gain_pro", "binary_search_gain_pro")):
            p = root / d / "methods" / sub / f"{f['fid']}.json"
            k = json.loads(p.read_text()).get("step") if p.exists() else None
            out[m][f["fid"]] = k if isinstance(k, int) and 0 <= k < f["n_steps"] else None
    res = root / d / "search_maxgain" / "aligned" / "result.json"
    if res.exists():
        for x in json.loads(res.read_text())["failures"]:
            out["aligned_search"][x["fid"]] = x["maxgain"]
    return out


# ------------------------------------------------------------------ oracle --
def oracle(args) -> None:
    from attrib.groundtruth import ORACLE_SYSTEM
    from attrib.groundtruth3 import oracle_prompt
    from attrib.review4_runs import _samples, _setup, drive_all, run_oracle, step_profile
    root = Path(args.root)
    dom, m, ctx, force_of, drive = _setup(args.domain)
    rep = root / args.domain / "gt"
    odir, rdir = rep / "oracle", rep / "replays"
    rep.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("RRSI_USAGE_LOG", str(rep / "usage.jsonl"))
    fails = {f["fid"]: f for f in json.loads((root / args.domain / "failures.json").read_text())}
    pk = picks(root, args.domain)
    missing_picks = {mm: sum(fid not in pk[mm] for fid in fails) for mm in METHODS}
    if any(missing_picks.values()) and not args.allow_missing:
        raise SystemExit(f"methods not finished: {missing_picks}")
    steps = {f: sorted({pk[mm].get(f) for mm in METHODS} - {None}) for f in fails}
    fids = sorted(fails)[:args.limit] if args.limit else sorted(fails)
    recs = {f: json.loads(Path(fails[f]["trace"]).read_text()) for f in fids}
    run_oracle([(f, recs[f], k) for f in fids for k in steps[f]], odir, ORACLE_SYSTEM, oracle_prompt, ctx,
               force_of, args.workers)
    refs = []
    for f in fids:
        base = {"task_id": fails[f]["task_id"], "trace": fails[f]["trace"]}
        for k in steps[f]:
            sm = _samples(odir, f, k)
            mist = [i for i, x in enumerate(sm) if x.get("verdict") == "mistake"]
            for i in mist:
                refs += [(fails[f]["harness"], {**base, "start": k, "force": sm[i]["force"],
                                                "key": f"{f}_k{k}_o{i}_{j}"}) for j in range(R_C)]
            if mist:
                refs += [(fails[f]["harness"], {**base, "start": k, "key": f"{f}_k{k}_n{j}"})
                         for j in range(N_NULL)]
    missing = drive_all(refs, rdir, dom, drive, args.replay_workers or args.workers)
    res = {"K": K, "R_C": R_C, "missing_replays": missing,
           "failures": {f: [step_profile(f, k, odir, rdir, N_NULL) for k in steps[f]] for f in fids}}
    (rep / "result.json").write_text(json.dumps(res, indent=1))
    print(f"{args.domain}: {len(fids)} failures, {sum(len(steps[f]) for f in fids)} steps, "
          f"{len(refs)} replays requested, {missing} missing", flush=True)


# ------------------------------------------------------------------- sheet --
def _action_text(action) -> str:
    if not isinstance(action, dict):
        return str(action)
    lines = []
    if action.get("message"):
        lines.append(f"  agent says: {action['message']}")
    for c in action.get("tool_calls") or []:
        lines.append(f"  agent calls {c.get('name')}({json.dumps(c.get('arguments'), ensure_ascii=False)})")
    return "\n".join(lines) or json.dumps(action, ensure_ascii=False)


def corrections(root: Path) -> list[dict]:
    """Every sample with an applied correction at a union step: its gain and its item key."""
    out = []
    for d in DOMAINS:
        res = json.loads((root / d / "gt" / "result.json").read_text())["failures"]
        odir = root / d / "gt" / "oracle"
        for f, prof in res.items():
            for st in prof:
                for i, s in enumerate(st["samples"]):
                    if s.get("verdict") != "mistake":
                        continue
                    applied = s.get("n") == R_C and s.get("corrected") is not None and st["null"] is not None
                    v = json.loads((odir / f"{f}_k{st['k']}_o{i}.json").read_text())
                    act = json.dumps(v.get("action"), sort_keys=True, ensure_ascii=False)
                    out.append({"domain": d, "fid": f, "k": st["k"], "i": i, "applied": applied,
                                "gain": (s["corrected"] - st["null"]) if applied else 0.0,
                                "action": v.get("action"), "item_key": f"{d}|{f}|{st['k']}|{act}"})
    return out


def sheet(args) -> None:
    from attrib.review4_runs import blind_prompt_tau2
    root = Path(args.root)
    adir = root / "audit"
    adir.mkdir(parents=True, exist_ok=True)
    cs = [c for c in corrections(root) if c["applied"] and abs(c["gain"]) > 1e-9]
    items = {}
    for c in cs:
        items.setdefault(c["item_key"], []).append(c)
    keys = sorted(items)
    random.Random(0).shuffle(keys)
    fails = {d: {f["fid"]: f for f in json.loads((root / d / "failures.json").read_text())} for d in DOMAINS}
    md, key = ["# 纠正合法性审核条目", "",
               "每条给出：该领域的政策（见 policy_<domain>.md）、到第 k 步为止的对话、agent 在第 k 步原来的动作、"
               "纠正后的动作。请按 labels_template.csv 的四个问题逐条回答 yes / no，标准见 criteria.md。", ""], {}
    for n, ik in enumerate(keys, 1):
        c = items[ik][0]
        iid = f"V{n:03d}"
        rec = json.loads(Path(fails[c["domain"]][c["fid"]]["trace"]).read_text())
        os.environ["TAU2_DOMAIN"] = c["domain"].split("_", 1)[1]
        ctx = blind_prompt_tau2(rec, c["k"])
        md += [f"## {iid}（{'零售 retail' if c['domain'] == 'tau2_retail' else '航空 airline'}，第 {c['k']} 步）", "",
               "```", ctx.rstrip(), "", f"=== CORRECTED ACTION AT STEP {c['k']} ===", _action_text(c["action"]),
               "```", ""]
        key[iid] = {"domain": c["domain"], "fid": c["fid"], "k": c["k"],
                    "samples": [x["i"] for x in items[ik]], "gains": [x["gain"] for x in items[ik]]}
    (adir / "items.md").write_text("\n".join(md))
    (adir / "key.json").write_text(json.dumps(key, indent=1))
    with (adir / "labels_template.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", *CRITERIA, "note"])
        for iid in key:
            w.writerow([iid, "", "", "", "", ""])
    from attrib.groundtruth import _context
    for d in DOMAINS:
        os.environ["TAU2_DOMAIN"] = d.split("_", 1)[1]
        import importlib
        from agent_exp import tau2_env
        importlib.reload(tau2_env)
        (adir / f"policy_{d}.md").write_text(_context(tau2_env))
    print(f"{len(cs)} outcome-changing corrections, {len(keys)} distinct items "
          f"({sum(items[k][0]['domain'] == 'tau2_retail' for k in keys)} retail)")


# ----------------------------------------------------------------- analyze --
def _labels(path: Path) -> dict:
    out = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            ans = {c: (row.get(c) or "").strip().lower() for c in CRITERIA}
            if not all(a in ("yes", "no") for a in ans.values()):
                raise SystemExit(f"{path}: item {row['id']} has an answer other than yes/no: {ans}")
            out[row["id"]] = {**ans, "valid": int(all(a == "yes" for a in ans.values()))}
    return out


def _kappa(a: list, b: list):
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    cats = set(a) | set(b)
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else None


def scores(root: Path, valid: dict) -> list[dict]:
    """Per failure: raw and valid R at each method's step. valid: {(fid, k, i): 0/1}."""
    rows = []
    cs = corrections(root)
    by = defaultdict(list)
    for c in cs:
        by[(c["fid"], c["k"])].append(c)
    for d in DOMAINS:
        fails = json.loads((root / d / "failures.json").read_text())
        pk = picks(root, d)
        for f in fails:
            fid = f["fid"]
            r = {"domain": d, "cluster": f"{d}/{f['task_id']}", "fid": fid}
            for m in METHODS:
                k = pk[m].get(fid)
                raw = val = 0.0
                lab, nv = 0, 0
                if isinstance(k, int):
                    for c in by.get((fid, k), []):
                        if not c["applied"]:
                            continue
                        raw += c["gain"] / K
                        if abs(c["gain"]) > 1e-9:
                            v = valid[(fid, k, c["i"])]
                            val += v * c["gain"] / K
                            lab += 1
                            nv += v
                r[m], r[m + "@valid"], r[m + "@n_lab"], r[m + "@n_valid"] = raw, val, lab, nv
            r["max_raw"] = max((sum(c["gain"] for c in by.get((fid, k), []) if c["applied"]) / K
                                for k in {k for (ff, k) in by if ff == fid}), default=0.0)
            rows.append(r)
    return rows


def _summary(rows: list[dict], rng) -> dict:
    from attrib.analyze import _boot, _mean
    out = {"n": len(rows), "methods": {}, "first_write_minus": {}}
    for m in METHODS:
        lab = sum(r[m + "@n_lab"] for r in rows)
        out["methods"][m] = {"raw": _boot(rows, lambda xs, m=m: _mean([x[m] for x in xs]), rng),
                             "valid": _boot(rows, lambda xs, m=m: _mean([x[m + "@valid"] for x in xs]), rng),
                             "labelled_corrections": lab,
                             "validity_rate": round(sum(r[m + "@n_valid"] for r in rows) / lab, 3) if lab else None}
    for m in METHODS[1:]:
        out["first_write_minus"][m] = {
            kind: _boot([{"d": r["first_write" + s] - r[m + s], "cluster": r["cluster"]} for r in rows],
                        lambda xs: _mean([x["d"] for x in xs]), rng)
            for kind, s in (("valid", "@valid"), ("raw", ""))}
    return out


def analyze(args) -> None:
    root = Path(args.root)
    key = json.loads((root / "audit" / "key.json").read_text())
    rng = random.Random(0)

    def expand(lab: dict) -> dict:
        missing = set(key) - set(lab)
        if missing:
            raise SystemExit(f"{len(missing)} items unlabelled, e.g. {sorted(missing)[:5]}")
        return {(v["fid"], v["k"], i): lab[iid]["valid"] for iid, v in key.items() for i in v["samples"]}

    adj = _labels(Path(args.labels))
    rows = scores(root, expand(adj))
    res = {"all": _summary(rows, rng),
           "rescuable_union": _summary([r for r in rows if r["max_raw"] >= 0.5], rng),
           **{d: _summary([r for r in rows if r["domain"] == d], rng) for d in DOMAINS},
           "items": len(key), "items_valid_adjudicated": sum(adj[i]["valid"] for i in key)}
    if args.auditors:
        labs = [_labels(Path(p)) for p in args.auditors]
        res["agreement"] = {}
        for c in CRITERIA + ("valid",):
            a = [labs[0][i][c] for i in key]
            b = [labs[1][i][c] for i in key]
            res["agreement"][c] = {"percent": round(sum(x == y for x, y in zip(a, b)) / len(a), 3),
                                   "kappa": _kappa(a, b)}
        for n, lab in enumerate(labs, 1):
            res[f"auditor{n}"] = {"items_valid": sum(lab[i]["valid"] for i in key),
                                  "all": _summary(scores(root, expand(lab)), rng)}
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("sample", "harness", "oracle", "sheet", "analyze"))
    ap.add_argument("--root", default=str(VROOT))
    ap.add_argument("--runs", nargs="+", default=["/home/user/e1r2", "/home/user/e1r3"])
    ap.add_argument("--bundle", default="/mnt/project-files/archive/rsi-local-branches.bundle")
    ap.add_argument("--domain", choices=DOMAINS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--replay-workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--allow-missing", action="store_true", help="oracle: run although some picks are missing")
    ap.add_argument("--labels")
    ap.add_argument("--auditors", nargs=2)
    ap.add_argument("--out")
    a = ap.parse_args()
    {"sample": sample, "harness": harness, "oracle": oracle, "sheet": sheet, "analyze": analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
