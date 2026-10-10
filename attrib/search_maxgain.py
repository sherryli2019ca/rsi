"""Counterfactual search with a selection rule matched to the metric (post hoc,
not registered).

The registered search (attrib.search) tests suspects in order and stops at the
first one whose corrected replays beat null replays by FLIP successes; R_k
instead rewards the step with the largest gain over null. Here every suspect
with an applicable action gets N_REPLAYS corrected and N_REPLAYS null replays
(at most SUSPECTS x 8 = 40, the registered budget), and the answer is the
suspect with the largest estimated gain (corrected minus null successes; ties
go to the earlier suspect). Run on the rescuable failures only.

  python -m attrib.search_maxgain run --domain tau2_retail|tau2_airline|appworld \
      --variant search|search_informed|aligned [--main /home/user/attrib_runs/main] \
      [--workers 8] [--replay-workers N] [--limit N]

variant search, search_informed
        the registered and the informed run's suspects and replays (linked),
        with the suspects they never tested completed.
variant aligned
        a new suspect call that sees what the judges see (as search_informed)
        and asks for the steps where a correction most increases success over
        letting the agent act again (the judges' post hoc gain definition,
        attrib.methods.GAIN_DEF); suspects and replays as above.

Output <main>/<domain>/search_maxgain/<variant>/{suspects,replays,result.json,usage.jsonl}.
result.json, per failure: each suspect's step, corrected and null successes and
estimated gain; "maxgain" (the answer here) and "first_flip" (the registered
stopping rule applied to the same replays, first suspect if none flips).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attrib.groundtruth import FLIP, N_REPLAYS  # noqa: E402

MAIN = Path("/home/user/attrib_runs/main")

ASK = ("Name the steps where the agent most likely went wrong, most suspect first (at most %d).")
ASK_GAIN = ("Name the steps where a correction would make the biggest difference, best first (at most "
            "%d): the steps at which acting correctly, and then continuing normally, would most increase "
            "the chance that the task succeeds compared with letting the agent act again from that step "
            "without any correction. A step from which the task would often have succeeded anyway, "
            "without a correction, is a weak choice even if a correction there would succeed.")


def aligned_system(dom: str) -> str:
    from attrib import search
    if dom == "appworld":
        from attrib import aw
        base = aw.search_system(search.SUSPECTS, "\n".join(f"- {k}: {v}" for k, v in aw.COMPONENTS.items()),
                                informed=True)
    else:
        base = search.SYSTEM_INFORMED
    a = ASK % search.SUSPECTS
    assert base.count(a) == 1
    return base.replace(a, ASK_GAIN % search.SUSPECTS)


def aligned_suspects(fails: list, sdir: Path, dom: str, m, workers: int) -> None:
    """search.suspects with the informed prompt and the gain question."""
    from attrib import search
    system = aligned_system(dom)
    if dom == "appworld":
        from attrib import aw
        orig = aw.search_system
        aw.search_system = lambda n, c, informed=False: system  # noqa: E731
        try:
            search.suspects(fails, sdir, None, workers, informed=True)
        finally:
            aw.search_system = orig
    else:
        orig = search.SYSTEM_INFORMED
        search.SYSTEM_INFORMED = system
        try:
            search.suspects(fails, sdir, m, workers, informed=True)
        finally:
            search.SYSTEM_INFORMED = orig


def _succ(rdir: Path, key: str, forced: bool):
    p = rdir / f"{key}.json"
    if not p.exists():
        return None
    x = json.loads(p.read_text())
    if forced and not (x.get("replay") or {}).get("forced"):
        return 0
    return int(x.get("reward", 0) >= 1)


def need(f: dict, sus: list, rdir: Path) -> list:
    base = {"task_id": f["task_id"], "trace": f["trace"]}
    out = []
    for i, s in enumerate(sus):
        if s["force"] is None:
            continue
        for j in range(N_REPLAYS):
            if not (rdir / f"{f['fid']}_s{i}_c{j}.json").exists():
                out.append({**base, "start": s["step"], "force": s["force"], "key": f"{f['fid']}_s{i}_c{j}"})
            if not (rdir / f"{f['fid']}_s{i}_n{j}.json").exists():
                out.append({**base, "start": s["step"], "key": f"{f['fid']}_s{i}_n{j}"})
    return out


def record(f: dict, sus: list, rdir: Path) -> dict:
    rows = []
    for i, s in enumerate(sus):
        if s["force"] is None:
            rows.append({"i": i, "step": s["step"], "valid": False})
            continue
        c = [_succ(rdir, f"{f['fid']}_s{i}_c{j}", True) for j in range(N_REPLAYS)]
        z = [_succ(rdir, f"{f['fid']}_s{i}_n{j}", False) for j in range(N_REPLAYS)]
        miss = sum(v is None for v in c + z)
        cs, zs = sum(v or 0 for v in c), sum(v or 0 for v in z)
        rows.append({"i": i, "step": s["step"], "valid": True, "corrected": cs, "null": zs, "missing": miss,
                     "gain": (cs - zs) / N_REPLAYS})
    valid = [r for r in rows if r["valid"]]
    best = max(valid, key=lambda r: (r["gain"], -r["i"])) if valid else None
    flip = next((r for r in valid if r["corrected"] >= FLIP and r["corrected"] - r["null"] >= FLIP), None)
    top = sus[0]["step"] if sus else None
    return {"fid": f["fid"], "suspects": rows,
            "maxgain": best["step"] if best else top,
            "first_flip": flip["step"] if flip else top,
            "replays": sum(2 * N_REPLAYS for _ in valid)}


def run(dom_key: str, variant: str, main: Path, workers: int, replay_workers: int | None, limit: int | None):
    if dom_key == "appworld":
        dom, m = "appworld", None
        from attrib.aw import drive
    else:
        dom = dom_key.split("_", 1)[1]
        os.environ["TAU2_DOMAIN"] = dom
        from agent_exp import tau2_env as m
        from attrib.groundtruth import _drive as drive
    from attrib.review4_runs import drive_all
    fails = json.loads((main / dom_key / "failures_rescuable.json").read_text())
    if limit:
        fails = fails[:limit]
    out = main / dom_key / "search_maxgain" / variant
    sdir, rdir = out / "suspects", out / "replays"
    sdir.mkdir(parents=True, exist_ok=True)
    rdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("RRSI_USAGE_LOG", str(out / "usage.jsonl"))
    if variant == "aligned":
        aligned_suspects(fails, sdir, dom, m, workers)
    else:
        src = main / dom_key / variant
        for f in fails:
            p = sdir / f"{f['fid']}.json"
            if not p.exists():
                p.symlink_to(src / "suspects" / f"{f['fid']}.json")
        for p in (src / "replays").glob("*.json"):
            if p.name.startswith("refs_"):
                continue
            q = rdir / p.name
            if not q.exists() and not q.is_symlink():
                q.symlink_to(p.resolve())
    sus = {f["fid"]: json.loads((sdir / f"{f['fid']}.json").read_text())["suspects"] for f in fails}
    refs = [(f["harness"], r) for f in fails for r in need(f, sus[f["fid"]], rdir)]
    print(f"{dom_key}/{variant}: {len(fails)} failures, {len(refs)} replays to run", flush=True)
    missing = drive_all(refs, rdir, dom, drive, replay_workers or workers) if refs else 0
    if dom == "appworld":
        shutil.rmtree(rdir / "_appworld", ignore_errors=True)
    res = [record(f, sus[f["fid"]], rdir) for f in fails]
    (out / "result.json").write_text(json.dumps({"variant": variant, "missing_replays": missing,
                                                 "failures": res}, indent=1))
    print(f"{dom_key}/{variant}: done, {missing} replays missing", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run",))
    ap.add_argument("--domain", required=True, choices=("tau2_retail", "tau2_airline", "appworld"))
    ap.add_argument("--variant", required=True, choices=("search", "search_informed", "aligned"))
    ap.add_argument("--main", default=str(MAIN))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--replay-workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(a.domain, a.variant, Path(a.main), a.workers, a.replay_workers, a.limit)


if __name__ == "__main__":
    main()
