"""Read an RRSI run: settled rounds, their incumbent and candidates."""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from rrsi import gitops as G                # noqa: E402
from rrsi.evaluate import EvalResult        # noqa: E402

READ_PREFIXES = ("get_", "find_", "list_", "search_", "calculate", "think")


@dataclass
class Cand:
    t: int
    variant: str
    commit: str | None
    job: str | None                 # RRSI's full evolve evaluation (runs/<d>/jobs/<job>)
    gate_failure: str | None
    decision: dict = field(default_factory=dict)
    edits: list = field(default_factory=list)

    @property
    def measured(self) -> bool:
        return bool(self.commit and self.job and not self.gate_failure)


@dataclass
class Round:
    t: int
    inc_commit: str
    inc_job: str
    cands: list
    accepted: str | None            # variant accepted by RRSI's own rule, or None


def rounds(run_dir: Path) -> list[Round]:
    """Settled rounds (decisions.json written), oldest first."""
    run_dir = Path(run_dir)
    fr = json.loads((run_dir / "frontier.json").read_text())
    traj = {x["t"]: x for x in fr["trajectory"]}
    out = []
    for rdir in sorted(run_dir.glob("r[0-9]*"), key=lambda p: int(p.name[1:])):
        t = int(rdir.name[1:])
        dp = rdir / "decisions.json"
        if not dp.exists() or t not in traj or t + 1 not in traj:
            continue
        decs = {d["variant"]: d for d in json.loads(dp.read_text())}
        cands = []
        for vdir in sorted(p for p in rdir.iterdir() if p.is_dir() and p.name.isalpha()
                           and p.name.isupper()):
            prep = json.loads((vdir / "prep.json").read_text()) if (vdir / "prep.json").exists() else {}
            ok = (vdir / "eval.json").exists()
            cands.append(Cand(t, vdir.name, prep.get("commit"), f"r{t}{vdir.name}" if ok else None,
                              prep.get("gate_failure") or (None if ok else "not_measured"),
                              decs.get(vdir.name, {}), prep.get("edits") or []))
        inc, nxt = traj[t], traj[t + 1]
        acc = None
        if nxt["commit"] != inc["commit"]:
            acc = next((c.variant for c in cands if c.commit and nxt["commit"].startswith(c.commit[:7])),
                       None)
        out.append(Round(t, inc["commit"], inc["job"], cands, acc))
    return out


def evaluation(run_dir: Path, job: str) -> EvalResult:
    return EvalResult.load(Path(run_dir) / "jobs" / job / "eval.json")


def trial_path(run_dir: Path, job: str, task_id: str, s: int) -> Path:
    return Path(run_dir) / "jobs" / job / f"s{s}" / f"{task_id}.json"


def worktree(repo: Path, wt_root: Path, commit: str) -> Path:
    """Detached worktree of `commit` (reused when it exists)."""
    path = Path(wt_root) / commit[:12]
    if (path / ".git").exists():
        return path
    G.worktree_remove(repo, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    G.git(repo, "worktree", "add", "--detach", str(path), commit, check=True)
    return path


def digest_step(path: Path) -> int | None:
    """Earliest step a failure digest cites as evidence."""
    try:
        d = json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001
        return None
    steps = [int(m) for e in d.get("evidence") or []
             for m in re.findall(r"step\s+(\d+)", str(e.get("where", "")))]
    return min(steps) if steps else None


def first_bad_write(rec: dict) -> int | None:
    """Step of the first state-changing call that is not a gold action (tau2
    records; records of other domains, whose steps hold no tool-use blocks,
    give None)."""
    gold = {(g["name"], json.dumps(g.get("arguments"), sort_keys=True))
            for g in rec.get("gold_actions") or []}
    for st in rec.get("steps") or []:
        blocks = st.get("assistant")
        for b in blocks if isinstance(blocks, list) else []:
            if isinstance(b, dict) and b.get("type") == "tool_use" and not b["name"].startswith(READ_PREFIXES) and \
                    (b["name"], json.dumps(b.get("input"), sort_keys=True)) not in gold:
                return int(st["index"])
    return None
