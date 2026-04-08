"""
org_decision.py — Pairwise AHP decision engine for org-as-code
Part of the SYNTRIAD org-as-code ecosystem.

Adds a 'decision' process type: multi-participant pairwise comparison (AHP)
that produces auditable YAML artifacts and integrates with the hash-chain
audit trail in registry/artifacts.jsonl.

Usage:
    python org_decision.py --help
    python org_decision.py session --id DEC-001 --options "Option A" "Option B" "Option C"
    python org_decision.py vote --id DEC-001 --participant alice
    python org_decision.py aggregate --id DEC-001
    python org_decision.py show --id DEC-001

Requirements: numpy (already in most Python envs), pyyaml (org-as-code dep)

All artifacts land in processes/{ID}/ and are logged to registry/artifacts.jsonl
with SHA-256 hash-chaining — identical to the rest of org-as-code.
"""

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

# ── Repo root resolution ──────────────────────────────────────────────────────
REPO_ROOT = Path(os.environ.get("ORG_REPO_PATH", Path(__file__).parent))
PROCESSES_DIR = REPO_ROOT / "processes"
REGISTRY_DIR = REPO_ROOT / "registry"
ARTIFACTS_JSONL = REGISTRY_DIR / "artifacts.jsonl"
AGENTS_YAML = REGISTRY_DIR / "agents.yaml"

# ── AHP Saaty scale (1–9) ─────────────────────────────────────────────────────
SAATY_LABELS = {
    1: "Equal importance",
    2: "Weak",
    3: "Moderate importance",
    4: "Moderate plus",
    5: "Strong importance",
    6: "Strong plus",
    7: "Very strong importance",
    8: "Very, very strong",
    9: "Extreme importance",
}

# Random Consistency Index (Saaty, n=1..10)
RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12,
      6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


# ── Hash-chain helpers (mirrors org_mcp_server.py) ───────────────────────────

def _get_chain_tip() -> str:
    """Return the entry_hash of the last line in artifacts.jsonl, or 'GENESIS'."""
    if not ARTIFACTS_JSONL.exists() or ARTIFACTS_JSONL.stat().st_size == 0:
        return "GENESIS"
    block = 8192
    with open(ARTIFACTS_JSONL, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        pos = max(0, size - block)
        f.seek(pos)
        tail = f.read().decode("utf-8", errors="replace")
    lines = [l.strip() for l in tail.splitlines() if l.strip()]
    if not lines:
        return "GENESIS"
    try:
        return json.loads(lines[-1]).get("entry_hash", "GENESIS")
    except json.JSONDecodeError:
        return "GENESIS"


def _append_artifact(entry: dict) -> None:
    """Append a hash-chained entry to artifacts.jsonl."""
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    prev_hash = _get_chain_tip()
    entry["prev_hash"] = prev_hash
    canonical = json.dumps(entry, sort_keys=True, ensure_ascii=False)
    entry["entry_hash"] = hashlib.sha256(
        (prev_hash + canonical).encode("utf-8")
    ).hexdigest()
    with open(ARTIFACTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _log(process_id: str, agent: str, action: str, description: str, extra: dict = None) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "decision_artifact",
        "process_id": process_id,
        "agent": agent,
        "action": action,
        "description": description,
    }
    if extra:
        entry.update(extra)
    _append_artifact(entry)


# ── YAML helpers ──────────────────────────────────────────────────────────────

def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)
    tmp.replace(path)


# ── AHP core ─────────────────────────────────────────────────────────────────

def _build_matrix(options: list[str], comparisons: list[dict]) -> np.ndarray:
    """Reconstruct the pairwise matrix from stored comparisons."""
    n = len(options)
    idx = {o: i for i, o in enumerate(options)}
    matrix = np.ones((n, n))
    for c in comparisons:
        i, j = idx[c["option_a"]], idx[c["option_b"]]
        w = c["weight"]
        if c["preferred"] == c["option_a"]:
            matrix[i][j] = w
            matrix[j][i] = 1.0 / w
        else:
            matrix[j][i] = w
            matrix[i][j] = 1.0 / w
    return matrix


def _ahp_scores(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Returns (priority_vector, consistency_ratio).
    Geometric mean method — robust for n > 4.
    """
    n = matrix.shape[0]
    geo = np.exp(np.mean(np.log(matrix), axis=1))
    priorities = geo / geo.sum()

    # Consistency: λ_max → CI → CR
    weighted = matrix @ priorities
    lambda_max = float(np.mean(weighted / priorities))
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = RI.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0.0
    return priorities, cr


def _aggregate(all_votes: list[dict], options: list[str]) -> tuple[np.ndarray, float]:
    """
    Weighted geometric mean aggregation (AIJ — Aggregation of Individual Judgments).
    Each voter has equal weight; NedXIS voters get 1.5× weight (configurable).
    Returns (aggregate_priority_vector, mean_cr).
    """
    WEIGHT_MAP = {"nedxis": 1.5, "deelnemer": 1.0, "default": 1.0}
    n = len(options)
    log_sum = np.zeros((n, n))
    weight_sum = 0.0
    crs = []

    for vote in all_votes:
        matrix = _build_matrix(options, vote["comparisons"])
        _, cr = _ahp_scores(matrix)
        crs.append(cr)
        w = WEIGHT_MAP.get(vote.get("role", "default"), 1.0)
        log_sum += w * np.log(matrix)
        weight_sum += w

    agg_matrix = np.exp(log_sum / weight_sum)
    priorities, _ = _ahp_scores(agg_matrix)
    return priorities, float(np.mean(crs))


# ── CLI commands ──────────────────────────────────────────────────────────────

def cmd_session(args):
    """Create a new decision session — writes P.0_decision_session.yaml."""
    process_dir = PROCESSES_DIR / args.id
    process_dir.mkdir(parents=True, exist_ok=True)

    session_file = process_dir / "P.0_decision_session.yaml"
    if session_file.exists() and not args.force:
        print(f"Session {args.id} already exists. Use --force to overwrite.")
        sys.exit(1)

    options = args.options
    if len(options) < 2:
        print("Need at least 2 options.")
        sys.exit(1)

    data = {
        "process_id": args.id,
        "title": args.title or f"Decision: {args.id}",
        "description": args.description or "",
        "options": options,
        "created_by": args.agent,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "open",
        "votes": [],
    }
    _write_yaml(session_file, data)

    # Per-process state
    state_file = process_dir / "state.yaml"
    _write_yaml(state_file, {
        "process_id": args.id,
        "type": "decision",
        "state": "P_READY",
        "assigned_agent": args.agent,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })

    _log(args.id, args.agent, "P.0_decision_session",
         f"Decision session created with {len(options)} options: {', '.join(options)}")

    print(f"\n✅ Session {args.id} created.")
    print(f"   Options: {', '.join(options)}")
    print(f"   Next: python org_decision.py vote --id {args.id} --participant <name>")


def cmd_vote(args):
    """Interactive pairwise voting — writes V.N_vote_<participant>.yaml."""
    process_dir = PROCESSES_DIR / args.id
    session_file = process_dir / "P.0_decision_session.yaml"

    if not session_file.exists():
        print(f"No session found for {args.id}. Run 'session' first.")
        sys.exit(1)

    session = _read_yaml(session_file)
    options = session["options"]
    n = len(options)
    pairs = list(combinations(range(n), 2))

    # Check duplicate vote
    existing_votes = [v for v in session.get("votes", [])
                      if v["participant"] == args.participant]
    if existing_votes and not args.force:
        print(f"⚠️  {args.participant} already voted. Use --force to revote.")
        sys.exit(1)

    role = args.role or "deelnemer"
    print(f"\n=== Pairwise Comparison — {args.id} ===")
    print(f"Participant: {args.participant} ({role})")
    print(f"Options: {', '.join(options)}\n")
    print("For each pair: choose which option is more important and by how much (1–9).")
    print("Scale: 1=equal, 3=moderate, 5=strong, 7=very strong, 9=extreme\n")

    comparisons = []
    for i, j in pairs:
        a, b = options[i], options[j]
        print(f"  A: {a}")
        print(f"  B: {b}")

        while True:
            choice = input("  Which is more important? (A/B): ").strip().upper()
            if choice in ("A", "B"):
                break
            print("  Enter A or B.")

        while True:
            raw = input("  How much more important? (1–9): ").strip()
            if raw.isdigit() and 1 <= int(raw) <= 9:
                weight = int(raw)
                break
            print("  Enter a number between 1 and 9.")

        preferred = a if choice == "A" else b
        comparisons.append({
            "option_a": a,
            "option_b": b,
            "preferred": preferred,
            "weight": weight,
            "label": SAATY_LABELS[weight],
        })
        print(f"  → {preferred} ({weight}×)\n")

    # Compute individual scores + CR
    matrix = _build_matrix(options, comparisons)
    priorities, cr = _ahp_scores(matrix)

    cr_status = "✅ consistent" if cr < 0.10 else ("⚠️  marginal" if cr < 0.20 else "❌ inconsistent")
    print(f"\nConsistency Ratio: {cr:.3f} {cr_status}")
    if cr >= 0.20:
        print("CR ≥ 0.20 — consider revisiting your comparisons.")

    # Individual result
    ranked = sorted(zip(options, priorities), key=lambda x: x[1], reverse=True)
    print("\nYour ranking:")
    for rank, (opt, score) in enumerate(ranked, 1):
        print(f"  {rank}. {opt}  ({score:.3f})")

    vote_data = {
        "participant": args.participant,
        "role": role,
        "voted_at": datetime.now(timezone.utc).isoformat(),
        "comparisons": comparisons,
        "individual_priorities": {o: float(p) for o, p in zip(options, priorities)},
        "consistency_ratio": round(cr, 4),
        "cr_status": cr_status.split()[0],  # ✅ / ⚠️ / ❌
    }

    # Write individual vote artifact
    vote_count = len([v for v in session.get("votes", [])
                      if v["participant"] != args.participant])
    vote_file = process_dir / f"V.{vote_count}_vote_{args.participant}.yaml"
    _write_yaml(vote_file, vote_data)

    # Update session
    session["votes"] = [v for v in session.get("votes", [])
                        if v["participant"] != args.participant]
    session["votes"].append(vote_data)
    _write_yaml(session_file, session)

    _log(args.id, args.participant, f"V.{vote_count}_vote",
         f"Pairwise vote submitted. CR={cr:.3f} ({cr_status.split()[0]})",
         extra={"consistency_ratio": round(cr, 4), "role": role})

    print(f"\n✅ Vote recorded as V.{vote_count}_vote_{args.participant}.yaml")


def cmd_aggregate(args):
    """Aggregate all votes — writes V.final_consensus.yaml."""
    process_dir = PROCESSES_DIR / args.id
    session_file = process_dir / "P.0_decision_session.yaml"

    if not session_file.exists():
        print(f"No session found for {args.id}.")
        sys.exit(1)

    session = _read_yaml(session_file)
    options = session["options"]
    votes = session.get("votes", [])

    if not votes:
        print("No votes yet. Nothing to aggregate.")
        sys.exit(1)

    # Flag inconsistent votes
    inconsistent = [v["participant"] for v in votes if v["consistency_ratio"] >= 0.20]
    if inconsistent:
        print(f"⚠️  Inconsistent votes (CR ≥ 0.20): {', '.join(inconsistent)}")
        if not args.include_inconsistent:
            votes = [v for v in votes if v["consistency_ratio"] < 0.20]
            print(f"   Excluded from aggregation. Use --include-inconsistent to override.")

    if not votes:
        print("No consistent votes to aggregate.")
        sys.exit(1)

    priorities, mean_cr = _aggregate(votes, options)
    ranked = sorted(zip(options, priorities), key=lambda x: x[1], reverse=True)

    consensus = {
        "process_id": args.id,
        "aggregated_by": args.agent,
        "aggregated_at": datetime.now(timezone.utc).isoformat(),
        "participants": [v["participant"] for v in votes],
        "participant_count": len(votes),
        "method": "AIJ weighted geometric mean",
        "role_weights": {"nedxis": 1.5, "deelnemer": 1.0},
        "mean_consistency_ratio": round(mean_cr, 4),
        "excluded_inconsistent": inconsistent if not args.include_inconsistent else [],
        "ranking": [
            {"rank": i + 1, "option": opt, "score": round(float(score), 4)}
            for i, (opt, score) in enumerate(ranked)
        ],
        "verdict": ranked[0][0],
    }

    consensus_file = process_dir / "V.final_consensus.yaml"
    _write_yaml(consensus_file, consensus)

    # Update state to COMMITTED
    state_file = process_dir / "state.yaml"
    state = _read_yaml(state_file)
    state["state"] = "COMMITTED"
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_yaml(state_file, state)

    _log(args.id, args.agent, "V.final_consensus",
         f"Consensus: {ranked[0][0]} (score {ranked[0][1]:.3f}). "
         f"{len(votes)} participants, mean CR={mean_cr:.3f}",
         extra={"verdict": ranked[0][0], "participant_count": len(votes)})

    print(f"\n✅ Consensus written to V.final_consensus.yaml\n")
    print(f"{'Rank':<6} {'Score':<8} Option")
    print("─" * 40)
    for i, (opt, score) in enumerate(ranked, 1):
        marker = " ◀ winner" if i == 1 else ""
        print(f"{i:<6} {score:<8.4f} {opt}{marker}")
    print(f"\nMean CR: {mean_cr:.3f} | Participants: {len(votes)}")


def cmd_show(args):
    """Show current state of a decision process."""
    process_dir = PROCESSES_DIR / args.id
    session_file = process_dir / "P.0_decision_session.yaml"
    consensus_file = process_dir / "V.final_consensus.yaml"

    if not session_file.exists():
        print(f"No decision session found for {args.id}.")
        sys.exit(1)

    session = _read_yaml(session_file)
    print(f"\n=== {args.id}: {session.get('title', '')} ===")
    print(f"Options : {', '.join(session['options'])}")
    print(f"Status  : {session.get('status', 'open')}")
    print(f"Votes   : {len(session.get('votes', []))}")

    for v in session.get("votes", []):
        cr = v["consistency_ratio"]
        flag = "✅" if cr < 0.10 else ("⚠️" if cr < 0.20 else "❌")
        top = max(v["individual_priorities"].items(), key=lambda x: x[1])
        print(f"  {flag} {v['participant']} ({v['role']}) → {top[0]}  CR={cr:.3f}")

    if consensus_file.exists():
        c = _read_yaml(consensus_file)
        print(f"\n📊 Consensus ({c['aggregated_at'][:10]}):")
        for r in c["ranking"]:
            marker = " ◀" if r["rank"] == 1 else ""
            print(f"  {r['rank']}. {r['option']}  {r['score']:.4f}{marker}")
        print(f"   Mean CR: {c['mean_consistency_ratio']:.3f}  |  "
              f"Participants: {c['participant_count']}")


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="org_decision.py",
        description="Pairwise AHP decision engine for org-as-code",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # session
    s = sub.add_parser("session", help="Create a new decision session")
    s.add_argument("--id", required=True, help="Process ID, e.g. DEC-001")
    s.add_argument("--options", nargs="+", required=True, help="Options to compare")
    s.add_argument("--agent", default="facilitator", help="Creating agent/human")
    s.add_argument("--title", default="", help="Human-readable title")
    s.add_argument("--description", default="", help="Decision context")
    s.add_argument("--force", action="store_true", help="Overwrite existing session")

    # vote
    v = sub.add_parser("vote", help="Submit a pairwise vote (interactive)")
    v.add_argument("--id", required=True, help="Process ID")
    v.add_argument("--participant", required=True, help="Participant name/id")
    v.add_argument("--role", default="deelnemer",
                   choices=["deelnemer", "nedxis"], help="Participant role (affects weight)")
    v.add_argument("--force", action="store_true", help="Allow revoting")

    # aggregate
    a = sub.add_parser("aggregate", help="Aggregate votes into consensus")
    a.add_argument("--id", required=True, help="Process ID")
    a.add_argument("--agent", default="facilitator", help="Aggregating agent")
    a.add_argument("--include-inconsistent", action="store_true",
                   help="Include votes with CR >= 0.20")

    # show
    sh = sub.add_parser("show", help="Show decision process state")
    sh.add_argument("--id", required=True, help="Process ID")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "session": cmd_session,
        "vote": cmd_vote,
        "aggregate": cmd_aggregate,
        "show": cmd_show,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
