#!/usr/bin/env python3
"""Record a privacy evaluation in MLflow, with the fields that make it readable.

Why this exists rather than plain log files: the CNIL requires that
re-identification risk be quantified "taking into account the diversity of voice
samples accessible to an attacker and the capabilities of that attacker"
(obligation G2, see ``docs/cnil-opinion.md``). A privacy number without its
population size and attacker model is therefore not merely incomplete, it does
not meet the standard the regulator set. This script refuses to log one.

It is deliberately local and file-backed. The tracking store is a directory, no
server and no vendor, which matters for a project whose own subject is data
governance.

Usage::

    python scripts/log_run_to_mlflow.py --results results/comparison/ignorant.json
    python scripts/log_run_to_mlflow.py --results ... --tracking-uri file:///path/mlruns
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

# MLflow 3.x put the filesystem store into maintenance mode and raises unless
# MLFLOW_ALLOW_FILE_STORE is set, so the backend is SQLite: still one local file,
# no server and no vendor, and queryable with plain SQL, which matters when the
# question is "which commit produced this number".
EXPERIMENTS = Path.home() / "PROJECTS" / "Interspeech2025" / "experiments"
DEFAULT_STORE = f"sqlite:///{EXPERIMENTS / 'mlflow.db'}"
DEFAULT_ARTIFACTS = (EXPERIMENTS / "mlflow-artifacts").as_uri()

# Without these a value does not meet CNIL obligation G2.
REQUIRED = ("n", "attacker", "dataset")


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def rows_from_comparison(payload: dict, attacker: str) -> list[dict]:
    """Flatten a 04_compare_privacy_metrics.py JSON into one row per (system, N)."""
    out = []
    for name, res in payload.get("systems", {}).items():
        for row in res.get("by_n", []):
            out.append(
                {
                    "system": name,
                    "dataset": res.get("dataset", "unknown"),
                    "label": res.get("label", name),
                    "attacker": attacker,
                    "separable": bool(res.get("separable", False)),
                    "slope": res.get("slope"),
                    "one_minus_eer": res.get("one_minus_eer"),
                    "eer_sem": res.get("eer_sem"),
                    **row,
                }
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--attacker", default=None,
                        help="defaults to the results filename stem")
    parser.add_argument("--experiment", default="privacy-metrics")
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()

    import mlflow

    store = args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_STORE
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    (EXPERIMENTS / "mlflow-artifacts").mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(store)
    if mlflow.get_experiment_by_name(args.experiment) is None:
        mlflow.create_experiment(args.experiment, artifact_location=DEFAULT_ARTIFACTS)
    mlflow.set_experiment(args.experiment)

    payload = json.loads(args.results.read_text())
    attacker = args.attacker or args.results.stem
    rows = rows_from_comparison(payload, attacker)
    if not rows:
        print(f"no system rows in {args.results}")
        return 1

    repo = Path(__file__).resolve().parents[1]
    commit = git_commit(repo)

    logged = 0
    for row in rows:
        missing = [k for k in REQUIRED if row.get(k) in (None, "", "unknown")]
        if missing:
            print(f"  skipped {row['system']} N={row.get('n')}: "
                  f"missing {', '.join(missing)}, which CNIL G2 requires")
            continue
        run_name = f"{row['dataset']}/{row['label']}/N{row['n']}/{attacker}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(
                {
                    "attacker": row["attacker"],
                    "dataset": row["dataset"],
                    "system": row["label"],
                    "separable_calibration": str(row["separable"]),
                    "git_commit": commit or "unknown",
                    "cnil_g2_fields_present": "true",
                }
            )
            mlflow.log_params(
                {"n_speakers": row["n"], "attacker": row["attacker"],
                 "dataset": row["dataset"], "lid_platt_slope": row.get("slope")}
            )
            for key in (
                "linkability", "linkability_sem", "singling_out",
                "singling_out_sem", "srd_mean", "srd_mean_sem", "srd_idr",
                "lid_mean", "lid_mean_sem", "lid_pdr", "one_minus_eer", "eer_sem",
            ):
                value = row.get(key)
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, float(value))
            logged += 1

    print(f"logged {logged} of {len(rows)} rows to {store}")
    print(f"  mlflow ui --backend-store-uri {store}")
    print(f"  sqlite3 {EXPERIMENTS / 'mlflow.db'} "
          f"\"select run_uuid, key, value from metrics limit 5\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
