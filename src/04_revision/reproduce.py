"""
Independent re-run of the whole revised analysis on the author's own computer
(supervisor review of 30 October 2026, P0-6: "Document what the AI tool executed versus what the student
independently reproduced").

Usage, from the repository root, with the dataset files in data/ (see the main README):
    python src/04_revision/reproduce.py            # every script whose data are present
    python src/04_revision/reproduce.py --cic-only # only r07, r08, r12, r22 (the scripts that need the CIC files)
    python src/04_revision/reproduce.py --only=r25_decision_sensitivity,r26_ttl_and_label_conflicts

What it does:
  1. records who ran it, where and with which software versions;
  2. checks the SHA-256 of the dataset files against the values recorded in r23_provenance_manifest.json;
  3. copies results/revision to results/reproduction/<timestamp>/reference (the stored results);
  4. runs every script in order and keeps its console output;
  5. compares every output file with the stored version (numbers equal within 1e-9; run times ignored; CSV, NPZ
     and text files by content, see compare_outputs.py) and writes results/reproduction/<timestamp>/REPRODUCTION_REPORT.md.
Nothing is uploaded anywhere. Commit the report folder only if you want it to be public.
"""
import datetime
import getpass
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

from compare_outputs import compare as content_compare

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "results" / "revision"
SRC = ROOT / "src" / "04_revision"
ORDER = ["r01_unsw_detection", "r02_calibration", "r03_attack_category_severity", "r04_ablation_leakage_audit",
         "r05_nslkdd_official_split", "r06_ahp_robustness", "r07_cross_dataset_v2", "r08_cicids2017_holdout_v2",
         "r09_build_scenarios", "r10_decision_evaluation", "r11_unsw_duplicates", "r12_dataset_manifest",
         "r13_external_panel", "r14_final_checks", "r15_figures", "r16_level_bounds", "r17_base_rate_thresholds",
         "r18_threshold_study", "r19_ablation_and_paired", "r20_error_propagation", "r21_panel_audit",
         "r22_cross_dataset_audit", "r25_decision_sensitivity", "r26_ttl_and_label_conflicts"]
CIC = {"r07_cross_dataset_v2", "r08_cicids2017_holdout_v2", "r12_dataset_manifest", "r22_cross_dataset_audit"}
IGNORE_KEYS = ("seconds", "platform", "python")
UNSW = ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"]
NEEDS = {"r01_unsw_detection": UNSW, "r02_calibration": UNSW, "r03_attack_category_severity": UNSW,
         "r04_ablation_leakage_audit": UNSW, "r05_nslkdd_official_split": ["KDDTrain+.txt", "KDDTest+.txt"],
         "r07_cross_dataset_v2": UNSW + ["cicids2017_cleaned.csv", "balanced_7classes_500000_each.csv"],
         "r08_cicids2017_holdout_v2": ["cicids2017_cleaned.csv"], "r09_build_scenarios": UNSW, "r11_unsw_duplicates": UNSW,
         "r12_dataset_manifest": UNSW + ["KDDTrain+.txt", "KDDTest+.txt", "cicids2017_cleaned.csv", "balanced_7classes_500000_each.csv"],
         "r17_base_rate_thresholds": UNSW,
         "r22_cross_dataset_audit": UNSW + ["cicids2017_cleaned.csv", "balanced_7classes_500000_each.csv"],
         "r26_ttl_and_label_conflicts": UNSW}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def leaves(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{p}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{p}[{i}]")
    else:
        yield p, x


def compare(ref, new):
    if not ref.exists():
        return "new output (no stored version)"
    if ref.suffix == ".json":
        a, b = dict(leaves(json.loads(ref.read_text("utf-8")))), dict(leaves(json.loads(new.read_text("utf-8"))))
        diff = []
        for k in set(a) | set(b):
            if any(s in k for s in IGNORE_KEYS):
                continue
            x, y = a.get(k, "MISSING"), b.get(k, "MISSING")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
                if abs(x - y) > 1e-9:
                    diff.append((k, x, y))
            elif x != y:
                diff.append((k, x, y))
        return "identical" if not diff else f"{len(diff)} values differ, e.g. {diff[:3]}"
    if ref.read_bytes() == new.read_bytes():
        return "byte-identical"
    return content_compare(ref, new)       # CSV, NPZ and text compared by content (compare_outputs.py)


def main():
    cic_only = "--cic-only" in sys.argv
    only = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--only=")), None)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "results" / "reproduction" / stamp
    shutil.copytree(REV, out / "reference")
    lines = [f"# Reproduction report {stamp}", "",
             f"- Run by: {getpass.getuser()} on {socket.gethostname()} ({platform.platform()})",
             f"- Started: {datetime.datetime.now().isoformat(timespec='seconds')}",
             f"- Python {platform.python_version()}"]
    for mod in ("numpy", "pandas", "sklearn", "scipy", "matplotlib"):
        try:
            lines.append(f"- {mod} {__import__(mod).__version__}")
        except ImportError:
            lines.append(f"- {mod} not installed")
    manifest = json.loads((REV / "r23_provenance_manifest.json").read_text("utf-8"))["datasets"]
    lines += ["", "## Dataset files", "", "| File | Expected SHA-256 | Found | Match |", "| --- | --- | --- | --- |"]
    present = set()
    for name, rec in manifest.items():
        p = next((c for c in (ROOT / name, ROOT / "data" / name) if c.exists()), None)
        if p is None:
            lines.append(f"| {name} | {rec if isinstance(rec, str) else rec['sha256'][:16] + '...'} | missing | |")
            continue
        present.add(name)
        h = sha256(p)
        exp = rec["sha256"] if isinstance(rec, dict) else "(not recorded)"
        lines.append(f"| {name} | {exp[:16]}... | {h[:16]}... | {'yes' if h == exp else ('recorded now' if exp.startswith('(') else 'NO')} |")
    lines += ["", "## Scripts", "", "| Script | Exit | Seconds | Output comparison |", "| --- | --- | --- | --- |"]
    for s in ORDER:
        if (cic_only and s not in CIC) or (only and s not in only):
            continue
        missing = [f for f in NEEDS.get(s, []) if f not in present]
        if missing:
            lines.append(f"| {s} | skipped | | data missing: {', '.join(missing)} (stored results kept) |")
            print(s, "skipped, missing", missing)
            continue
        before = {p.name: p.stat().st_mtime for p in REV.glob("*") if p.is_file()}
        t0 = datetime.datetime.now()
        r = subprocess.run([sys.executable, f"{s}.py"], cwd=SRC, capture_output=True, text=True)
        sec = (datetime.datetime.now() - t0).total_seconds()
        (out / f"log_{s}.txt").write_text(r.stdout + "\n" + r.stderr, "utf-8")
        changed = [p for p in REV.glob("*") if p.is_file() and before.get(p.name) != p.stat().st_mtime]
        comp = "; ".join(f"{p.name}: {compare(out / 'reference' / p.name, p)}" for p in changed) or "no output written"
        lines.append(f"| {s} | {r.returncode} | {sec:.0f} | {comp} |")
        print(s, r.returncode, round(sec), comp[:160])
    lines += ["", f"Finished {datetime.datetime.now().isoformat(timespec='seconds')}."]
    (out / "REPRODUCTION_REPORT.md").write_text("\n".join(lines) + "\n", "utf-8")
    print("Report:", out / "REPRODUCTION_REPORT.md")


if __name__ == "__main__":
    main()
