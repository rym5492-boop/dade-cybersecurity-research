"""Minutes between each expert's October AHP revision and the same expert's scenario labels.

Added on 4 November 2026. Reads data/label_submission_times.csv (submission times read from the form service,
Tally, on 4 November 2026; names not recorded) and checks the revision times against the revision_log of
data/ahp_expert_judgments_anonymized.json, which records them to the minute in Saudi time (UTC+3).
Writes results/revision/label_timing.json. Run from src/04_revision or from the repository root.
"""
import csv
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KSA = timezone(timedelta(hours=3))


def utc(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


rows = list(csv.DictReader(open(ROOT / "data" / "label_submission_times.csv", encoding="utf-8")))
log = json.loads((ROOT / "data" / "ahp_expert_judgments_anonymized.json").read_text("utf-8"))["revision_log"]

out = {"source": "data/label_submission_times.csv (form-service submission times, UTC)", "experts": {}}
for r in rows:
    e = r["expert_id"]
    lab = utc(r["labels_submitted_utc"])
    rec = {"scenario_forms": r["scenario_forms"], "ahp_revision": r["ahp_revision"],
           "labels_submitted_ksa": lab.astimezone(KSA).strftime("%Y-%m-%d %H:%M:%S")}
    if r["ahp_revision_submitted_utc"]:
        rev = utc(r["ahp_revision_submitted_utc"])
        m = re.search(r"submitted (\d{4}-\d\d-\d\d \d\d:\d\d) \+0300", log[e])
        logged = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M").replace(tzinfo=KSA)
        assert abs((rev - logged).total_seconds()) < 60, f"{e}: revision time differs from revision_log"
        rec["ahp_revision_submitted_ksa"] = rev.astimezone(KSA).strftime("%Y-%m-%d %H:%M:%S")
        rec["minutes_from_revision_to_labels"] = round((lab - rev).total_seconds() / 60, 2)
        assert rec["minutes_from_revision_to_labels"] > 0
    out["experts"][e] = rec

mins = [v["minutes_from_revision_to_labels"] for v in out["experts"].values() if "minutes_from_revision_to_labels" in v]
out["feedback_round"] = {"n_experts": len(mins), "min_minutes": min(mins), "max_minutes": max(mins),
                         "range_in_whole_minutes": [round(min(mins)), round(max(mins))]}
dst = ROOT / "results" / "revision" / "label_timing.json"
dst.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(out["feedback_round"]))
