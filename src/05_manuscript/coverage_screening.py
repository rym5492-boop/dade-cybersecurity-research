"""Title screening of the Crossref coverage search (top 20 per query). One screener (the drafting assistant,
under the first author's direction); every decision is listed so the author can check it.
Categories: CITED (already in the reference list), TRIAGE (an alert or incident prioritisation, triage or
scoring method in cyber security), CONTEXT (asset criticality, business impact or continuity in cyber
decision support), OTHER (not relevant)."""
import csv, json, re, sys
d = json.load(open(sys.argv[1]))
cited = {"10.14722/ndss.2019.23349": "considered in the related-work report, not added"}
CITED = set()  # DOIs of the manuscript's reference list that could appear
TRIAGE = re.compile(r"(alert|incident).{0,40}(prioriti|triage|scoring|classification and prioriti)|triage|prioritization of exploit", re.I)
CONTEXT = re.compile(r"risk matrix|asset criticality|business impact|business continuity|cyber resilience|criticality levels|digital resilience|risk register|decision support.*(cyber|incident)|(cyber|incident).*decision support|asset prioritization|cybersecurity risk assessment through analytic|cybersecurity decision.making", re.I)
OTHER_OVERRIDE = re.compile(r"wildlife|animal|bank locker|smart home|face recognition|fall detection|clinical|vessel occlusion|sewer|water asset|work order|wage|covid|table tennis|crane|telecom|mutual funds|earthquake|flood|land consolidation|oil sands|sanitary|automotive|requirement|manual material|human-centric and sustainable city|small business enterprise|performance measures|customer voice|covariance|supply chains: application|hazop|software requisite|alternatives based on|table of contents|employee|duopoly|fleets|gsm-based|cyberloafing|insurance|victimi|extremist|optimism bias|ransomware attacks: a predictive|financial impact|economic impact|gamma regression|loss severity|kepada|pengelompokan|operational cyber resilience sebagai", re.I)
rows, seen = [], {}
for qi, q in enumerate(d["queries"], 1):
    for rank, it in enumerate(q["top20"], 1):
        t = it["title"] or ""
        if it["doi"] in CITED: cat = "CITED"
        elif OTHER_OVERRIDE.search(t): cat = "OTHER"
        elif TRIAGE.search(t): cat = "TRIAGE"
        elif CONTEXT.search(t): cat = "CONTEXT"
        else: cat = "OTHER"
        rows.append({"query": f"Q{qi}", "rank": rank, "doi": it["doi"], "year": it["year"], "title": t, "category": cat,
                     "note": cited.get(it["doi"], "")})
        seen.setdefault(it["doi"], cat)
with open(sys.argv[2], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
from collections import Counter
print("records", len(rows), "unique", len(seen), Counter(seen.values()))
for r in rows:
    if r["category"] != "OTHER": print(r["query"], r["rank"], r["category"], r["doi"], r["title"][:100])
