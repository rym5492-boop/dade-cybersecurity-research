"""Documented coverage search for the related-work selection protocol (30 October 2026).

Runs each query against the Crossref REST API (https://api.crossref.org/works?query=...),
restricted to records published from 2015-01-01 to 2026-10-30 and to journal articles and
proceedings articles, sorted by relevance. Records the query string, the date and time of the
request, the total hit count reported by Crossref, and the top 20 records (DOI, title, year).
The top 20 titles of each query are screened by title with the rules in coverage_screening.py.
Only Crossref was searched. Usage (from the repository root):
    python src/05_manuscript/crossref_coverage_search.py results/manuscript/crossref_coverage_search_<date>.json
"""
import datetime, json, sys, urllib.parse, urllib.request

QUERIES = [
    "security alert prioritization triage business impact",
    "cyber incident prioritization asset criticality recovery time objective",
    "national cyber incident scoring system",
    "cyber risk matrix likelihood impact",
    "cost-sensitive alert prioritization security operations center",
    "human-in-the-loop alert triage intrusion detection",
    "business continuity cyber resilience machine learning decision support",
    "analytic hierarchy process cybersecurity risk prioritization",
]
FILTER = "from-pub-date:2015-01-01,until-pub-date:2026-10-30,type:journal-article,type:proceedings-article"
out = {"source": "Crossref REST API", "filter": FILTER, "rows_screened": 20, "queries": []}
for q in QUERIES:
    url = ("https://api.crossref.org/works?query=" + urllib.parse.quote(q) + "&filter=" + FILTER +
           "&rows=20&sort=relevance&select=DOI,title,issued,container-title,type")
    req = urllib.request.Request(url, headers={"User-Agent": "dade-coverage-check/1.0"})
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    with urllib.request.urlopen(req, timeout=90) as r:
        m = json.load(r)["message"]
    items = [{"doi": it.get("DOI"), "title": (it.get("title") or [""])[0],
              "year": (it.get("issued", {}).get("date-parts") or [[None]])[0][0],
              "venue": (it.get("container-title") or [""])[0]} for it in m["items"]]
    out["queries"].append({"query": q, "url": url, "requested_utc": stamp,
                           "total_results": m.get("total-results"), "top20": items})
    print(q, "|", m.get("total-results"))
json.dump(out, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
