"""
R09 - Scenario bank for the expert decision-validation study (supervisor review 3.2, Critical 4/5).

30 hypothetical incident scenarios = 10 UNSW-NB15 traffic categories x 3 hypothetical business
profiles. Every scenario is anchored on ONE real held-out test flow:
  * detector score  = HistGradientBoosting (fitted on FIT) + isotonic calibration (fitted on CAL),
                      i.e. the leakage-safe detector of r01/r02;
  * shown category  = the category PREDICTED by r03 (decision-time information), never the true one;
  * true category   = kept only for analysis, never shown to experts.
Business profiles are HYPOTHETICAL, written from generic business-impact-analysis practice; they are
not data from any organisation and must be described as such in the paper.

Balance:
  * each category is paired with one high-, one medium- and one low-criticality profile;
  * each of the 6 profiles is used exactly 5 times;
  * detector-confidence tiers (high >= 0.90, moderate 0.60-0.90, low 0.30-0.60) rotate across the three
    profile levels, so confidence is not confounded with criticality;
  * forms A, B and C each contain every category exactly once (10 scenarios, about 6 minutes).
Outputs: results/revision/r09_scenarios.csv (analysis file, includes true category)
         results/revision/r09_scenario_cards_ar.md (expert-facing Arabic cards, no true category)
"""
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline

from common import OUT_DIR, SEED, SEVERITY_MAP, feature_cols, load_unsw, split_fit_val_cal, tree_preprocessor

CATS = ["Normal", "Reconnaissance", "DoS", "Exploits", "Backdoor", "Fuzzers", "Shellcode", "Worms",
        "Generic", "Analysis"]
CAT_AR = {
    "Normal": "غير محدد (يشبه الحركة الطبيعية)",
    "Reconnaissance": "استطلاع ومسح للشبكة (Reconnaissance)",
    "DoS": "حجب الخدمة (DoS)",
    "Exploits": "استغلال ثغرة (Exploit)",
    "Backdoor": "باب خلفي (Backdoor)",
    "Fuzzers": "إرسال بيانات عشوائية لاكتشاف ثغرات (Fuzzing)",
    "Shellcode": "حقن شيفرة تنفيذية (Shellcode)",
    "Worms": "دودة ذاتية الانتشار (Worm)",
    "Generic": "هجوم عام على التشفير أو البروتوكول (Generic)",
    "Analysis": "فحص وتحليل متقدم للأنظمة (Analysis)",
}
# Hypothetical profiles. AC/BI on 1-5; RTO/RPO in hours; HI = serious incidents in last 12 months.
PROFILES = {
    "P1": dict(level="high", service="نظام السجلات الطبية الإلكترونية في مستشفى", AC=5, BI=5, RTO_h=1, RPO_h=0.25, HI=1),
    "P2": dict(level="high", service="بوابة المدفوعات الإلكترونية في بنك", AC=5, BI=5, RTO_h=2, RPO_h=0.25, HI=2),
    "P3": dict(level="medium", service="بوابة الخدمات الإلكترونية للمواطنين في جهة حكومية", AC=4, BI=4, RTO_h=4, RPO_h=4, HI=1),
    "P4": dict(level="medium", service="نظام التعلم الإلكتروني في جامعة", AC=3, BI=3, RTO_h=24, RPO_h=24, HI=0),
    "P5": dict(level="low", service="نظام الموارد البشرية الداخلي في شركة", AC=2, BI=2, RTO_h=72, RPO_h=24, HI=0),
    "P6": dict(level="low", service="خادم تطوير واختبار في شركة برمجيات", AC=1, BI=1, RTO_h=168, RPO_h=168, HI=3),
}
LEVEL_AR = {1: "منخفضة جدًا", 2: "منخفضة", 3: "متوسطة", 4: "عالية", 5: "حرجة"}
TIERS = {"high": (0.90, 1.01), "moderate": (0.60, 0.90), "low": (0.30, 0.60)}
TIER_ORDER = ["high", "moderate", "low"]


def hours_ar(h):
    if h < 1:
        return f"{int(round(h * 60))} دقيقة"
    if h < 24:
        return "ساعة" if h == 1 else ("ساعتان" if h == 2 else f"{int(h)} ساعات")
    d = int(round(h / 24))
    return "يوم" if d == 1 else ("3 أيام" if d == 3 else ("أسبوع" if d == 7 else f"{d} أيام"))


def urgency(h):  # shorter objective = more urgent (bins from the revision plan, Section 4.1)
    return 1.0 if h <= 1 else 0.75 if h <= 4 else 0.5 if h <= 24 else 0.25 if h <= 72 else 0.0


# ---- leakage-safe detector: HGB on FIT + isotonic on CAL ----
train, test = load_unsw()
cols = feature_cols(train)
fit, val, cal = split_fit_val_cal(train)
base = Pipeline([("prep", tree_preprocessor(cols)), ("clf", HistGradientBoostingClassifier(random_state=SEED))])
base.fit(fit[cols], fit["label"])
det = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic").fit(cal[cols], cal["label"])
p_test = det.predict_proba(test[cols])[:, 1]
pred = np.load(OUT_DIR / "r03_test_predicted_severity.npz", allow_pickle=True)
pred_cat = pred["pred_cat"]
assert (pred["true_cat"] == test["attack_cat"].values).all()

rng = np.random.default_rng(SEED)
high_p, mid_p, low_p = ["P1", "P2"], ["P3", "P4"], ["P5", "P6"]
rows, used = [], set()
for i, cat in enumerate(CATS):
    profiles = [high_p[i % 2], mid_p[(i + 1) % 2], low_p[i % 2]]
    for slot, prof in enumerate(profiles):
        tier = TIER_ORDER[(i + slot) % 3]
        lo, hi = TIERS[tier]
        pool = np.where((test["attack_cat"].values == cat) & (p_test >= lo) & (p_test < hi))[0]
        pool = np.array([j for j in pool if j not in used])
        tier_used = tier
        if len(pool) == 0:  # fall back to the nearest tier that has flows
            for alt in TIER_ORDER:
                lo2, hi2 = TIERS[alt]
                pool = np.array([j for j in np.where((test["attack_cat"].values == cat) & (p_test >= lo2) & (p_test < hi2))[0]
                                 if j not in used])
                if len(pool):
                    tier_used = alt
                    break
        j = int(rng.choice(pool))
        used.add(j)
        P = PROFILES[prof]
        rows.append({
            # form = (i + 2*slot) mod 3: each form gets every category once, and because the tier is
            # (i + slot) mod 3, tiers and criticality levels are mixed inside every form.
            "form": "ABC"[(i + 2 * slot) % 3], "true_category": cat, "shown_category": pred_cat[j],
            "test_row_index": j, "test_id": int(test["id"].iloc[j]),
            "p_malicious": round(float(p_test[j]), 3), "confidence_tier_target": tier, "confidence_tier": tier_used,
            "profile": prof, "profile_level": P["level"], "service": P["service"],
            "AC_1to5": P["AC"], "BI_1to5": P["BI"], "RTO_hours": P["RTO_h"], "RPO_hours": P["RPO_h"], "HI_count": P["HI"],
            # normalised criteria for DADE and baselines (revision plan, Section 4.1)
            "MLS": round(float(p_test[j]), 3), "TS_pred": SEVERITY_MAP[pred_cat[j]],
            "AC": (P["AC"] - 1) / 4, "BI": (P["BI"] - 1) / 4, "HI": min(P["HI"] / 5, 1.0),
            "RTO": urgency(P["RTO_h"]), "RPO": urgency(P["RPO_h"]),
        })

df = pd.DataFrame(rows).sort_values(["form", "true_category"]).reset_index(drop=True)
df["scenario_id"] = [f"{f}{k + 1:02d}" for f, g in df.groupby("form") for k in range(len(g))]
# shuffle presentation order inside each form (fixed seed) so categories are not in a recognisable order
order = []
for f, g in df.groupby("form"):
    idx = g.index.to_numpy()
    order.extend(rng.permutation(idx))
df = df.loc[order].reset_index(drop=True)
df["scenario_id"] = [f"{f}{k + 1:02d}" for f, g in df.groupby("form", sort=False) for k in range(len(g))]
df.to_csv(OUT_DIR / "r09_scenarios.csv", index=False, encoding="utf-8-sig")

lines = ["# بطاقات السيناريوهات (للخبراء)", "",
         "سيناريوهات افتراضية. نوع الهجوم ودرجة الاحتمال من نظام كشف حقيقي على سجل شبكي فعلي؛ سياق الأعمال افتراضي.", ""]
for f, g in df.groupby("form", sort=True):
    lines += [f"## الاستمارة {f}", ""]
    for _, r in g.iterrows():
        lines += [f"**السيناريو {r.scenario_id}**",
                  f"- التنبيه: نوع الهجوم المتوقع: {CAT_AR[r.shown_category]}. احتمال أن الحركة خبيثة: "
                  f"{'أكثر من 99' if r.p_malicious >= 0.995 else round(100 * r.p_malicious)}%.",
                  f"- الخدمة المتأثرة: {r.service}.",
                  f"- أهمية الأصل: {LEVEL_AR[r.AC_1to5]} ({r.AC_1to5} من 5) · أثر توقف الخدمة على الأعمال: {LEVEL_AR[r.BI_1to5]} ({r.BI_1to5} من 5)",
                  f"- أقصى وقت مقبول للتعافي (RTO): {hours_ar(r.RTO_hours)} · أقصى فقد بيانات مقبول (RPO): {hours_ar(r.RPO_hours)}",
                  f"- حوادث جسيمة على هذه الخدمة خلال 12 شهرًا: {r.HI_count}", ""]
(OUT_DIR / "r09_scenario_cards_ar.md").write_text("\n".join(lines), encoding="utf-8")
print(df[["scenario_id", "true_category", "shown_category", "p_malicious", "confidence_tier", "profile"]].to_string())
print("Saved results/revision/r09_scenarios.csv and r09_scenario_cards_ar.md")
