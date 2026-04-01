# Ahmed Iqbal
# Credit Risk Scorecard
# Predicts probability of loan default using logistic regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import textwrap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime

print("Loading real credit data...")
df = pd.read_csv("cs-training.csv")

df = df.rename(columns={
    "SeriousDlqin2yrs":                       "default",
    "RevolvingUtilizationOfUnsecuredLines":    "revolving_utilization",
    "age":                                     "age",
    "NumberOfTime30-59DaysPastDueNotWorse":    "past_due_30_59",
    "DebtRatio":                               "debt_ratio",
    "MonthlyIncome":                           "monthly_income",
    "NumberOfOpenCreditLinesAndLoans":         "open_credit_lines",
    "NumberOfTimes90DaysLate":                 "times_90_days_late",
    "NumberRealEstateLoansOrLines":            "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse":    "past_due_60_89",
    "NumberOfDependents":                      "dependents",
})

df = df.drop(columns=["Unnamed: 0"], errors="ignore")
df = df.dropna()
df = df[df["revolving_utilization"] <= 1.0]
df = df[df["debt_ratio"] <= 5.0]
df = df[df["monthly_income"] <= 50000]
df = df[df["age"] >= 18]
df = df.sample(10000, random_state=42).reset_index(drop=True)

print(f"Dataset: {len(df):,} real borrowers | Default rate: {df['default'].mean():.1%}")

features = [
    "revolving_utilization", "age", "past_due_30_59",
    "debt_ratio", "monthly_income", "times_90_days_late"
]

X = df[features]
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
model.fit(X_train_s, y_train)
y_pred  = model.predict(X_test_s)
y_proba = model.predict_proba(X_test_s)[:, 1]

auc    = roc_auc_score(y_test, y_proba)
report = classification_report(y_test, y_pred,
         target_names=["No Default", "Default"], output_dict=True)

print("\n--- Model Performance ---")
print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))
print(f"AUC-ROC: {auc:.3f}")

df["default_prob"] = model.predict_proba(scaler.transform(X))[:, 1]

def assign_risk_band(prob):
    if prob < 0.25:   return "Low"
    elif prob < 0.50: return "Medium"
    elif prob < 0.70: return "High"
    else:             return "Very High"

df["risk_band"] = df["default_prob"].apply(assign_risk_band)

band_order  = ["Low", "Medium", "High", "Very High"]
NAVY   = "#0a2342"
GOLD   = "#b8860b"
GREEN  = "#27ae60"
ORANGE = "#e67e22"
RED    = "#c0392b"
band_colors = {"Low": GREEN, "Medium": GOLD, "High": ORANGE, "Very High": RED}

band_summary = df.groupby("risk_band").agg(
    Borrowers=("risk_band", "count"),
    Actual_Default_Rate=("default", "mean"),
    Avg_Revolving_Util=("revolving_utilization", "mean"),
    Avg_Debt_Ratio=("debt_ratio", "mean"),
).round(3)

print("\n--- Risk Band Summary ---")
print(band_summary.to_string())

high_pct    = (df["risk_band"].isin(["High","Very High"])).mean() * 100
avg_dr_high = df[df["risk_band"].isin(["High","Very High"])]["debt_ratio"].mean()
overall_dr  = df["default"].mean() * 100
imp_ser     = pd.Series(np.abs(model.coef_[0]), index=features)
top_feat    = imp_ser.idxmax().replace("_", " ").title()
second_feat = imp_ser.nlargest(2).index[1].replace("_", " ").title()

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.patch.set_facecolor("#f8f9fa")

fig.text(0.5, 0.98,
         "Consumer Loan Credit Risk Scorecard  —  Real Portfolio Data",
         ha="center", va="top", fontsize=15, fontweight="bold", color=NAVY)
fig.text(0.5, 0.955,
         f"Prepared by Ahmed Iqbal  |  {datetime.today().strftime('%B %d, %Y')}  |  "
         f"n = {len(df):,} borrowers  |  Source: Kaggle Give Me Some Credit  |  AUC: {auc:.3f}",
         ha="center", va="top", fontsize=9, color="#666666")

ax1, ax2, ax3, ax4 = axes[0,0], axes[0,1], axes[1,0], axes[1,1]

band_counts = df["risk_band"].value_counts().reindex(band_order)
bars = ax1.bar(band_order, band_counts,
               color=[band_colors[b] for b in band_order], edgecolor="white", width=0.6)
ax1.set_title("Borrower Distribution by Risk Band", fontweight="bold", color=NAVY, pad=10)
ax1.set_ylabel("Number of Borrowers")
ax1.set_facecolor("#f8f9fa")
for bar, count in zip(bars, band_counts):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
             f"{count:,}", ha="center", fontsize=10, fontweight="bold")

dr_vals = [band_summary.loc[b, "Actual_Default_Rate"] * 100
           for b in band_order if b in band_summary.index]
bars2 = ax2.bar(band_order, dr_vals,
                color=[band_colors[b] for b in band_order], edgecolor="white", width=0.6)
ax2.set_title("Actual Default Rate by Risk Band", fontweight="bold", color=NAVY, pad=10)
ax2.set_ylabel("Default Rate (%)")
ax2.axhline(y=overall_dr, color="gray", linestyle="--", alpha=0.7,
            label=f"Portfolio avg: {overall_dr:.1f}%")
ax2.legend(fontsize=9)
ax2.set_facecolor("#f8f9fa")
for bar, rate in zip(bars2, dr_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{rate:.1f}%", ha="center", fontsize=10, fontweight="bold")

sample = df.sample(2000, random_state=1)
colors_s = [band_colors[b] for b in sample["risk_band"]]
ax3.scatter(sample["revolving_utilization"], sample["default_prob"],
            c=colors_s, alpha=0.35, s=14)
ax3.set_title("Revolving Utilization vs Predicted Default Probability",
              fontweight="bold", color=NAVY, pad=10)
ax3.set_xlabel("Revolving Utilization Rate")
ax3.set_ylabel("Predicted Default Probability")
ax3.set_facecolor("#f8f9fa")
patches = [mpatches.Patch(color=band_colors[b], label=b) for b in band_order]
ax3.legend(handles=patches, title="Risk Band", fontsize=9)

imp = imp_ser.sort_values()
labels = [f.replace("_", " ").title() for f in imp.index]
colors_i = [NAVY if v > imp.median() else GOLD for v in imp]
ax4.barh(labels, imp.values, color=colors_i)
ax4.set_title("Feature Importance (Model Coefficients)", fontweight="bold", color=NAVY, pad=10)
ax4.set_xlabel("Absolute Coefficient Value")
ax4.set_facecolor("#f8f9fa")

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("credit_risk_dashboard.png", dpi=150,
            bbox_inches="tight", facecolor="#f8f9fa")
plt.show()
print("\nDashboard saved as credit_risk_dashboard.png")

print("\n" + "="*60)
save_memo = input("Would you like to save an analyst memo? (y/n): ").strip().lower()

if save_memo == "y":
    memo = f"""
ANALYST MEMO
Consumer Loan Credit Risk Scorecard — Real Portfolio Data
Prepared by Ahmed Iqbal | {datetime.today().strftime('%B %d, %Y')}
Source: Kaggle Give Me Some Credit | n = {len(df):,} borrowers
Model AUC: {auc:.3f}
{"="*60}

KEY FINDINGS

{top_feat} and {second_feat} are the strongest predictors of default
in this portfolio, consistent with consumer credit risk research.
The logistic regression model achieves an AUC of {auc:.2f}, indicating
meaningful discriminatory power between performing and
non-performing borrowers.

{high_pct:.1f}% of the portfolio falls in the High or Very High risk
bands, representing elevated concentration risk. Borrowers in
these bands carry an average debt ratio of {avg_dr_high:.2f}, well
above the portfolio mean of {df['debt_ratio'].mean():.2f}. The overall
portfolio default rate stands at {overall_dr:.1f}%.

RECOMMENDATION

Tighten revolving utilization thresholds above 0.80 for new
originations. Flag accounts with 2 or more 90-day delinquencies
for early intervention review. Consider tiered pricing for Medium
band borrowers to reflect elevated risk while preserving
origination volume.

MODEL DETAILS
Logistic Regression (class_weight=balanced)
Train/Test Split: 80/20 stratified
Precision (Default): {report['Default']['precision']:.2f}
Recall (Default):    {report['Default']['recall']:.2f}
AUC-ROC:             {auc:.3f}
"""
    with open("analyst_memo.txt", "w") as f:
        f.write(memo)
    print("Analyst memo saved as analyst_memo.txt")
else:
    print("No memo saved. Dashboard is ready.")