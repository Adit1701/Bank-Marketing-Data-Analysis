import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------
# 0. PATH CONSTANTS
# -------------------------------------------------------------
RAW_FILE   = "D:/nilesh/bank_marketing-nilesh/bank-additional-full.csv"
CLEAN_FILE = "D:/nilesh/bank_marketing-nilesh/bank-additional-full-cleaned.csv"
PLOTS_DIR  = "D:/nilesh/bank_marketing-nilesh/plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

PLOT_PATHS = {
    "age_hist":          os.path.join(PLOTS_DIR, "01_age_hist.png"),
    "job_bar":           os.path.join(PLOTS_DIR, "02_job_bar.png"),
    "target_count":      os.path.join(PLOTS_DIR, "03_target_count.png"),
    "education_bar":     os.path.join(PLOTS_DIR, "04_education_bar.png"),
    "duration_box":      os.path.join(PLOTS_DIR, "05_duration_box.png"),
    "age_vs_duration":   os.path.join(PLOTS_DIR, "06_age_vs_duration.png"),
    "sub_by_job":        os.path.join(PLOTS_DIR, "07_sub_by_job.png"),
    "campaign_vs_target":os.path.join(PLOTS_DIR, "08_campaign_vs_target.png"),
}

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 14

# -------------------------------------------------------------
# 1. LOAD THE RAW FILE  (';'-delimited)
# -------------------------------------------------------------
print(f"Loading raw file: {RAW_FILE}")
df = pd.read_csv(RAW_FILE, sep=";")
print("Raw shape:", df.shape)

# Standardise column names (strip accidental whitespace)
df.columns = [c.strip() for c in df.columns]

# -------------------------------------------------------------
# 2. REPLACE 'unknown' -> NaN
# -------------------------------------------------------------
obj_cols = df.select_dtypes(include="object").columns.tolist()
df[obj_cols] = df[obj_cols].apply(
    lambda s: s.str.strip().replace("unknown", np.nan)
)

print("\nMissing values per column (before imputation):")
print(df.isna().sum()[df.isna().sum() > 0])

# -------------------------------------------------------------
# 3. DATA TYPE CORRECTION
# -------------------------------------------------------------
numeric_cols = [
    "age", "duration", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed"
]
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# -------------------------------------------------------------
# 4. ENCODE THE TARGET VARIABLE  (y -> 0/1)
# -------------------------------------------------------------
df["y"] = df["y"].str.strip().str.lower().map({"yes": 1, "no": 0})

# -------------------------------------------------------------
# 5. FLAG pdays = 999  (sentinel for "never contacted")
# -------------------------------------------------------------
df["previously_contacted"] = (df["pdays"] != 999).astype(int)

# -------------------------------------------------------------
# 6. REMOVE DUPLICATES
# -------------------------------------------------------------
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"\nDuplicates removed: {before - len(df)}  |  New shape: {df.shape}")

# -------------------------------------------------------------
# 7. IMPUTE REMAINING MISSING VALUES
# -------------------------------------------------------------
df = df.dropna(subset=["y"]).reset_index(drop=True)

# Categorical -> mode
for c in obj_cols:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].mode()[0])

# Numeric -> median  (pdays excluded: 999 is intentional)
for c in numeric_cols:
    if c != "pdays" and df[c].isna().any():
        df[c] = df[c].fillna(df[c].median())

print("\nRemaining missing values:", int(df.isna().sum().sum()))

# -------------------------------------------------------------
# 8. OUTLIER DETECTION & HANDLING (99th-percentile winsorization)
# -------------------------------------------------------------
def report_outliers(series, factor=1.5):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - factor * iqr, q3 + factor * iqr
    return int(((series < lo) | (series > hi)).sum())

for c in ["duration", "campaign", "previous"]:
    n_out = report_outliers(df[c])
    upper = df[c].quantile(0.99)
    df[c] = df[c].clip(upper=upper)
    print(f"{c:10s}: {n_out:5d} outliers detected -> capped at 99th pct ({upper:.1f})")

# -------------------------------------------------------------
# 9. ADD USEFUL DERIVED COLUMNS
# -------------------------------------------------------------
df["month_num"]       = pd.to_datetime(df["month"], format="%b").dt.month
df["day_of_week_num"] = df["day_of_week"].map(
    {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5})
df["total_contacts"]  = df["campaign"] + df["previous"]
df["duration_min"]    = (df["duration"] / 60).round(2)

# -------------------------------------------------------------
# 10. SAVE THE CLEANED CSV
# -------------------------------------------------------------
df.to_csv(CLEAN_FILE, index=False)
print(f"\n✅ Cleaned CSV written -> {CLEAN_FILE}  |  shape: {df.shape}")

# =============================================================
# 11. VISUALISATIONS
# =============================================================
print("\nGenerating plots ...\n")

# -------- (1) AGE DISTRIBUTION – HISTOGRAM -------------------
plt.figure()
sns.histplot(df["age"], bins=30, kde=True, color="steelblue")
plt.title("Age Distribution of Customers")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(PLOT_PATHS["age_hist"], dpi=120)
plt.close()
print(f"Saved plot -> {PLOT_PATHS['age_hist']}")

# -------- (2) JOB DISTRIBUTION – BAR CHART -------------------
plt.figure(figsize=(11, 6))
order = df["job"].value_counts().index
sns.countplot(y="job", data=df, order=order, palette="viridis", hue="job",
              legend=False)
plt.title("Number of Customers by Job Type")
plt.xlabel("Count")
plt.ylabel("Job")
plt.tight_layout()
plt.savefig(PLOT_PATHS["job_bar"], dpi=120)
plt.close()
print(f"Saved plot -> {PLOT_PATHS['job_bar']}")

# -------- (3) SUBSCRIPTION STATUS – COUNT PLOT ---------------
plt.figure()
sns.countplot(x="y", data=df, palette=["#d9534f", "#5cb85c"], hue="y",
              legend=False)
plt.title("Subscription to Term Deposit (y)")
plt.xlabel("Subscribed (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(PLOT_PATHS["target_count"], dpi=120)
plt.close()
print(f"Saved plot -> {PLOT_PATHS['target_count']}")

# -------- (4) EDUCATION DISTRIBUTION – BAR CHART -------------
plt.figure(figsize=(10, 6))
order = df["education"].value_counts().index
sns.countplot(y="education", data=df, order=order, palette="mako",
              hue="education", legend=False)
plt.title("Customers by Education Level")
plt.xlabel("Count")
plt.ylabel("Education")
plt.tight_layout()
plt.savefig(PLOT_PATHS["education_bar"], dpi=120)
plt.close()
print(f"Saved plot -> {PLOT_PATHS['education_bar']}")

# -------- (5) CALL DURATION – BOXPLOT ------------------------
plt.figure()
sns.boxplot(x=df["duration"], color="skyblue")
plt.title("Boxplot of Call Duration (seconds) – outlier check")
plt.xlabel("Duration (s)")
plt.tight_layout()
plt.savefig(PLOT_PATHS["duration_box"], dpi=120)
plt.close()
print(f"Saved plot -> {PLOT_PATHS['duration_box']}")

# -------- (6) AGE vs CALL DURATION – SCATTER -----------------
plt.figure()
sns.scatterplot(x="age", y="duration", hue="y", data=df,
                palette={0: "#d9534f", 1: "#5cb85c"},
                alpha=0.5, s=18)
plt.title("Age vs Call Duration (coloured by subscription)")
plt.xlabel("Age")
plt.ylabel("Duration (s)")
plt.tight_layout()
plt.savefig(PLOT_PATHS["age_vs_duration"], dpi=120)
plt.close()
print(f"Saved plot -> {PLOT_PATHS['age_vs_duration']}")

# -------- (7) SUBSCRIPTION BY JOB – NORMALISED BAR -----------
plt.figure(figsize=(11, 6))
cross = (pd.crosstab(df["job"], df["y"], normalize="index") * 100)
cross[1].sort_values().plot(kind="barh", color="#5cb85c")
plt.title("Subscription Rate (%) by Job")
plt.xlabel("% Subscribed")
plt.ylabel("Job")
plt.tight_layout()
plt.savefig(PLOT_PATHS["sub_by_job"], dpi=120)
plt.close()
print(f"Saved plot -> {PLOT_PATHS['sub_by_job']}")

# -------- (8) CAMPAIGN CONTACTS vs SUBSCRIPTION – BOXPLOT ----
plt.figure()
sns.boxplot(x="y", y="campaign", data=df,
            palette=["#d9534f", "#5cb85c"], hue="y", legend=False)
plt.title("Number of Contacts During Campaign vs Subscription")
plt.xlabel("Subscribed (0 = No, 1 = Yes)")
plt.ylabel("Campaign Contacts")
plt.tight_layout()
plt.savefig(PLOT_PATHS["campaign_vs_target"], dpi=120)
plt.close()
print(f"Saved plot -> {PLOT_PATHS['campaign_vs_target']}")

# =============================================================
# 12. FINAL SUMMARY OF ALL OUTPUT PATHS
# =============================================================
print("\n=========================================================")
print("  ALL OUTPUTS GENERATED")
print("=========================================================")
print(f"  Cleaned CSV : {CLEAN_FILE}")
for name, path in PLOT_PATHS.items():
    print(f"  Plot ({name:<20s}) -> {path}")
print("=========================================================")
