import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# 0. Global configuration
# ------------------------------------------------------------------------------
DATA_FILE    = "D:/ADITYA MEHTA/bank_marketing/bank-additional-full.csv"
CLEANED_FILE = "D:/ADITYA MEHTA/bank_marketing/bank-additional-full-cleaned.csv"     # <-- cleaned output
OUT_DIR      = "plots"
RANDOM_SEED  = 42

os.makedirs(OUT_DIR, exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titleweight"] = "bold"


def save(name: str):
    """Save the current figure to the output folder and close it."""
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> saved {path}")


def banner(title: str):
    print("\n" + "=" * 78)
    print(f" {title}")
    print("=" * 78)


# ------------------------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------------------------
banner("1. LOADING DATASET")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"'{DATA_FILE}' not found. Place it in the same folder as this script."
    )

# The file uses ';' as separator and quotes strings with '"'
df = pd.read_csv(DATA_FILE, sep=";", quotechar='"')

print(f"Shape           : {df.shape}")
print(f"Columns         : {list(df.columns)}")
print("\nData types (before cleaning):")
print(df.dtypes)

print("\nFirst 5 rows:")
print(df.head())

# Save original shape for later comparison
ORIGINAL_SHAPE = df.shape


# ==============================================================================
# 2. DATA CLEANING  —  every step below MODIFIES df and is written to CSV
# ==============================================================================
banner("2. DATA CLEANING  (results written to cleaned CSV)")


# ---------- 2.1 Missing values ----------
print("\n[2.1] Missing values")
native_nan = df.isnull().sum()
print(f"  Native NaN cells     : {int(native_nan.sum())}")

# This dataset encodes missing categorical info as the string "unknown"
unknown_report = []
for col in df.select_dtypes(include="object").columns:
    n = (df[col] == "unknown").sum()
    if n:
        unknown_report.append((col, int(n), round(n / len(df) * 100, 2)))

if unknown_report:
    print("  'unknown'-encoded missing values:")
    for col, n, pct in unknown_report:
        print(f"    - {col:12s}: {n:5d}  ({pct:5.2f} %)")
print("  Decision: keep 'unknown' as its own category (informative, not noise).")
print("            Add a *_unknown indicator column for every such column.")


# ---------- 2.2 Duplicates ----------
print("\n[2.2] Duplicate rows")
n_dups_before = int(df.duplicated().sum())
print(f"  Duplicates found     : {n_dups_before}")
df = df.drop_duplicates().reset_index(drop=True)
print(f"  Shape after dedup    : {df.shape}")


# ---------- 2.3 Data types ----------
print("\n[2.3] Data type validation / correction")

print("  Age summary  :",
      df["age"].describe()[["min", "max", "mean"]].round(2).to_dict())
print("  pdays note   : 999 = 'never contacted before' (handled in 2.6)")
print("  Target values:", df["y"].unique().tolist())

# Encode target as binary for numeric analysis
df["y_bin"] = (df["y"] == "yes").astype(int)

# Convert categorical columns to pandas 'category' dtype
cat_cols = ["job", "marital", "education", "default", "housing", "loan",
            "contact", "month", "day_of_week", "poutcome", "y"]
for c in cat_cols:
    df[c] = df[c].astype("category")

print("  Added y_bin (0/1) and converted categorical columns to 'category' dtype.")


# ---------- 2.4 Outlier detection (IQR) ----------
print("\n[2.4] Outlier detection (IQR method)")
num_cols = ["age", "duration", "campaign", "pdays", "previous",
            "emp.var.rate", "cons.price.idx", "cons.conf.idx",
            "euribor3m", "nr.employed"]

outlier_rows = []
for col in num_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
    outlier_rows.append((col, round(lo, 3), round(hi, 3),
                         n_out, round(n_out / len(df) * 100, 2)))

out_df = pd.DataFrame(
    outlier_rows,
    columns=["feature", "lower_bound", "upper_bound", "n_outliers", "pct_%"]
)
print(out_df.to_string(index=False))


# ---------- 2.5 Outlier treatment (winsorization) ----------
print("\n[2.5] Outlier treatment — winsorization at the 99th percentile")
print("  Strategy: cap extreme counts, but PRESERVE informative outliers.")

winsorize_cols = ["campaign", "previous"]
for col in winsorize_cols:
    cap        = int(df[col].quantile(0.99))
    n_capped   = int((df[col] > cap).sum())
    before_max = int(df[col].max())
    df[col]    = df[col].clip(upper=cap).astype(int)
    print(f"  {col:10s}: max {before_max:3d} -> {cap:3d} "
          f"(winsorized {n_capped:4d} values)")

print("  Kept unchanged (informative extremes, NOT noise):")
print("    duration : long calls strongly predict 'yes' — must be preserved")
print("    age      : all ages (17-98) are legitimate")
print("    pdays    : sentinel value handled separately in 2.6")


# ---------- 2.6 Handle pdays sentinel (999) ----------
print("\n[2.6] pdays sentinel handling")
n_sentinel = int((df["pdays"] == 999).sum())
print(f"  pdays == 999 (sentinel)  : {n_sentinel:,} rows "
      f"({n_sentinel/len(df)*100:.2f} %)")

df["was_contacted_before"] = (df["pdays"] != 999).astype(int)
df["pdays"] = df["pdays"].replace(999, np.nan)

print("  Replaced 999 with NaN and added 'was_contacted_before' (0/1) flag.")


# ---------- 2.7 Derived columns & missing-value flags ----------
print("\n[2.7] Creating derived columns and missing-value flag columns")

# Unknown-flag indicator for every categorical column that contains 'unknown'
unknown_cols = [c for c in cat_cols if "unknown" in df[c].astype(str).values]
for c in unknown_cols:
    df[f"{c}_unknown"] = (df[c].astype(str) == "unknown").astype(int)
print(f"  Added *_unknown flags for: {unknown_cols}")

# Any-loan flag (housing OR personal loan)
df["has_any_loan"] = ((df["housing"].astype(str) == "yes") |
                      (df["loan"].astype(str) == "yes")).astype(int)

# Duration in minutes (rounded to 2 dp)
df["duration_min"] = (df["duration"] / 60).round(2)

# Total contact count = current campaign + previous campaigns
df["total_contacts"] = df["campaign"] + df["previous"]

# Age group bucketing
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 24, 34, 44, 54, 64, 100],
    labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
)

print("  Added: has_any_loan, duration_min, total_contacts, age_group")


# ---------- 2.8 Write cleaned CSV ----------
print("\n[2.8] Writing cleaned dataset to disk")

# Convert category dtype -> object so the CSV is portable
df_to_write = df.copy()
for c in cat_cols:
    df_to_write[c] = df_to_write[c].astype(str)

df_to_write.to_csv(CLEANED_FILE, index=False)
abs_clean = os.path.abspath(CLEANED_FILE)
print(f"  Cleaned CSV written  : {abs_clean}")
print(f"  Cleaned shape        : {df_to_write.shape}")
print(f"  Cleaned columns      : {list(df_to_write.columns)}")

# ---- Verification: reload from disk and confirm ----
print("\n  Verification — reloading cleaned CSV from disk …")
df_check = pd.read_csv(CLEANED_FILE, sep=",", quotechar='"')
assert df_check.shape == df_to_write.shape, "Shape mismatch after reload!"
assert list(df_check.columns) == list(df_to_write.columns), "Column mismatch!"
print(f"  ✔ Reload OK — {df_check.shape[0]:,} rows × {df_check.shape[1]} columns")


# ==============================================================================
# 3. VISUALIZATIONS  (all run on the CLEANED dataframe)
# ==============================================================================
banner("3. VISUALIZATIONS")


# ---------- Viz 1: Age distribution ----------
print("\n[Viz 1] Age distribution (histogram)")
plt.figure()
sns.histplot(df["age"], bins=30, kde=True, color="steelblue", edgecolor="white")
plt.axvline(df["age"].median(), color="red", linestyle="--",
            label=f"Median = {df['age'].median():.0f}")
plt.title("Customer Age Distribution")
plt.xlabel("Age (years)")
plt.ylabel("Count")
plt.legend()
save("01_age_histogram.png")


# ---------- Viz 2: Job distribution ----------
print("\n[Viz 2] Job distribution (horizontal bar chart)")
plt.figure(figsize=(11, 6))
job_order = df["job"].value_counts().index
sns.countplot(data=df, y="job", order=job_order, palette="viridis")
plt.title("Number of Customers by Job")
plt.xlabel("Count")
plt.ylabel("Job")
save("02_job_bar.png")


# ---------- Viz 3: Subscription status ----------
print("\n[Viz 3] Subscription status (count plot)")
plt.figure()
ax = sns.countplot(data=df, x="y", palette=["#d9534f", "#5cb85c"])
for p in ax.patches:
    h = int(p.get_height())
    ax.annotate(f"{h:,}\n({h/len(df)*100:.1f}%)",
                (p.get_x() + p.get_width() / 2, p.get_height() + 400),
                ha="center", fontsize=11)
plt.title("Campaign Outcome: Term-Deposit Subscription")
plt.xlabel("Subscribed (y)")
plt.ylabel("Count")
plt.ylim(0, df["y"].value_counts().max() * 1.15)
save("03_subscription_count.png")


# ---------- Viz 4: Education distribution ----------
print("\n[Viz 4] Education distribution (bar chart, split by outcome)")
plt.figure(figsize=(11, 6))
edu_order = df["education"].value_counts().index
sns.countplot(data=df, y="education", order=edu_order,
              hue="y", palette={"no": "#aec7e8", "yes": "#ff7f0e"})
plt.title("Education Level vs Subscription Outcome")
plt.xlabel("Count")
plt.ylabel("Education")
plt.legend(title="Subscribed")
save("04_education_bar.png")


# ---------- Viz 5: Duration boxplot ----------
print("\n[Viz 5] Call duration distribution (boxplot)")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(x=df["duration"], ax=axes[0], color="coral")
axes[0].set_title("Call Duration — Overall Distribution")
axes[0].set_xlabel("Duration (seconds)")

sns.boxplot(data=df, x="y", y="duration", ax=axes[1],
            palette={"no": "#d9534f", "yes": "#5cb85c"})
axes[1].set_title("Call Duration vs Subscription")
axes[1].set_xlabel("Subscribed (y)")
axes[1].set_ylabel("Duration (seconds)")

save("05_duration_boxplot.png")


# ---------- Viz 6: Age vs Duration scatter ----------
print("\n[Viz 6] Age vs Call Duration (scatter plot)")
sample = df.sample(min(5000, len(df)), random_state=RANDOM_SEED)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=sample, x="age", y="duration",
                hue="y", palette={"no": "#888888", "yes": "#d62728"},
                alpha=0.6, s=22, edgecolor=None)
plt.title("Age vs Call Duration (coloured by subscription)")
plt.xlabel("Age (years)")
plt.ylabel("Duration (seconds)")
plt.legend(title="Subscribed")
save("06_age_vs_duration.png")


# ==============================================================================
# 4. ADDITIONAL BONUS VISUALIZATIONS
# ==============================================================================
banner("4. ADDITIONAL BONUS VISUALIZATIONS")


# ---------- Bonus 1: Subscription rate by job ----------
print("\n[Bonus 1] Subscription rate by job")
rate_by_job = (df.groupby("job")["y_bin"].mean() * 100).sort_values()
plt.figure(figsize=(10, 6))
rate_by_job.plot(kind="barh", color="teal", edgecolor="white")
for i, v in enumerate(rate_by_job):
    plt.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)
plt.title("Subscription Rate (%) by Job")
plt.xlabel("% of customers who said 'yes'")
plt.ylabel("Job")
save("07_rate_by_job.png")


# ---------- Bonus 2: Previous campaign outcome vs subscription ----------
print("\n[Bonus 2] Previous campaign outcome vs current subscription")
plt.figure()
sns.countplot(data=df, x="poutcome", hue="y",
              palette={"no": "#aec7e8", "yes": "#ff7f0e"})
plt.title("Previous Campaign Outcome vs Current Subscription")
plt.xlabel("Previous campaign outcome")
plt.ylabel("Count")
plt.legend(title="Subscribed")
save("08_previous_outcome.png")


# ---------- Bonus 3: Correlation heatmap ----------
print("\n[Bonus 3] Correlation heatmap (numeric features)")
plt.figure(figsize=(10, 8))
corr = df[num_cols + ["y_bin"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix — Numeric Features vs Target")
save("09_correlation_heatmap.png")


# ---------- Bonus 4: Subscription rate by month ----------
print("\n[Bonus 4] Subscription rate by contact month")
month_rate = (df.groupby("month")["y_bin"].mean() * 100)
month_order = ["jan", "feb", "mar", "apr", "may", "jun",
               "jul", "aug", "sep", "oct", "nov", "dec"]
month_rate = month_rate.reindex([m for m in month_order
                                 if m in month_rate.index])

plt.figure(figsize=(11, 5))
bars = plt.bar(month_rate.index, month_rate.values,
               color="mediumpurple", edgecolor="white")
for bar, v in zip(bars, month_rate.values):
    plt.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
             f"{v:.1f}%", ha="center", fontsize=9)
plt.title("Subscription Rate (%) by Contact Month")
plt.xlabel("Month")
plt.ylabel("% yes")
save("10_rate_by_month.png")


# ==============================================================================
# 5. SUMMARY REPORT
# ==============================================================================
banner("5. SUMMARY REPORT")

total       = len(df)
n_yes       = int((df["y"] == "yes").sum())
n_no        = total - n_yes
yes_rate    = n_yes / total * 100

top_job     = df["job"].value_counts().index[0]
top_edu     = df["education"].value_counts().index[0]
med_dur_yes = df.loc[df["y"] == "yes", "duration"].median()
med_dur_no  = df.loc[df["y"] == "no",  "duration"].median()

best_poutcome = (
    df[df["poutcome"] != "nonexistent"]
    .groupby("poutcome")["y_bin"].mean().sort_values(ascending=False)
)
best_month = month_rate.idxmax()

corr_with_y = (df[num_cols + ["y_bin"]].corr()["y_bin"]
               .drop("y_bin").sort_values(key=lambda s: s.abs(), ascending=False))

print(f"  Original shape               : {ORIGINAL_SHAPE[0]:,} rows × {ORIGINAL_SHAPE[1]} columns")
print(f"  Cleaned shape                : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Rows removed as duplicates   : {n_dups_before}")
print(f"  New columns added            : "
      f"{df.shape[1] - ORIGINAL_SHAPE[1]}")
print(f"  Cleaned CSV saved to         : {abs_clean}")

print(f"\n  Subscribed (yes)             : {n_yes:,}  ({yes_rate:.2f} %)")
print(f"  Not subscribed (no)          : {n_no:,}  ({100 - yes_rate:.2f} %)")
print(f"  Most common job              : {top_job}")
print(f"  Most common education level  : {top_edu}")
print(f"  Median call duration (yes)   : {med_dur_yes:.0f} s")
print(f"  Median call duration (no)    : {med_dur_no:.0f} s")
print(f"  Best 'previous outcome'      : {best_poutcome.index[0]} "
      f"({best_poutcome.iloc[0] * 100:.1f} % conversion)")
print(f"  Best contact month           : {best_month} "
      f"({month_rate.loc[best_month]:.1f} % conversion)")

print("\n  Top absolute correlations with y_bin:")
for feat, val in corr_with_y.head(5).items():
    print(f"    - {feat:15s}: {val:+.3f}")

print(f"\n  All plots saved to: {os.path.abspath(OUT_DIR)}/")
print("  Done.\n")
