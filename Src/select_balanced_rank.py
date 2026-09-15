import pandas as pd
import numpy as np

# ============================================================
# LOAD 4D RESULTS
# ============================================================

results = pd.read_csv("results_4d.csv")

# ============================================================
# NORMALIZE METRICS
# ============================================================

# NMSE:
# Lower NMSE is better
nmse_min = results["NMSE"].min()
nmse_max = results["NMSE"].max()

results["NMSE_Normalized"] = (
    (results["NMSE"] - nmse_min) /
    (nmse_max - nmse_min + 1e-30)
)

# Parameters:
# Lower complexity is better
param_min = results["Parameters"].min()
param_max = results["Parameters"].max()

results["Complexity_Normalized"] = (
    (results["Parameters"] - param_min) /
    (param_max - param_min + 1e-30)
)

# Execution time:
# Lower computation time is better
time_min = results["Execution Time (s)"].min()
time_max = results["Execution Time (s)"].max()

results["Time_Normalized"] = (
    (results["Execution Time (s)"] - time_min) /
    (time_max - time_min + 1e-30)
)

# ============================================================
# BALANCED SCORE
# ============================================================
#
# Equal importance:
#
# 50% accuracy
# 25% complexity
# 25% execution time
#
# Lower score = better trade-off

results["Balanced Score"] = (
    0.50 * results["NMSE_Normalized"]
    + 0.25 * results["Complexity_Normalized"]
    + 0.25 * results["Time_Normalized"]
)

# ============================================================
# SELECT BALANCED RANK
# ============================================================

best_index = results["Balanced Score"].idxmin()

balanced_rank = int(
    results.loc[best_index, "Rank"]
)

# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("BALANCED LOW-RANK SELECTION")
print("========================================")

print("\nNormalized metrics and score:")

print(
    results[
        [
            "Rank",
            "NMSE (dB)",
            "Parameters",
            "Execution Time (s)",
            "NMSE_Normalized",
            "Complexity_Normalized",
            "Time_Normalized",
            "Balanced Score"
        ]
    ].to_string(index=False)
)

# ============================================================
# FINAL RESULT
# ============================================================

selected = results.loc[best_index]

print("\n========================================")
print("SELECTED BALANCED RANK")
print("========================================")

print("Balanced Rank:", balanced_rank)
print("NMSE:", selected["NMSE"])
print("NMSE (dB):", selected["NMSE (dB)"])
print("Parameters:", int(selected["Parameters"]))
print("Execution Time:",
      selected["Execution Time (s)"], "seconds")
print("Balanced Score:",
      selected["Balanced Score"])

# ============================================================
# SAVE
# ============================================================

results.to_csv(
    "balanced_rank_results.csv",
    index=False
)

print("\n========================================")
print("SAVED")
print("========================================")

print("balanced_rank_results.csv")

print("\nBalanced rank selection completed successfully!")
