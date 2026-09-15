import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# LOAD RESULTS
# ============================================================

results_3d = pd.read_csv("results_3d.csv")
results_4d = pd.read_csv("results_4d.csv")

print("\n========================================")
print("3D vs 4D COMPARISON")
print("========================================")

print("\n3D RESULTS:")
print(results_3d.to_string(index=False))

print("\n4D RESULTS:")
print(results_4d.to_string(index=False))

# ============================================================
# 1. NMSE vs RANK
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    results_3d["Rank"],
    results_3d["NMSE (dB)"],
    marker="o",
    label="3D"
)

plt.plot(
    results_4d["Rank"],
    results_4d["NMSE (dB)"],
    marker="o",
    label="4D"
)

plt.xlabel("CP Rank")
plt.ylabel("NMSE (dB)")
plt.title("3D vs 4D: NMSE vs CP Rank")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("nmse_3d_vs_4d.png", dpi=300)
plt.show()

# ============================================================
# 2. PARAMETERS vs RANK
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    results_3d["Rank"],
    results_3d["Parameters"],
    marker="o",
    label="3D"
)

plt.plot(
    results_4d["Rank"],
    results_4d["Parameters"],
    marker="o",
    label="4D"
)

plt.xlabel("CP Rank")
plt.ylabel("Number of Parameters")
plt.title("3D vs 4D: Model Complexity")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("parameters_3d_vs_4d.png", dpi=300)
plt.show()

# ============================================================
# 3. EXECUTION TIME vs RANK
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    results_3d["Rank"],
    results_3d["Execution Time (s)"],
    marker="o",
    label="3D"
)

plt.plot(
    results_4d["Rank"],
    results_4d["Execution Time (s)"],
    marker="o",
    label="4D"
)

plt.xlabel("CP Rank")
plt.ylabel("Execution Time (seconds)")
plt.title("3D vs 4D: Execution Time")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("time_3d_vs_4d.png", dpi=300)
plt.show()

# ============================================================
# 4. NMSE vs PARAMETERS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    results_3d["Parameters"],
    results_3d["NMSE (dB)"],
    marker="o",
    label="3D"
)

plt.plot(
    results_4d["Parameters"],
    results_4d["NMSE (dB)"],
    marker="o",
    label="4D"
)

plt.xlabel("Number of Parameters")
plt.ylabel("NMSE (dB)")
plt.title("Accuracy vs Model Complexity")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("accuracy_vs_complexity.png", dpi=300)
plt.show()

# ============================================================
# 5. COMBINED COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame({
    "Rank": results_3d["Rank"],
    "3D NMSE (dB)": results_3d["NMSE (dB)"],
    "4D NMSE (dB)": results_4d["NMSE (dB)"],
    "3D Parameters": results_3d["Parameters"],
    "4D Parameters": results_4d["Parameters"],
    "3D Time (s)": results_3d["Execution Time (s)"],
    "4D Time (s)": results_4d["Execution Time (s)"]
})

print("\n========================================")
print("COMPARISON TABLE")
print("========================================")

print(comparison.to_string(index=False))

comparison.to_csv(
    "comparison_3d_4d.csv",
    index=False
)

print("\n========================================")
print("FILES SAVED")
print("========================================")

print("nmse_3d_vs_4d.png")
print("parameters_3d_vs_4d.png")
print("time_3d_vs_4d.png")
print("accuracy_vs_complexity.png")
print("comparison_3d_4d.csv")

print("\nComparison completed successfully!")