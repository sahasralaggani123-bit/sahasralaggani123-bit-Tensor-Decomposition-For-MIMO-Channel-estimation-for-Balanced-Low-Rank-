import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.cp_tensor import cp_to_tensor
import time
import csv

# ==========================================
# 3D BASELINE EXPERIMENT
# ==========================================

# Load existing 3D tensor
X = np.load("channel_tensor.npy").astype(np.complex128)

print("========================================")
print("        3D TENSOR BASELINE")
print("========================================")

print("Tensor shape:", X.shape)
print("Tensor dimensions:", X.ndim)
print("NaN values:", np.isnan(X).sum())
print("Inf values:", np.isinf(X).sum())

# Numerical scaling
scale = np.max(np.abs(X))
X_scaled = X / scale

# Ranks to test
ranks = [1, 2, 3, 4, 5]

results = []

for rank in ranks:

    print("\n----------------------------------------")
    print(f"Running CP decomposition - Rank {rank}")
    print("----------------------------------------")

    start_time = time.time()

    # CP decomposition
    cp_tensor = parafac(
        X_scaled,
        rank=rank,
        init="random",
        n_iter_max=200,
        tol=1e-7,
        random_state=42,
        verbose=0
    )

    # Reconstruction
    X_reconstructed = cp_to_tensor(cp_tensor)

    # Error calculations
    absolute_error = np.linalg.norm(
        X_scaled - X_reconstructed
    )

    relative_error = (
        absolute_error /
        np.linalg.norm(X_scaled)
    )

    nmse = (
        np.linalg.norm(X_scaled - X_reconstructed) ** 2
        /
        np.linalg.norm(X_scaled) ** 2
    )

    nmse_db = 10 * np.log10(nmse)

    execution_time = time.time() - start_time

    # CP parameter count
    parameters = rank * sum(X.shape) + rank

    results.append([
        rank,
        absolute_error,
        relative_error,
        nmse,
        nmse_db,
        parameters,
        execution_time
    ])

    print(f"Rank: {rank}")
    print(f"Relative Error: {relative_error:.6f}")
    print(f"NMSE: {nmse:.8f}")
    print(f"NMSE (dB): {nmse_db:.4f} dB")
    print(f"Parameters: {parameters}")
    print(f"Time: {execution_time:.2f} seconds")


# ==========================================
# SAVE RESULTS
# ==========================================

with open("results_3d.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Rank",
        "Absolute Error",
        "Relative Error",
        "NMSE",
        "NMSE (dB)",
        "Parameters",
        "Execution Time (s)"
    ])

    writer.writerows(results)


print("\n========================================")
print("3D BASELINE COMPLETED")
print("========================================")
print("Results saved to: results_3d.csv")