import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.cp_tensor import cp_to_tensor

tl.set_backend("numpy")

# --------------------------------------------------
# 1. Load channel tensor
# --------------------------------------------------

print("Loading channel tensor...")

X = np.load("channel_tensor.npy").astype(np.complex128)

print("Original tensor shape:", X.shape)
print("Original dtype:", X.dtype)

# --------------------------------------------------
# 2. Check for invalid values
# --------------------------------------------------

print("\n===== NUMERICAL CHECK =====")
print("NaN values:", np.isnan(X).sum())
print("Inf values:", np.isinf(X).sum())

# --------------------------------------------------
# 3. Scale the tensor
# --------------------------------------------------

max_value = np.max(np.abs(X))

print("Maximum magnitude:", max_value)

X_scaled = X / max_value

print("Scaled maximum magnitude:",
      np.max(np.abs(X_scaled)))

# --------------------------------------------------
# 4. CP decomposition
# --------------------------------------------------

rank = 3

print("\n===== CP DECOMPOSITION =====")
print("CP Rank:", rank)

factors = parafac(
    X_scaled,
    rank=rank,
    n_iter_max=100,
    init="random",
    tol=1e-7,
    verbose=1
)

print("\nCP decomposition completed successfully!")

# --------------------------------------------------
# 5. Reconstruct tensor
# --------------------------------------------------

X_hat_scaled = cp_to_tensor(factors)

# Convert back to original scale
X_hat = X_hat_scaled * max_value

print("\n===== RECONSTRUCTION =====")
print("Original shape:", X.shape)
print("Reconstructed shape:", X_hat.shape)

# --------------------------------------------------
# 6. Calculate errors
# --------------------------------------------------

error = np.linalg.norm(X - X_hat)

original_norm = np.linalg.norm(X)

relative_error = error / original_norm

nmse = (
    np.linalg.norm(X - X_hat) ** 2
    / np.linalg.norm(X) ** 2
)

nmse_db = 10 * np.log10(nmse)

# --------------------------------------------------
# 7. Results
# --------------------------------------------------

print("\n===== RESULTS =====")
print("CP Rank:", rank)
print("Absolute reconstruction error:", error)
print("Relative reconstruction error:", relative_error)
print("NMSE:", nmse)
print("NMSE (dB):", nmse_db)

# --------------------------------------------------
# 8. Save reconstructed tensor
# --------------------------------------------------

np.save(
    "reconstructed_tensor_rank3.npy",
    X_hat
)

print("\n===== SAVED =====")
print("reconstructed_tensor_rank3.npy")