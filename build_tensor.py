import deepmimo as dm
import numpy as np

scenario = "asu_campus_3p5"

print("Loading DeepMIMO dataset...")
dataset = dm.load(scenario)

# --------------------------------------------------
# 1. Generate MIMO channel
# --------------------------------------------------

ch_params = dm.ChannelParameters()

ch_params.bs_antenna.shape = [8, 1]
ch_params.ue_antenna.shape = [1, 1]

print("Generating channel...")
dataset.compute_channels(ch_params)

H = np.asarray(dataset.channel)

print("\n===== ORIGINAL CHANNEL =====")
print("Shape:", H.shape)
print("Dtype:", H.dtype)

# Remove singleton dimensions
H = H[:, 0, :, 0]

print("\n===== CHANNEL MATRIX =====")
print("Shape:", H.shape)

# --------------------------------------------------
# 2. Find valid channel users
# --------------------------------------------------

magnitudes = np.abs(H)

valid_users = np.where(
    np.isfinite(magnitudes).all(axis=1)
    & (np.max(magnitudes, axis=1) > 0)
)[0]

print("\n===== VALID USERS =====")
print("Total users:", H.shape[0])
print("Valid users:", len(valid_users))

# Select 1000 users
num_users = min(1000, len(valid_users))
selected_users = valid_users[:num_users]

H_selected = H[selected_users]

print("\n===== SELECTED CHANNEL =====")
print("Users:", num_users)
print("Antennas:", H_selected.shape[1])
print("Matrix shape:", H_selected.shape)

# --------------------------------------------------
# 3. Get path powers
# --------------------------------------------------

power = np.asarray(dataset.power)[selected_users]

print("\n===== PATH POWER =====")
print("Shape:", power.shape)

# Check invalid path-power values
print("NaN in power:", np.isnan(power).sum())
print("Inf in power:", np.isinf(power).sum())

# --------------------------------------------------
# 4. Replace invalid path powers
# --------------------------------------------------

power_clean = np.where(
    np.isfinite(power),
    power,
    -200.0
)

# Convert dB to linear power
path_power = 10.0 ** (power_clean / 10.0)

# --------------------------------------------------
# 5. Normalize path powers
# --------------------------------------------------

path_power_sum = np.sum(
    path_power,
    axis=1,
    keepdims=True
)

path_profile = path_power / (
    path_power_sum + 1e-30
)

# Safety check
path_profile = np.nan_to_num(
    path_profile,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

# --------------------------------------------------
# 6. Build channel tensor
# --------------------------------------------------

channel_tensor = (
    H_selected[:, :, np.newaxis]
    * np.sqrt(path_profile[:, np.newaxis, :])
)

channel_tensor = channel_tensor.astype(np.complex64)

# --------------------------------------------------
# 7. Final numerical check
# --------------------------------------------------

print("\n===== CHANNEL TENSOR =====")
print("Tensor shape:", channel_tensor.shape)
print("Tensor dtype:", channel_tensor.dtype)

print("\n===== NUMERICAL CHECK =====")
print("NaN values:", np.isnan(channel_tensor).sum())
print("Inf values:", np.isinf(channel_tensor).sum())

print(
    "Maximum magnitude:",
    np.max(np.abs(channel_tensor))
)

# --------------------------------------------------
# 8. First user
# --------------------------------------------------

print("\n===== FIRST USER =====")
print("User:", selected_users[0])

print(channel_tensor[0, 0, :])

# --------------------------------------------------
# 9. Save
# --------------------------------------------------

np.save(
    "channel_tensor.npy",
    channel_tensor
)

np.save(
    "tensor_user_indices.npy",
    selected_users
)

print("\n===== SAVED =====")
print("channel_tensor.npy")
print("tensor_user_indices.npy")

print("\nTensor construction completed successfully!")