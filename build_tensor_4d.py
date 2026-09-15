import deepmimo as dm
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

SCENARIO = "asu_campus_3p5"
NUM_USERS = 1000
NUM_ANTENNAS = 8
NUM_PATHS = 10
NUM_SUBCARRIERS = 32

# ============================================================
# 1. LOAD DEEPMIMO
# ============================================================

print("Loading DeepMIMO dataset...")

dataset = dm.load(SCENARIO)

print("Dataset loaded successfully!")

# ============================================================
# 2. GENERATE MIMO CHANNEL
# ============================================================

ch_params = dm.ChannelParameters()

ch_params.bs_antenna.shape = [NUM_ANTENNAS, 1]
ch_params.ue_antenna.shape = [1, 1]

# Use first 32 OFDM subcarriers
ch_params.ofdm.selected_subcarriers = np.arange(NUM_SUBCARRIERS)

print("\nComputing channels...")

dataset.compute_channels(ch_params)

H = np.asarray(dataset.channel)

print("\n========================================")
print("CHANNEL INFORMATION")
print("========================================")

print("Original channel shape:", H.shape)
print("Channel dtype:", H.dtype)

# Expected:
# (users, UE antennas, BS antennas, subcarriers)

H = H[:, 0, :, :]

print("Processed channel shape:", H.shape)

# ============================================================
# 3. FIND VALID USERS
# ============================================================

magnitude = np.abs(H)

valid_users = np.where(
    np.isfinite(magnitude).all(axis=(1, 2))
    & (np.max(magnitude, axis=(1, 2)) > 0)
)[0]

print("\n========================================")
print("VALID USERS")
print("========================================")

print("Total users:", H.shape[0])
print("Valid users:", len(valid_users))

# Select first 1000 valid users
num_users = min(NUM_USERS, len(valid_users))

selected_users = valid_users[:num_users]

H_selected = H[selected_users]

print("Selected users:", num_users)
print("Selected channel shape:", H_selected.shape)

# ============================================================
# 4. GET PATH INFORMATION
# ============================================================

power = np.asarray(dataset.power)[selected_users]
phase = np.asarray(dataset.phase)[selected_users]
delay = np.asarray(dataset.delay)[selected_users]

print("\n========================================")
print("PATH INFORMATION")
print("========================================")

print("Power shape:", power.shape)
print("Phase shape:", phase.shape)
print("Delay shape:", delay.shape)

# ============================================================
# 5. CLEAN PATH DATA
# ============================================================

power_clean = np.where(
    np.isfinite(power),
    power,
    -200.0
)

phase_clean = np.where(
    np.isfinite(phase),
    phase,
    0.0
)

delay_clean = np.where(
    np.isfinite(delay),
    delay,
    0.0
)

# Convert path power from dB to linear amplitude
path_amplitude = 10.0 ** (power_clean / 20.0)

# Convert phase to radians
phase_rad = np.deg2rad(phase_clean)

# Complex path coefficient
path_coefficient = (
    path_amplitude *
    np.exp(1j * phase_rad)
)

# ============================================================
# 6. CREATE SUBCARRIER FREQUENCIES
# ============================================================

bandwidth = ch_params.ofdm.bandwidth
total_subcarriers = ch_params.ofdm.subcarriers

# Subcarrier spacing
subcarrier_spacing = bandwidth / total_subcarriers

# First 32 selected subcarriers
selected_subcarriers = np.arange(NUM_SUBCARRIERS)

# Frequency offsets from the center frequency
frequency_offsets = (
    selected_subcarriers - total_subcarriers / 2
) * subcarrier_spacing

print("\n========================================")
print("OFDM INFORMATION")
print("========================================")

print("Bandwidth:", bandwidth)
print("Total subcarriers:", total_subcarriers)
print("Selected subcarriers:", NUM_SUBCARRIERS)
print("Subcarrier spacing:", subcarrier_spacing)
print("Frequency offset range:",
      frequency_offsets[0],
      "to",
      frequency_offsets[-1])

print("\n========================================")
print("OFDM INFORMATION")
print("========================================")

print("Bandwidth:", bandwidth)
print("Total subcarriers:", total_subcarriers)
print("Selected subcarriers:", NUM_SUBCARRIERS)
print("Subcarrier spacing:", subcarrier_spacing)


# ============================================================
# 7. PATH FREQUENCY RESPONSE
# ============================================================

print("\nBuilding path-frequency response...")

# Shape:
# path_coefficient -> (users, paths)
# delay            -> (users, paths)
# frequencies      -> (subcarriers)

frequency_response = (
    path_coefficient[:, :, np.newaxis]
    *
    np.exp(
        -1j
        * 2.0
        * np.pi
        * delay_clean[:, :, np.newaxis]
        * frequency_offsets[np.newaxis, np.newaxis, :]
    )
)

print(
    "Path-frequency response shape:",
    frequency_response.shape
)

# Expected:
# (1000, 10, 32)

# ============================================================
# 8. NORMALIZE PATH CONTRIBUTIONS
# ============================================================

path_power_linear = 10.0 ** (power_clean / 10.0)

path_power_sum = np.sum(
    path_power_linear,
    axis=1,
    keepdims=True
)

path_weights = path_power_linear / (
    path_power_sum + 1e-30
)

path_weights = np.nan_to_num(
    path_weights,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

# ============================================================
# 9. BUILD 4D TENSOR
# ============================================================

print("\nBuilding 4D tensor...")

# Normalize path-frequency contribution
frequency_response = (
    frequency_response
    * np.sqrt(path_weights[:, :, np.newaxis])
)

# Antenna channel response:
#
# H_selected:
# (users, antennas, subcarriers)
#
# We use the antenna-dependent channel magnitude/phase
# to distribute the path-frequency structure across antennas.

antenna_reference = H_selected

# Normalize antenna response per user/subcarrier
antenna_norm = np.sqrt(
    np.mean(
        np.abs(antenna_reference) ** 2,
        axis=1,
        keepdims=True
    )
)

antenna_norm = np.maximum(
    antenna_norm,
    1e-12
)

antenna_factor = antenna_reference / antenna_norm

# Build:
# users × antennas × paths × subcarriers

channel_tensor_4d = (
    antenna_factor[:, :, np.newaxis, :]
    *
    frequency_response[:, np.newaxis, :, :]
)

channel_tensor_4d = channel_tensor_4d.astype(np.complex64)

# ============================================================
# 10. NUMERICAL CHECK
# ============================================================

print("\n========================================")
print("FINAL 4D TENSOR")
print("========================================")

print("Tensor shape:", channel_tensor_4d.shape)
print("Tensor dtype:", channel_tensor_4d.dtype)

print("\nDimensions:")

print("Users:", channel_tensor_4d.shape[0])
print("BS antennas:", channel_tensor_4d.shape[1])
print("Paths:", channel_tensor_4d.shape[2])
print("Subcarriers:", channel_tensor_4d.shape[3])

print("\n========================================")
print("NUMERICAL CHECK")
print("========================================")

print(
    "NaN values:",
    np.isnan(channel_tensor_4d).sum()
)

print(
    "Inf values:",
    np.isinf(channel_tensor_4d).sum()
)

print(
    "Maximum magnitude:",
    np.max(np.abs(channel_tensor_4d))
)

print(
    "Minimum magnitude:",
    np.min(np.abs(channel_tensor_4d))
)

print(
    "Mean magnitude:",
    np.mean(np.abs(channel_tensor_4d))
)

# ============================================================
# 11. FIRST USER CHECK
# ============================================================

print("\n========================================")
print("FIRST USER CHECK")
print("========================================")

print("DeepMIMO user index:", selected_users[0])

print(
    "First user tensor slice shape:",
    channel_tensor_4d[0].shape
)

print(
    "First user / antenna 0 / path 0:"
)

print(
    channel_tensor_4d[0, 0, 0, :]
)

# ============================================================
# 12. SAVE
# ============================================================

np.save(
    "channel_tensor_4d.npy",
    channel_tensor_4d
)

np.save(
    "tensor_4d_user_indices.npy",
    selected_users
)

print("\n========================================")
print("SAVED")
print("========================================")

print("channel_tensor_4d.npy")
print("tensor_4d_user_indices.npy")

print("\n4D tensor construction completed successfully!")