import deepmimo as dm
import numpy as np

scenario = "asu_campus_3p5"

print("Loading DeepMIMO dataset...")
dataset = dm.load(scenario)

# Channel configuration
ch_params = dm.ChannelParameters()

# 8 BS antennas
ch_params.bs_antenna.shape = [8, 1]

# 1 antenna at the user
ch_params.ue_antenna.shape = [1, 1]

print("Generating channel...")
dataset.compute_channels(ch_params)

# Get channel
H = np.asarray(dataset.channel)

print("\n===== CHANNEL INFORMATION =====")
print("Shape:", H.shape)
print("Dtype:", H.dtype)

# Remove unnecessary dimensions
H_vectors = H[:, 0, :, 0]

print("\n===== SEARCHING FOR NON-ZERO CHANNELS =====")

# Calculate channel magnitude
magnitudes = np.abs(H_vectors)

# Find users having non-zero channel values
nonzero_users = np.where(np.max(magnitudes, axis=1) > 0)[0]

print("Total users:", H_vectors.shape[0])
print("Users with non-zero channel:", len(nonzero_users))

if len(nonzero_users) == 0:
    print("\nERROR: No non-zero channel values found.")
    exit()

# Take first 1000 valid users
num_users = min(1000, len(nonzero_users))

selected_indices = nonzero_users[:num_users]
H_small = H_vectors[selected_indices]

print("\n===== SELECTED CHANNEL =====")
print("Selected users:", num_users)
print("Channel shape:", H_small.shape)

# Display first valid channel
print("\n===== FIRST VALID USER =====")
print("User index:", selected_indices[0])

print(H_small[0])

print("\n===== MAGNITUDE =====")
print(np.abs(H_small[0]))

# Save
np.save("channel_data.npy", H_small)

# Save the corresponding user indices
np.save("channel_indices.npy", selected_indices)

print("\n===== SAVED =====")
print("channel_data.npy")
print("channel_indices.npy")
