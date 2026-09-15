import deepmimo as dm
import numpy as np

scenario = "asu_campus_3p5"

print("Loading DeepMIMO dataset...")
dataset = dm.load(scenario)

print("\n===== DATASET PATH INFORMATION =====")

print("Power shape:", np.asarray(dataset.power).shape)
print("Phase shape:", np.asarray(dataset.phase).shape)
print("Delay shape:", np.asarray(dataset.delay).shape)

print("\n===== DATA TYPES =====")
print("Power:", np.asarray(dataset.power).dtype)
print("Phase:", np.asarray(dataset.phase).dtype)
print("Delay:", np.asarray(dataset.delay).dtype)

print("\n===== FIRST VALID USER =====")

# We already found user 9 to be valid
user_index = 9

power = np.asarray(dataset.power)
phase = np.asarray(dataset.phase)
delay = np.asarray(dataset.delay)

print("User index:", user_index)

print("\nPower:")
print(power[user_index])

print("\nPhase:")
print(phase[user_index])

print("\nDelay:")
print(delay[user_index])
