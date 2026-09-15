import deepmimo as dm

scenario = "asu_campus_3p5"

print("Downloading DeepMIMO scenario...")
dm.download(scenario)

print("Loading dataset...")
dataset = dm.load(scenario)

print("Dataset loaded successfully!")
print(dataset)
