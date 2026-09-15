import deepmimo as dm

scenario = "asu_campus_3p5"

dataset = dm.load(scenario)

print("\n===== DATASET INFORMATION =====")
print("Dataset type:", type(dataset))

print("\n===== AVAILABLE FIELDS =====")
for key in dataset.keys():
    print(key)

print("\n===== CHANNEL INFORMATION =====")

if "channel" in dataset:
    channel = dataset["channel"]
    print("Channel type:", type(channel))
    print("Channel shape:", channel.shape)
    print("Channel dtype:", channel.dtype)
else:
    print("No direct 'channel' field found.")