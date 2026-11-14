from dataset_ucfqnrf import UCFQNRFDataset

ds = UCFQNRFDataset("../ucf_qnrf", split="Train")

img, density = ds[0]

print("Image shape:", img.shape)
print("Density shape:", density.shape)
print("GT Count:", density.sum().item())
