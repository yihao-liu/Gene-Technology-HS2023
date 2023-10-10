import torch
device = torch.device("cuda:0")
print(f" Using device {device}")

print(f"Cuda available: {torch.cuda.is_available()}")

print(f"Number of devices: {torch.cuda.device_count()}")

print(f"Device 0 name: {torch.cuda.get_device_name(0)}")