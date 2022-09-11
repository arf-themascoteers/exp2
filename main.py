import torch
import hsi_train
import hsi_test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Training started...")
hsi_train.train(device)

print("Testing started...")
hsi_test.test(device)