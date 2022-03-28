import torch
import pickle

with open('features.txt', 'rb') as f:
    data = pickle.load(f)

print(data[10031].shape)