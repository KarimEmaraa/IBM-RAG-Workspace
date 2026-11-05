import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


emb = nn.Embedding(num_embeddings=4, embedding_dim=2)

print(emb.weight)  
print(emb(torch.tensor([1,2,3])))  