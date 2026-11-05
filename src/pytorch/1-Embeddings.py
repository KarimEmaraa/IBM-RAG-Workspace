import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(42)

vocab = ["cat", "dog", "car", "truck"]
inputs = torch.tensor([0, 1, 2, 3], dtype=torch.long)
targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)

class ToyEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=4, embedding_dim=2)
        self.classifier = nn.Linear(in_features=2, out_features=2)

    def forward(self, x):
        coords = self.embedding(x)      
        logits = self.classifier(coords)
        return coords, logits

model = ToyEmbeddingModel()

print("--- Embeddings BEFORE Training ---")
for idx, word in enumerate(vocab):
    vec = model.embedding.weight.data[idx].numpy()
    print(f"{word:<6} (ID {idx}): [{vec[0]:.4f}, {vec[1]:.4f}]")

optimizer = optim.Adam(model.parameters(), lr=0.05)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

for epoch in range(300):
    optimizer.zero_grad()
    coords, logits = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    optimizer.zero_grad()

print("\n--- Embeddings AFTER Training ---")
for idx, word in enumerate(vocab):
    vec = model.embedding.weight.data[idx].numpy()
    print(f"{word:<6} (ID {idx}): [{vec[0]:.4f}, {vec[1]:.4f}]")

from sklearn.metrics.pairwise import cosine_similarity

# Calculate cosine similarities between embeddings
similarity_matrix = cosine_similarity(model.embedding.weight.data.numpy())
print("\n--- Cosine Similarities ---")
for i, word1 in enumerate(vocab):
    print(f"{word1:<6}", end="")
    for j, word2 in enumerate(vocab):
        print(f" [{similarity_matrix[i][j]:.4f}]", end="")
    print()