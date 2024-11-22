import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from dataset import load_and_prepare_data
from models import ImprovedSiameseNetwork, ContrastiveLoss

# Load data
train_dataset, test_dataset, word_index = load_and_prepare_data()
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model
vocab_size = len(word_index) + 1
embed_size = 128
hidden_size = 128
model = ImprovedSiameseNetwork(vocab_size, embed_size, hidden_size)
criterion = ContrastiveLoss(margin=1.0)
optimizer = Adam(model.parameters(), lr=0.0015)
scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

# Training loop
epochs = 5
batch_limit = 100
model.train()
for epoch in range(epochs):
    total_loss = 0
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
        for batch_idx, (code1, code2, labels) in enumerate(pbar):
            if batch_idx >= batch_limit:
                break
            optimizer.zero_grad()
            similarity = model(code1, code2)
            loss = criterion(similarity, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (batch_idx + 1))
    scheduler.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / batch_limit:.4f}")
