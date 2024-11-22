import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from dataset import load_and_prepare_data
from models import ImprovedSiameseNetwork

# Load data
_, test_dataset, word_index = load_and_prepare_data()
test_loader = DataLoader(test_dataset, batch_size=32)

# Initialize model
vocab_size = len(word_index) + 1
embed_size = 128
hidden_size = 128
model = ImprovedSiameseNetwork(vocab_size, embed_size, hidden_size)

# Load trained model weights
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Evaluate
y_true = []
y_pred = []
with torch.no_grad():
    for code1, code2, labels in test_loader:
        similarity = model(code1, code2)
        predictions = (similarity >= 0.5).float()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())

# Compute metrics
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
