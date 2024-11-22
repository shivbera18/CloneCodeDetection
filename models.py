import torch.nn as nn
import torch

class ImprovedSiameseNetwork(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(ImprovedSiameseNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.bilstm = nn.LSTM(embed_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )

    def forward_once(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.bilstm(x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # Concatenate forward and backward states
        return self.fc(hidden)

    def forward(self, code1, code2):
        embedding1 = self.forward_once(code1)
        embedding2 = self.forward_once(code2)
        similarity = nn.functional.cosine_similarity(embedding1, embedding2)
        return similarity

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, similarity, labels):
        positive_loss = (1 - labels) * torch.pow(similarity, 2)
        negative_loss = labels * torch.pow(torch.clamp(self.margin - similarity, min=0.0), 2)
        return torch.mean(positive_loss + negative_loss)
