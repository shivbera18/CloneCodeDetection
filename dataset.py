from datasets import load_dataset
from torch.utils.data import Dataset
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import torch

class CodePairDataset(Dataset):
    def __init__(self, code1, code2, labels):
        self.code1 = code1
        self.code2 = code2
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.code1[idx], self.code2[idx], self.labels[idx]

def load_and_prepare_data(train_subset_size=10000, test_subset_size=2000, max_seq_length=100, num_words=5000):
    # Load dataset
    dataset = load_dataset("google/code_x_glue_cc_clone_detection_big_clone_bench")
    train_data = dataset['train']
    test_data = dataset['validation']

    def prepare_data(data):
        code1 = [item['func1'] for item in data]
        code2 = [item['func2'] for item in data]
        labels = [1 if item['label'] else 0 for item in data]
        return code1, code2, labels

    # Subset data
    train_data_subset = [train_data[i] for i in range(min(len(train_data), train_subset_size))]
    test_data_subset = [test_data[i] for i in range(min(len(test_data), test_subset_size))]
    train_code1, train_code2, train_labels = prepare_data(train_data_subset)
    test_code1, test_code2, test_labels = prepare_data(test_data_subset)

    # Tokenization and padding
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(train_code1 + train_code2 + test_code1 + test_code2)

    def tokenize_and_pad(codes1, codes2):
        code1_sequences = tokenizer.texts_to_sequences(codes1)
        code2_sequences = tokenizer.texts_to_sequences(codes2)
        code1_padded = pad_sequences(code1_sequences, maxlen=max_seq_length, padding="post")
        code2_padded = pad_sequences(code2_sequences, maxlen=max_seq_length, padding="post")
        return torch.tensor(code1_padded, dtype=torch.long), torch.tensor(code2_padded, dtype=torch.long)

    train_code1_padded, train_code2_padded = tokenize_and_pad(train_code1, train_code2)
    test_code1_padded, test_code2_padded = tokenize_and_pad(test_code1, test_code2)

    train_labels = torch.tensor(train_labels, dtype=torch.float)
    test_labels = torch.tensor(test_labels, dtype=torch.float)

    train_dataset = CodePairDataset(train_code1_padded, train_code2_padded, train_labels)
    test_dataset = CodePairDataset(test_code1_padded, test_code2_padded, test_labels)

    return train_dataset, test_dataset, tokenizer.word_index
