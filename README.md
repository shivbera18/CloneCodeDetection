# Siamese Network for Code Clone Detection

This project implements a **Siamese Network** with a **Bidirectional LSTM** backbone for detecting semantic similarity in source code. The network predicts whether two code snippets are functionally identical, enabling use cases like code clone detection, plagiarism detection, and software refactoring.

---

## Features

- **Dataset**: Utilizes the **BigCloneBench** dataset from Hugging Face's `datasets` library.
- **Model**:
  - Embedding layer for converting tokenized code into dense vectors.
  - Bidirectional LSTM to capture sequential dependencies from both directions.
  - Fully connected layers for feature extraction.
  - Cosine similarity to compute similarity between code snippet embeddings.
- **Loss Function**: Implements **contrastive loss**, ensuring that embeddings for similar code snippets are close while dissimilar embeddings are far apart.
- **Evaluation**:
  - Metrics include **accuracy**, **F1-score**, **precision**, and **recall**.
  - Provides detailed results on the test dataset for robust analysis.

---

## Installation

### Prerequisites

Ensure the following tools and libraries are installed:

- Python 3.8 or higher
- PyTorch 1.9 or newer
- Additional Python dependencies listed in `requirements.txt`

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/shivbera18/CloneCodeDetection.git
   cd CloneCodeDetection
Install dependencies:

pip install -r requirements.txt
Verify installation by running:

python --version
python -c "import torch; print(torch.__version__)"
Dataset
The BigCloneBench dataset is used for training and testing. It is a large benchmark dataset for detecting functionally similar code snippets.

Dataset Details:
Source: BigCloneBench on Hugging Face
Classes: Binary labels (1 for similar code pairs, 0 for dissimilar).
Preprocessing:
Tokenized code snippets using Keras' Tokenizer.
Padded sequences to ensure uniform input length.


Model Architecture:

The Siamese network consists of two identical sub-networks for feature extraction. Key components:

1.Embedding Layer:
Maps tokenized code into dense vector representations.
Embedding size: 128.

2.Bidirectional LSTM:
Captures contextual relationships in code from both directions.
Hidden size: 128.
Two layers with dropout for regularization.

3.Fully Connected Layers:
Reduces feature dimensionality.
Includes Batch Normalization and Dropout.

4.Cosine Similarity:
Measures the similarity between embeddings of code snippets.

5.Contrastive Loss:
Encourages similar pairs to have high cosine similarity and dissimilar pairs to have low similarity.

Training
To train the model, run:
```bash
python train.py
```
Testing
Evaluate the model's performance on the test dataset:
```bash
python evaluate.py
```
 Adjust hyperparameters (e.g., learning rate, batch size, number of epochs) in the respective scripts (train.py, evaluate.py).
