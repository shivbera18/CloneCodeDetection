# Siamese Network for Code Clone Detection

This repository implements a Siamese Network using a Bidirectional LSTM for code clone detection, trained on the [BigCloneBench](https://huggingface.co/datasets/google/code_x_glue_cc_clone_detection_big_clone_bench) dataset. The model identifies whether two pieces of source code perform the same function.

---

## Features

- **Dataset**: Leverages the "BigCloneBench" dataset from Hugging Face's `datasets` library.
- **Preprocessing**:
  - Tokenization and padding of source code sequences.
  - Conversion of text data into numerical tensors.
- **Model**:
  - Bidirectional LSTM to capture sequential relationships in code.
  - Fully connected layers for feature embedding.
  - Cosine similarity for pairwise comparison.
- **Loss Function**: Contrastive loss for training on similar and dissimilar code pairs.
- **Metrics**: Evaluation using accuracy, F1-score, precision, and recall.

---

## Installation

### Prerequisites
- Python 3.8+
- Install required packages:
  ```bash
   pip install -r requirements.txt


- Training :
To train the model
 ```
   python train.py


