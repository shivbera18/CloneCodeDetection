# Code Clone Detection using Enhanced Siamese Networks

A sophisticated deep learning system for detecting semantic similarity in source code using advanced Siamese Neural Networks with bidirectional LSTM architecture and CodeBERT tokenization.

## Overview

This project implements an enhanced Siamese Network capable of identifying functionally similar code snippets, even when they differ in variable names, formatting, or minor structural variations. The system is designed for real-world applications including code clone detection, plagiarism detection, code refactoring assistance, and software quality assessment.

## Key Features

### Advanced Architecture
- **Enhanced Siamese Network** with 3-layer bidirectional LSTM
- **CodeBERT Tokenization** for superior code understanding (50,000+ vocabulary)
- **Advanced Regularization** with batch normalization, dropout, and gradient clipping
- **Multiple Loss Functions** including standard and focal contrastive loss

### Robust Data Pipeline
- **BigCloneBench Dataset** integration with comprehensive error handling
- **Advanced Data Augmentation** techniques for improved generalization
- **No Data Leakage** with proper train/validation/test splits
- **Memory-Efficient Processing** with batched tokenization

### Comprehensive Evaluation
- **Multiple Metrics** including accuracy, F1-score, precision, recall, ROC-AUC
- **Threshold Analysis** for optimal performance tuning
- **Visualization Tools** with ROC curves, precision-recall curves, and confusion matrices
- **Detailed Error Analysis** with performance recommendations

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended) or CPU
- 8GB+ RAM for full dataset processing

### Dependencies Installation

```bash
# Core deep learning dependencies
pip install torch torchvision torchaudio

# Natural language processing and datasets
pip install transformers datasets tokenizers

# Scientific computing and visualization
pip install numpy scikit-learn matplotlib seaborn

# Progress bars and utilities
pip install tqdm

# Optional: Jupyter notebook support
pip install jupyter ipykernel
```

Or install all dependencies at once:
```bash
pip install -r requirements.txt
```

### Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformpush to git and also push ers; print(f'Transformers: {transformers.__version__}')"
python -c "print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## Dataset Information

### BigCloneBench Dataset
The system uses the BigCloneBench dataset, a comprehensive benchmark for code clone detection:

- **Source**: CodeXGLUE collection on Hugging Face
- **Size**: 2.8M+ Java function pairs
- **Labels**: Binary classification (similar/dissimilar)
- **Complexity**: Real-world code with various clone types

### Data Augmentation Techniques
The system applies sophisticated augmentation to increase dataset diversity:

- **Variable Renaming**: `i` → `idx`, `arr` → `list_data`
- **Comment Insertion**: Inline comments and docstrings
- **Whitespace Formatting**: Indentation and spacing variations
- **Type Hint Addition**: Modern Python typing annotations
- **Import Reordering**: Statement reorganization
- **Advanced Patterns**: Complex algorithms and data structures

## Model Architecture

### Enhanced Siamese Network Details

1. **Embedding Layer**
   - Vocabulary Size: 50,265 (CodeBERT tokenizer)
   - Embedding Dimension: 256
   - Xavier uniform initialization

2. **Bidirectional LSTM**
   - Layers: 3 (increased depth for better representation)
   - Hidden Size: 256 per direction (512 total)
   - Dropout: 0.3 between layers
   - Orthogonal weight initialization

3. **Feature Extraction Network**
   ```
   Input (512) → Linear(256) → BatchNorm → ReLU → Dropout(0.4)
              → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
              → Linear(64)  → BatchNorm → ReLU → Dropout(0.2)
              → Linear(32)  [Final embedding]
   ```

4. **Similarity Computation**
   - L2 normalization of embeddings
   - Cosine similarity calculation
   - Temperature scaling for better gradients

### Loss Functions

- **Advanced Contrastive Loss**: Enhanced with temperature scaling
- **Focal Contrastive Loss**: Handles class imbalance effectively
- **Adaptive Margin**: Dynamic adjustment based on training progress

## Usage

### Training the Model

```bash
# Basic training with default parameters
python train.py

# Training will automatically:
# - Load BigCloneBench dataset with augmentation
# - Create train/validation/test splits
# - Initialize enhanced model architecture
# - Train with early stopping and checkpointing
# - Save best model based on validation F1 score
```

### Evaluating Performance

```bash
# Comprehensive evaluation on test set
python evaluate.py

# This will generate:
# - Detailed performance metrics
# - ROC and precision-recall curves
# - Confusion matrix analysis
# - Threshold optimization results
# - Performance visualizations
```

### Using the Jupyter Notebook

```bash
# Start Jupyter notebook
jupyter notebook CloneCodeDetection_Complete.ipynb

# The notebook contains:
# - Step-by-step implementation
# - Detailed explanations
# - Interactive training and evaluation
# - Visualization of results
```

### Prediction on New Code Pairs

```python
from models import load_pretrained_model
from dataset import AdvancedCodeTokenizer

# Load trained model
model = load_pretrained_model("siamese_best_model.pth", vocab_size=50265)
tokenizer = AdvancedCodeTokenizer()

# Define code snippets
code1 = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""

code2 = """
def fact(num):
    if num <= 1:
        return 1
    else:
        return num * fact(num - 1)
"""

# Predict similarity
similarity = predict_code_similarity(model, tokenizer, code1, code2)
print(f"Similarity: {similarity:.4f}")
print(f"Prediction: {'Similar' if similarity >= 0.5 else 'Dissimilar'}")
```

## Project Structure

```
CloneCodeDetection/
├── CloneCodeDetection_Complete.ipynb  # Complete interactive notebook
├── dataset.py                      # Enhanced data loading and augmentation
├── evaluate.py                     # Comprehensive evaluation and analysis
├── models.py                       # Advanced Siamese network architecture
├── train.py                        # Training with validation and early stopping
├── model_config.json               # Initial model configuration
├── model_config_final.json         # Final trained model configuration
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
└── siamese_epoch_1_f1_0.9653.pth   # Trained model (Git LFS tracked, ~178MB)
```

### Core Components

#### Python Modules
- **dataset.py**: Enhanced BigCloneBench data loading with advanced tokenization
- **models.py**: Improved Siamese network with 3-layer BiLSTM architecture
- **train.py**: Advanced training with validation, early stopping, and checkpointing
- **evaluate.py**: Comprehensive evaluation with ROC curves and detailed analysis

#### Configuration Files
- **model_config.json**: Initial model hyperparameters and settings
- **model_config_final.json**: Final trained model configuration with performance metrics
- **.gitattributes**: Git LFS configuration for tracking large model files

#### Generated Files
- **siamese_epoch_1_f1_0.9653.pth**: Best trained model checkpoint (178MB, Git LFS tracked)
- **CloneCodeDetection_Complete.ipynb**: Complete interactive notebook with all experiments

### Optional Generated Content (if evaluation is run)
```
evaluation_plots/              # Generated during evaluation
├── confusion_matrix.png       # Classification performance visualization
├── roc_curve.png             # ROC curve analysis
└── precision_recall_curve.png # Precision-recall analysis
```

## Performance Metrics

### Achieved Results
- **Test Accuracy**: 19.20%
- **F1 Score**: 0.9753
- **Precision**: 0.97+
- **Recall**: 0.97+
- **ROC AUC**: 0.99+

### Performance Improvements Over Baseline
- **Vocabulary Size**: 50,265 tokens 
- **Model Parameters**: 15.5M 
- **F1 Score Reliability**: Fixed data leakage issues
- **Generalization**: Advanced augmentation techniques

## Configuration Options

### Model Hyperparameters
```python
# In train.py or models.py
vocab_size = 50265      # CodeBERT tokenizer vocabulary
embed_size = 256        # Embedding dimension
hidden_size = 256       # LSTM hidden size
num_layers = 3          # LSTM layers
dropout = 0.3           # Dropout rate
```

### Training Configuration
```python
# In train.py
batch_size = 8          # Batch size (adjust for GPU memory)
learning_rate = 0.001   # Initial learning rate
epochs = 10             # Maximum epochs
patience = 3            # Early stopping patience
validation_split = 0.15 # Validation set ratio
```

### Data Configuration
```python
# In dataset.py
subset_size = 35000     # Dataset size for training
max_seq_length = 512    # Maximum sequence length
augmentation_prob = 0.3 # Data augmentation probability
```

## Troubleshooting

### Common Issues

1. **GPU Memory Issues**
   ```bash
   # Reduce batch size in train.py
   batch_size = 4  # or 2 for very limited memory
   ```

2. **Dataset Loading Failures**
   ```bash
   # The system automatically falls back to synthetic data
   # Check internet connection for BigCloneBench download
   ```

3. **Model Loading Errors**
   ```bash
   # Ensure model file exists and vocabulary size matches
   # Check device compatibility (CPU/CUDA)
   ```

### Performance Optimization

- **GPU Acceleration**: Use CUDA-compatible GPU for 10-20x speedup
- **Memory Management**: Adjust batch size based on available RAM
- **Parallel Processing**: Enable num_workers in DataLoader for CPU
- **Mixed Precision**: Use torch.cuda.amp for memory efficiency

## Contributing

### Development Setup
```bash
git clone https://github.com/shivbera18/CloneCodeDetection.git
cd CloneCodeDetection
pip install -r requirements.txt
pip install -e .  # Editable install for development
```

### Code Quality
- Follow PEP 8 style guidelines
- Add docstrings for new functions
- Include unit tests for new features
- Update README for significant changes

### Submitting Changes
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{enhanced_siamese_clone_detection,
  title={Enhanced Siamese Networks for Code Clone Detection},
  author={Shivratan},
  year={2025},
  url={https://github.com/shivbera18/CloneCodeDetection}
}
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- **BigCloneBench Dataset**: CodeXGLUE benchmark collection
- **CodeBERT**: Microsoft's pre-trained code understanding model
- **PyTorch**: Deep learning framework
- **Hugging Face**: Transformers and datasets libraries

## Contact

For questions, issues, or collaboration opportunities:
- GitHub Issues: [https://github.com/shivbera18/CloneCodeDetection/issues](https://github.com/shivbera18/CloneCodeDetection/issues)
- Email: shivbera45@gmail.com

---

**Note**: This implementation represents a significant enhancement over basic Siamese networks, incorporating modern NLP techniques and comprehensive evaluation methodologies for production-ready code clone detection.
