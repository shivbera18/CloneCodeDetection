"""
Enhanced Dataset Module for Code Clone Detection
Implements BigCloneBench loading with advanced tokenization and data augmentation
"""

import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer
import numpy as np
import re
from collections import Counter
import random

class CodePairDataset(Dataset):
    """
    Enhanced PyTorch Dataset for handling code pairs with similarity labels.
    """
    
    def __init__(self, code1, code2, labels):
        self.code1 = code1
        self.code2 = code2
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.code1[idx], self.code2[idx], self.labels[idx]

class AdvancedCodeTokenizer:
    """
    Advanced tokenizer using pre-trained CodeBERT for better code understanding.
    """
    def __init__(self, model_name="microsoft/codebert-base", max_length=512):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.max_length = max_length
            self.vocab_size = self.tokenizer.vocab_size
            print(f"Loaded {model_name} tokenizer (vocab_size: {self.vocab_size})")
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            raise e
        
    def tokenize_and_pad(self, texts):
        """Tokenize and pad text sequences."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return encoded['input_ids']

class CodeAugmenter:
    """
    Advanced code augmentation for increasing dataset diversity.
    """
    
    @staticmethod
    def augment_code(code_text, augmentation_prob=0.7):
        """Apply various code augmentations to increase diversity."""
        if random.random() > augmentation_prob:
            return code_text
        
        augmentations = [
            CodeAugmenter._variable_rename,
            CodeAugmenter._add_comments,
            CodeAugmenter._reformat_whitespace,
            CodeAugmenter._add_type_hints,
            CodeAugmenter._reorder_imports,
            CodeAugmenter._add_docstrings,
        ]
        
        # Apply 1-2 random augmentations
        num_augs = random.randint(1, 2)
        selected_augs = random.sample(augmentations, min(num_augs, len(augmentations)))
        
        augmented_code = code_text
        for aug_func in selected_augs:
            try:
                augmented_code = aug_func(augmented_code)
            except:
                continue  # Skip if augmentation fails
        
        return augmented_code
    
    @staticmethod
    def _variable_rename(code):
        """Rename variables to increase diversity."""
        common_vars = ['temp', 'result', 'value', 'item', 'data', 'element']
        replacements = {
            'i': random.choice(['idx', 'index', 'counter']),
            'j': random.choice(['jdx', 'inner_idx', 'col']),
            'arr': random.choice(['array', 'list_data', 'items']),
            'str': random.choice(['text', 'string_val', 'content']),
            'num': random.choice(['number', 'value', 'digit']),
        }
        
        for old_var, new_var in replacements.items():
            if old_var in code:
                code = re.sub(r'\b' + old_var + r'\b', new_var, code)
        
        return code
    
    @staticmethod
    def _add_comments(code):
        """Add inline comments."""
        lines = code.split('\n')
        commented_lines = []
        
        for line in lines:
            commented_lines.append(line)
            if 'def ' in line and random.random() < 0.3:
                indent = len(line) - len(line.lstrip())
                commented_lines.append(' ' * (indent + 4) + '# Function implementation')
            elif 'return' in line and random.random() < 0.2:
                commented_lines[-1] = line + '  # Return result'
        
        return '\n'.join(commented_lines)
    
    @staticmethod
    def _reformat_whitespace(code):
        """Modify whitespace formatting."""
        if random.random() < 0.5:
            # Change indentation from 4 to 2 spaces or vice versa
            if '    ' in code:
                code = code.replace('    ', '  ')
            else:
                code = code.replace('  ', '    ')
        
        return code
    
    @staticmethod
    def _add_type_hints(code):
        """Add simple type hints."""
        if 'def ' in code:
            # Add return type hints
            code = re.sub(r'def (\w+)\((.*?)\):', r'def \1(\2) -> int:', code)
        return code
    
    @staticmethod
    def _reorder_imports(code):
        """Reorder import statements."""
        lines = code.split('\n')
        import_lines = [line for line in lines if line.strip().startswith('import') or line.strip().startswith('from')]
        other_lines = [line for line in lines if not (line.strip().startswith('import') or line.strip().startswith('from'))]
        
        if import_lines:
            random.shuffle(import_lines)
            return '\n'.join(import_lines + other_lines)
        return code
    
    @staticmethod
    def _add_docstrings(code):
        """Add simple docstrings to functions."""
        if 'def ' in code and '"""' not in code:
            lines = code.split('\n')
            new_lines = []
            for i, line in enumerate(lines):
                new_lines.append(line)
                if 'def ' in line and random.random() < 0.2:
                    indent = len(line) - len(line.lstrip())
                    new_lines.append(' ' * (indent + 4) + '"""Function docstring."""')
            return '\n'.join(new_lines)
        return code

def load_bigclonebench_data_robust(subset_size=20000, max_seq_length=512, validation_split=0.15):
    """
    Robust BigCloneBench data loading with comprehensive error handling and augmentation.
    """
    try:
        from datasets import load_dataset
        
        print("Loading BigCloneBench dataset...")
        
        # Load dataset
        dataset = load_dataset(
            "code_x_glue_cc_clone_detection_big_clone_bench",
            trust_remote_code=True,
            split={
                'train': f'train[:{subset_size}]', 
                'test': f'test[:{subset_size//4}]'
            }
        )
        
        train_data = dataset['train']
        test_data = dataset['test']
        
        print(f"Dataset loaded successfully!")
        print(f"   Train samples: {len(train_data)}")
        print(f"   Test samples: {len(test_data)}")
        
        # Initialize advanced tokenizer
        print("Initializing CodeBERT tokenizer...")
        tokenizer = AdvancedCodeTokenizer(max_length=max_seq_length)
        
        def process_split_with_augmentation(data_split, split_name, apply_augmentation=False):
            """Process dataset split with optional augmentation."""
            print(f"Processing {split_name} data...")
            
            codes1 = []
            codes2 = []
            labels = []
            
            augmenter = CodeAugmenter() if apply_augmentation else None
            
            for i, example in enumerate(data_split):
                if i % 5000 == 0 and i > 0:
                    print(f"   Processed {i}/{len(data_split)} examples")
                
                # Clean and prepare code snippets
                code1 = str(example['func1']).strip()
                code2 = str(example['func2']).strip()
                
                # Apply augmentation to training data only
                if apply_augmentation and augmenter:
                    if random.random() < 0.3:  # Augment 30% of training data
                        code1 = augmenter.augment_code(code1)
                    if random.random() < 0.3:
                        code2 = augmenter.augment_code(code2)
                
                codes1.append(code1)
                codes2.append(code2)
                labels.append(float(example['label']))
            
            print(f"   Tokenizing {len(codes1)} code pairs...")
            
            # Tokenize in batches for memory efficiency
            batch_size = 1000
            code1_tokens_list = []
            code2_tokens_list = []
            
            for i in range(0, len(codes1), batch_size):
                batch_codes1 = codes1[i:i+batch_size]
                batch_codes2 = codes2[i:i+batch_size]
                
                batch_tokens1 = tokenizer.tokenize_and_pad(batch_codes1)
                batch_tokens2 = tokenizer.tokenize_and_pad(batch_codes2)
                
                code1_tokens_list.append(batch_tokens1)
                code2_tokens_list.append(batch_tokens2)
            
            # Concatenate all batches
            code1_tokens = torch.cat(code1_tokens_list, dim=0)
            code2_tokens = torch.cat(code2_tokens_list, dim=0)
            labels_tensor = torch.tensor(labels, dtype=torch.float)
            
            print(f"{split_name} processing complete!")
            print(f"   Shape: {code1_tokens.shape}")
            print(f"   Labels: {len(labels)} ({sum(labels)} positive)")
            
            return CodePairDataset(code1_tokens, code2_tokens, labels_tensor)
        
        # Process training data with augmentation
        train_dataset = process_split_with_augmentation(train_data, "Training", apply_augmentation=True)
        
        # Split training data into train/validation
        train_size = int((1 - validation_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        # Process test data without augmentation
        test_dataset = process_split_with_augmentation(test_data, "Test", apply_augmentation=False)
        
        print(f"\nFinal dataset split:")
        print(f"   Training: {len(train_dataset)}")
        print(f"   Validation: {len(val_dataset)}")
        print(f"   Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset, tokenizer
        
    except Exception as e:
        print(f"BigCloneBench loading failed: {e}")
        print("Falling back to enhanced synthetic data...")
        return load_enhanced_synthetic_data(subset_size, max_seq_length, validation_split)

def load_enhanced_synthetic_data(subset_size=20000, max_seq_length=512, validation_split=0.15):
    """
    Enhanced synthetic data generation with advanced patterns and augmentation.
    """
    print("Creating enhanced synthetic dataset with augmentation...")
    
    # Comprehensive base functions with more complexity
    base_functions = [
        # Advanced algorithms
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
        
        "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
        
        "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)",
        
        # Data structure operations
        "def heap_push(heap, item):\n    heap.append(item)\n    _sift_down(heap, 0, len(heap)-1)",
        
        "def dfs_traversal(graph, start, visited=None):\n    if visited is None:\n        visited = set()\n    visited.add(start)\n    for neighbor in graph[start]:\n        if neighbor not in visited:\n            dfs_traversal(graph, neighbor, visited)\n    return visited",
        
        # Mathematical functions
        "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
        
        "def matrix_multiply(A, B):\n    rows_A, cols_A = len(A), len(A[0])\n    rows_B, cols_B = len(B), len(B[0])\n    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]\n    for i in range(rows_A):\n        for j in range(cols_B):\n            for k in range(cols_A):\n                result[i][j] += A[i][k] * B[k][j]\n    return result",
        
        # String processing
        "def longest_common_subsequence(str1, str2):\n    m, n = len(str1), len(str2)\n    dp = [[0] * (n + 1) for _ in range(m + 1)]\n    for i in range(1, m + 1):\n        for j in range(1, n + 1):\n            if str1[i-1] == str2[j-1]:\n                dp[i][j] = dp[i-1][j-1] + 1\n            else:\n                dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n    return dp[m][n]",
    ]
    
    # Generate augmented data
    augmenter = CodeAugmenter()
    
    def create_augmented_similar_pairs(base_functions, num_pairs):
        """Create similar pairs with advanced augmentation."""
        similar_pairs = []
        
        for _ in range(num_pairs):
            func = np.random.choice(base_functions)
            
            # Apply augmentation to create similar version
            similar_func = augmenter.augment_code(func, augmentation_prob=0.8)
            similar_pairs.append((func, similar_func, 1))
        
        return similar_pairs
    
    def create_dissimilar_pairs(base_functions, num_pairs):
        """Create dissimilar pairs."""
        dissimilar_pairs = []
        
        for _ in range(num_pairs):
            func1, func2 = np.random.choice(base_functions, 2, replace=False)
            dissimilar_pairs.append((func1, func2, 0))
        
        return dissimilar_pairs
    
    # Generate datasets
    train_size = int(subset_size * (1 - validation_split) * 0.75)
    val_size = int(subset_size * validation_split * 0.75)
    test_size = subset_size - train_size - val_size
    
    print(f"   Generating {train_size} training pairs...")
    train_similar = create_augmented_similar_pairs(base_functions, train_size // 2)
    train_dissimilar = create_dissimilar_pairs(base_functions, train_size // 2)
    train_data = train_similar + train_dissimilar
    
    print(f"   Generating {val_size} validation pairs...")
    val_similar = create_augmented_similar_pairs(base_functions, val_size // 2)
    val_dissimilar = create_dissimilar_pairs(base_functions, val_size // 2)
    val_data = val_similar + val_dissimilar
    
    print(f"   Generating {test_size} test pairs...")
    test_similar = create_augmented_similar_pairs(base_functions, test_size // 2)
    test_dissimilar = create_dissimilar_pairs(base_functions, test_size // 2)
    test_data = test_similar + test_dissimilar
    
    # Process data
    def extract_and_process(data, split_name):
        codes1 = [item[0] for item in data]
        codes2 = [item[1] for item in data]
        labels = [item[2] for item in data]
        
        try:
            tokenizer = AdvancedCodeTokenizer(max_length=max_seq_length)
            code1_tokens = tokenizer.tokenize_and_pad(codes1)
            code2_tokens = tokenizer.tokenize_and_pad(codes2)
        except:
            # Fallback tokenizer
            tokenizer = SimpleTokenizer(num_words=10000)
            tokenizer.fit_on_texts(codes1 + codes2)
            
            code1_sequences = tokenizer.texts_to_sequences(codes1)
            code2_sequences = tokenizer.texts_to_sequences(codes2)
            
            code1_tokens = torch.tensor(pad_sequences(code1_sequences, maxlen=max_seq_length), dtype=torch.long)
            code2_tokens = torch.tensor(pad_sequences(code2_sequences, maxlen=max_seq_length), dtype=torch.long)
        
        labels_tensor = torch.tensor(labels, dtype=torch.float)
        return CodePairDataset(code1_tokens, code2_tokens, labels_tensor), tokenizer
    
    train_dataset, tokenizer = extract_and_process(train_data, "training")
    val_dataset, _ = extract_and_process(val_data, "validation")
    test_dataset, _ = extract_and_process(test_data, "test")
    
    print(f"Enhanced synthetic dataset created with augmentation!")
    
    return train_dataset, val_dataset, test_dataset, tokenizer

# Simple tokenizer fallback
class SimpleTokenizer:
    def __init__(self, num_words=None):
        self.num_words = num_words
        self.word_index = {}
        self.vocab_size = 0

    def fit_on_texts(self, texts):
        word_counts = Counter()
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            word_counts.update(words)

        sorted_words = word_counts.most_common(self.num_words - 1 if self.num_words else None)
        for idx, (word, _) in enumerate(sorted_words, 1):
            self.word_index[word] = idx
        self.vocab_size = len(self.word_index) + 1

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            sequence = [self.word_index.get(word, 0) for word in words]
            sequences.append(sequence)
        return sequences

def pad_sequences(sequences, maxlen, padding='post'):
    padded = []
    for seq in sequences:
        if len(seq) > maxlen:
            padded_seq = seq[:maxlen]
        else:
            padded_seq = seq + [0] * (maxlen - len(seq))
        padded.append(padded_seq)
    return np.array(padded)
