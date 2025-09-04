"""
Comprehensive Evaluation Module for Code Clone Detection
Implements detailed evaluation with multiple metrics and analysis tools
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import seaborn as sns
from tqdm import tqdm
import json
import os

from dataset import load_bigclonebench_data_robust
from models import ImprovedSiameseNetwork, load_pretrained_model

class ComprehensiveEvaluator:
    """
    Advanced evaluator with detailed metrics and visualization capabilities.
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate_dataset(self, test_loader, threshold=0.5):
        """
        Comprehensive evaluation on test dataset.
        
        Args:
            test_loader: DataLoader for test data
            threshold: Classification threshold
            
        Returns:
            dict: Comprehensive evaluation results
        """
        print(f"Starting comprehensive evaluation...")
        print(f"Classification threshold: {threshold}")
        
        y_true = []
        y_scores = []
        similarities = []
        
        # Collect predictions
        with torch.no_grad():
            for code1, code2, labels in tqdm(test_loader, desc="Evaluating"):
                code1 = code1.to(self.device)
                code2 = code2.to(self.device)
                
                similarity = self.model(code1, code2)
                
                y_true.extend(labels.cpu().numpy())
                similarities.extend(similarity.cpu().numpy())
        
        y_true = np.array(y_true)
        similarities = np.array(similarities)
        y_pred = (similarities >= threshold).astype(int)
        
        # Compute comprehensive metrics
        results = self._compute_detailed_metrics(y_true, y_pred, similarities)
        
        # Add threshold information
        results['threshold'] = threshold
        results['total_samples'] = len(y_true)
        
        return results
    
    def _compute_detailed_metrics(self, y_true, y_pred, y_scores):
        """Compute comprehensive evaluation metrics."""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Advanced metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # ROC metrics
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall metrics
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        
        # Similarity score analysis
        similar_scores = y_scores[y_true == 1]
        dissimilar_scores = y_scores[y_true == 0]
        
        return {
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'sensitivity': float(sensitivity),
            'roc_auc': float(roc_auc),
            'avg_precision': float(avg_precision),
            'confusion_matrix': {
                'tn': int(tn), 'fp': int(fp), 
                'fn': int(fn), 'tp': int(tp)
            },
            'similarity_stats': {
                'mean': float(np.mean(y_scores)),
                'std': float(np.std(y_scores)),
                'min': float(np.min(y_scores)),
                'max': float(np.max(y_scores)),
                'similar_mean': float(np.mean(similar_scores)) if len(similar_scores) > 0 else 0,
                'similar_std': float(np.std(similar_scores)) if len(similar_scores) > 0 else 0,
                'dissimilar_mean': float(np.mean(dissimilar_scores)) if len(dissimilar_scores) > 0 else 0,
                'dissimilar_std': float(np.std(dissimilar_scores)) if len(dissimilar_scores) > 0 else 0,
            },
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            },
            'precision_recall_curve': {
                'precision': precision_curve.tolist(),
                'recall': recall_curve.tolist()
            }
        }
    
    def threshold_analysis(self, test_loader, thresholds=None):
        """
        Analyze performance across different thresholds.
        
        Args:
            test_loader: Test data loader
            thresholds: List of thresholds to test
            
        Returns:
            dict: Results for each threshold
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)
        
        print(f"Analyzing {len(thresholds)} different thresholds...")
        
        # Collect all predictions once
        y_true = []
        similarities = []
        
        with torch.no_grad():
            for code1, code2, labels in tqdm(test_loader, desc="Collecting predictions"):
                code1 = code1.to(self.device)
                code2 = code2.to(self.device)
                
                similarity = self.model(code1, code2)
                
                y_true.extend(labels.cpu().numpy())
                similarities.extend(similarity.cpu().numpy())
        
        y_true = np.array(y_true)
        similarities = np.array(similarities)
        
        # Test each threshold
        threshold_results = {}
        
        for threshold in thresholds:
            y_pred = (similarities >= threshold).astype(int)
            
            # Compute metrics for this threshold
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            
            threshold_results[f"{threshold:.1f}"] = {
                'threshold': float(threshold),
                'accuracy': float(accuracy),
                'f1_score': float(f1),
                'precision': float(precision),
                'recall': float(recall)
            }
        
        return threshold_results
    
    def plot_evaluation_results(self, results, save_dir="evaluation_plots"):
        """
        Create comprehensive evaluation plots.
        
        Args:
            results: Evaluation results dictionary
            save_dir: Directory to save plots
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Confusion Matrix
        cm = np.array([[results['confusion_matrix']['tn'], results['confusion_matrix']['fp']],
                       [results['confusion_matrix']['fn'], results['confusion_matrix']['tp']]])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Dissimilar', 'Similar'],
                    yticklabels=['Dissimilar', 'Similar'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=300)
        plt.close()
        
        # 2. ROC Curve
        plt.figure(figsize=(8, 6))
        fpr = results['roc_curve']['fpr']
        tpr = results['roc_curve']['tpr']
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {results["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/roc_curve.png", dpi=300)
        plt.close()
        
        # 3. Precision-Recall Curve
        plt.figure(figsize=(8, 6))
        precision = results['precision_recall_curve']['precision']
        recall = results['precision_recall_curve']['recall']
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {results["avg_precision"]:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/precision_recall_curve.png", dpi=300)
        plt.close()
        
        print(f"Evaluation plots saved to {save_dir}/")
    
    def print_detailed_report(self, results):
        """Print comprehensive evaluation report."""
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("="*80)
        
        # Overall Performance
        print(f"\nOverall Performance:")
        print(f"   Test Accuracy:  {results['accuracy']*100:.2f}%")
        print(f"   F1 Score:       {results['f1_score']:.4f}")
        print(f"   Precision:      {results['precision']:.4f}")
        print(f"   Recall:         {results['recall']:.4f}")
        print(f"   Specificity:    {results['specificity']:.4f}")
        print(f"   ROC AUC:        {results['roc_auc']:.4f}")
        print(f"   Avg Precision:  {results['avg_precision']:.4f}")
        
        # Confusion Matrix Analysis
        cm = results['confusion_matrix']
        total = cm['tp'] + cm['tn'] + cm['fp'] + cm['fn']
        print(f"\nConfusion Matrix Analysis:")
        print(f"   True Positives:  {cm['tp']} ({cm['tp']/total*100:.1f}%)")
        print(f"   True Negatives:  {cm['tn']} ({cm['tn']/total*100:.1f}%)")
        print(f"   False Positives: {cm['fp']} ({cm['fp']/total*100:.1f}%)")
        print(f"   False Negatives: {cm['fn']} ({cm['fn']/total*100:.1f}%)")
        
        # Similarity Score Statistics
        stats = results['similarity_stats']
        print(f"\nSimilarity Score Statistics:")
        print(f"   Overall Mean:    {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"   Range:           [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"   Similar pairs:   {stats['similar_mean']:.4f} ± {stats['similar_std']:.4f}")
        print(f"   Dissimilar pairs: {stats['dissimilar_mean']:.4f} ± {stats['dissimilar_std']:.4f}")
        
        # Performance Assessment
        print(f"\nPerformance Assessment:")
        if results['f1_score'] >= 0.95:
            assessment = "EXCELLENT"
        elif results['f1_score'] >= 0.90:
            assessment = "VERY GOOD"
        elif results['f1_score'] >= 0.80:
            assessment = "GOOD"
        elif results['f1_score'] >= 0.70:
            assessment = "FAIR"
        else:
            assessment = "NEEDS IMPROVEMENT"
        
        print(f"   Overall Rating: {assessment}")
        
        # Recommendations
        print(f"\nRecommendations:")
        if results['precision'] < 0.8:
            print("   - Consider increasing threshold to reduce false positives")
        if results['recall'] < 0.8:
            print("   - Consider decreasing threshold to reduce false negatives")
        if stats['similar_std'] == 0.0:
            print("   - WARNING: All similar pairs have identical scores (possible overfitting)")
        if results['roc_auc'] < 0.8:
            print("   - Model discrimination ability needs improvement")
        
        print("="*80)

def main():
    """Main evaluation function."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    print("Loading test dataset...")
    try:
        _, _, test_dataset, tokenizer = load_bigclonebench_data_robust(
            subset_size=20000,
            max_seq_length=512
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"Test dataset loaded: {len(test_dataset)} samples")
    
    # Load trained model
    model_path = "siamese_epoch_1_f1_0.9653.pth"  # Update with your model path
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please train a model first or update the model path.")
        return
    
    try:
        model = load_pretrained_model(
            model_path=model_path,
            vocab_size=tokenizer.vocab_size,
            device=device
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator(model, device)
    
    # Comprehensive evaluation
    results = evaluator.evaluate_dataset(test_loader, threshold=0.5)
    
    # Print detailed report
    evaluator.print_detailed_report(results)
    
    # Threshold analysis
    print("\nPerforming threshold analysis...")
    threshold_results = evaluator.threshold_analysis(test_loader)
    
    # Find optimal threshold
    best_f1 = 0
    best_threshold = 0.5
    for thresh_str, metrics in threshold_results.items():
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_threshold = metrics['threshold']
    
    print(f"\nOptimal threshold analysis:")
    print(f"   Best threshold: {best_threshold:.1f}")
    print(f"   Best F1 score: {best_f1:.4f}")
    
    # Save results
    results['threshold_analysis'] = threshold_results
    results['optimal_threshold'] = best_threshold
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation results saved to: evaluation_results.json")
    
    # Generate plots
    try:
        evaluator.plot_evaluation_results(results)
    except Exception as e:
        print(f"Error generating plots: {e}")

if __name__ == "__main__":
    main()
