#!/usr/bin/env python3
"""
Model Evaluation Script for Rock vs Mines Classification
This script provides comprehensive model evaluation metrics and cross-validation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

def generate_sample_data():
    """Generate sample data for demonstration."""
    np.random.seed(42)
    n_samples = 208
    n_features = 60
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate labels with some correlation to features
    y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.5 > 0).astype(int)
    y = ['M' if label == 1 else 'R' for label in y]
    
    return X, y

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate a single model with comprehensive metrics."""
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='M')
    recall = recall_score(y_test, y_pred, pos_label='M')
    f1 = f1_score(y_test, y_pred, pos_label='M')
    
    # Calculate ROC AUC if probabilities are available
    roc_auc = None
    if y_pred_proba is not None:
        y_test_binary = (np.array(y_test) == 'M').astype(int)
        roc_auc = roc_auc_score(y_test_binary, y_pred_proba)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=['R', 'M'])
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def plot_confusion_matrix(cm, model_name):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Rock', 'Mine'], 
                yticklabels=['Rock', 'Mine'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curves(results, X_test, y_test):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    y_test_binary = (np.array(y_test) == 'M').astype(int)
    
    for result in results:
        if result['y_pred_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test_binary, result['y_pred_proba'])
            auc = result['roc_auc']
            plt.plot(fpr, tpr, label=f"{result['model_name']} (AUC = {auc:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def cross_validate_models(models, X, y, cv=5):
    """Perform cross-validation for all models."""
    cv_results = {}
    
    for name, model in models.items():
        # Convert labels to binary for cross-validation
        y_binary = (np.array(y) == 'M').astype(int)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y_binary, cv=cv, scoring='accuracy')
        
        cv_results[name] = {
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std(),
            'cv_scores': cv_scores
        }
    
    return cv_results

def print_evaluation_summary(results, cv_results):
    """Print comprehensive evaluation summary."""
    print("=" * 60)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 60)
    
    # Create summary DataFrame
    summary_data = []
    for result in results:
        summary_data.append({
            'Model': result['model_name'],
            'Accuracy': f"{result['accuracy']:.3f}",
            'Precision': f"{result['precision']:.3f}",
            'Recall': f"{result['recall']:.3f}",
            'F1-Score': f"{result['f1_score']:.3f}",
            'ROC AUC': f"{result['roc_auc']:.3f}" if result['roc_auc'] else "N/A",
            'CV Mean': f"{cv_results[result['model_name']]['mean_accuracy']:.3f}",
            'CV Std': f"{cv_results[result['model_name']]['std_accuracy']:.3f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print("\n" + "=" * 60)

def main():
    """Main function to run comprehensive model evaluation."""
    print("Loading and preparing data...")
    X, y = generate_sample_data()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Define models to evaluate
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    print("\nEvaluating models...")
    results = []
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        result = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results.append(result)
        
        # Plot confusion matrix
        plot_confusion_matrix(result['confusion_matrix'], name)
        
        # Print detailed classification report
        print(f"\nClassification Report - {name}:")
        print(classification_report(y_test, result['y_pred'], target_names=['Rock', 'Mine']))
    
    # Perform cross-validation
    print("\nPerforming cross-validation...")
    cv_results = cross_validate_models(models, X, y, cv=5)
    
    # Print summary
    print_evaluation_summary(results, cv_results)
    
    # Plot ROC curves
    plot_roc_curves(results, X_test, y_test)
    
    print("\nEvaluation completed! Check the generated plots and summary above.")

if __name__ == "__main__":
    main() 