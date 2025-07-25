#!/usr/bin/env python3
"""
Data Visualization Script for Rock vs Mines Classification
This script provides various visualizations to understand the dataset better.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_sample_data():
    """
    Create sample data for visualization demonstration.
    In a real scenario, you would load your actual dataset here.
    """
    np.random.seed(42)
    n_samples = 208
    n_features = 60
    
    # Generate sample data similar to sonar signals
    data = np.random.randn(n_samples, n_features)
    labels = np.random.choice(['R', 'M'], n_samples, p=[0.5, 0.5])
    
    return pd.DataFrame(data, columns=[f'feature_{i}' for i in range(n_features)]), labels

def plot_feature_distribution(data, labels, n_features=10):
    """Plot distribution of first n features by class."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i in range(min(n_features, len(data.columns))):
        feature = data.columns[i]
        rock_data = data[labels == 'R'][feature]
        mine_data = data[labels == 'M'][feature]
        
        axes[i].hist(rock_data, alpha=0.7, label='Rock', bins=20)
        axes[i].hist(mine_data, alpha=0.7, label='Mine', bins=20)
        axes[i].set_title(f'{feature}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_correlation_matrix(data, n_features=20):
    """Plot correlation matrix of features."""
    corr_matrix = data.iloc[:, :n_features].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_pca_visualization(data, labels):
    """Plot PCA visualization of the data."""
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Apply PCA
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)
    
    # Create DataFrame for plotting
    pca_df = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])
    pca_df['label'] = labels
    
    plt.figure(figsize=(10, 8))
    colors = {'R': 'blue', 'M': 'red'}
    
    for label in ['R', 'M']:
        mask = pca_df['label'] == label
        plt.scatter(pca_df[mask]['PC1'], pca_df[mask]['PC2'], 
                   c=colors[label], label=label, alpha=0.7, s=50)
    
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('PCA Visualization of Rock vs Mines Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pca_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_class_distribution(labels):
    """Plot the distribution of classes."""
    plt.figure(figsize=(8, 6))
    class_counts = pd.Series(labels).value_counts()
    
    colors = ['blue', 'red']
    plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    plt.title('Distribution of Rock vs Mines Classes')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run all visualizations."""
    print("Loading sample data...")
    data, labels = load_sample_data()
    
    print("Creating visualizations...")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Create all plots
    plot_class_distribution(labels)
    plot_feature_distribution(data, labels)
    plot_correlation_matrix(data)
    plot_pca_visualization(data, labels)
    
    print("All visualizations completed! Check the generated PNG files.")

if __name__ == "__main__":
    main() 