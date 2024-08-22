from .dataset_class import Dataset
from .dataset_utils import calculate_entropy, calculate_roc_auc

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def plot_entropy(dataset: Dataset, normalized: bool = True):
    """
    Plot entropy for each categorical column in the Dataset.

    :param dataset: The Dataset object.
    :param normalized: Boolean flag to determine whether to plot normalized entropy (True) or raw entropy value (False).
    """
    # Calculate entropy for each column
    entropy_results = calculate_entropy(dataset)

    # Extract column names and entropy values for plotting
    columns = list(entropy_results.keys())
    if normalized:
        entropies = [result['normalized_entropy'] for result in entropy_results.values()]
        ylabel = 'Normalized Entropy'
    else:
        entropies = [result['entropy'] for result in entropy_results.values()]
        ylabel = 'Entropy'

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.bar(columns, entropies, color='skyblue')
    plt.xlabel('Columns')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} by Column')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    return {'message': "Plot generated successfully.", 'entropies': entropies, 'column_names': columns}


def plot_roc_curve(dataset: Dataset, prob_col: str, label_col: str):
    """
    Function to plot the ROC curve using the Dataset object and column names for probabilities and true labels.
    
    :param dataset: The Dataset object.
    :param prob_col: Name of the column containing the predicted probabilities.
    :param label_col: Name of the column containing the true labels.
    """
    # Calculate ROC points and AUC
    roc_results = calculate_roc_auc(dataset, prob_col, label_col)
    roc_points = roc_results['roc_points']
    auc = roc_results['auc']
    
    plt.figure()
    
    # Plot ROC curve with a blue line
    plt.plot(roc_points[:, 0], roc_points[:, 1], 'b-', label='ROC curve', linewidth=2)
    plt.scatter(roc_points[:, 0], roc_points[:, 1], color='red')  # Scatter points in red
    
    # Plotting the diagonal line representing a random classifier
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')  # Diagonal line in dashed black
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    
    if auc is not None and not np.isnan(auc):
        plt.text(0.6, 0.2, f'AUC = {round(auc, 2)}', color='black')
    
    plt.legend()
    plt.show()

    return {'message': "ROC curve plotted successfully.", 'auc': auc}


def plot_correlation_matrix(dataset: Dataset):
    """
    Visualizes the correlation matrix of numeric columns within the DataFrame in the Dataset object.
    Non-numeric columns are ignored, and a message is printed about their exclusion.

    Parameters:
    - dataset: Dataset object containing the DataFrame.

    Returns:
    - None: Displays the correlation matrix heatmap.
    """
    # Extract the DataFrame from the Dataset object
    data = dataset.get_dataframe()
    
    # Identify numeric and non-numeric columns
    numeric_columns = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
    non_numeric_columns = [col for col in data.columns if not pd.api.types.is_numeric_dtype(data[col])]
    
    if non_numeric_columns:
        print(f"Non-numeric columns detected and ignored: {', '.join(non_numeric_columns)}")
    
    # Select only numeric columns for correlation calculation
    numeric_data = data[numeric_columns]
    
    # Calculate the correlation matrix
    cor_matrix = numeric_data.corr()
    
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cor_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                vmin=-1, vmax=1, center=0, linewidths=0.5, square=True)
    
    plt.title('Correlation Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.show()