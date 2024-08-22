import numpy as np
import pandas as pd
from .dataset_class import Dataset

# Hidden helper functions
def _normalize_column(column):
    min_val = np.min(column)
    max_val = np.max(column)
    return (column - min_val) / (max_val - min_val)

def _standardize_column(column):
    mean_val = np.mean(column)
    std_val = np.std(column)
    return (column - mean_val) / std_val

# Public utility functions
def normalize(dataset: Dataset) -> Dataset:
    """Returns a new Dataset object with normalized numerical columns."""
    normalized_data = dataset.data.apply(
        lambda col: _normalize_column(col) if np.issubdtype(col.dtype, np.number) else col
    )
    return Dataset(normalized_data, class_column=dataset.class_column)

def standardize(dataset: Dataset) -> Dataset:
    """Returns a new Dataset object with standardized numerical columns."""
    standardized_data = dataset.data.apply(
        lambda col: _standardize_column(col) if np.issubdtype(col.dtype, np.number) else col
    )
    return Dataset(standardized_data, class_column=dataset.class_column)

def column_variance(dataset: Dataset, column_name: str = None) -> pd.Series:
    """
    Returns the variance of the specified column in the Dataset. If no column is specified,
    returns the variance of all numeric columns.

    :param dataset: The Dataset object.
    :param column_name: The name of the column for which to calculate the variance. 
                        If None, returns variance for all numeric columns.
    :return: A pandas Series with the variance of the specified column or all numeric columns.
    """
    if column_name:
        if column_name not in dataset.data.columns:
            raise ValueError(f"Column '{column_name}' is not in the dataset.")
        
        column = dataset.data[column_name]
        
        if not np.issubdtype(column.dtype, np.number):
            raise TypeError(f"Column '{column_name}' is not numeric, so variance cannot be calculated.")
        
        return np.var(column)
    
    # If no column_name is specified, return variance of all numeric columns
    numeric_columns = dataset.data.select_dtypes(include=[np.number])
    return numeric_columns.var(ddof=0)

#=================================================
# EQUAL WIDTH DISCRETIZATION 
#=================================================

#Hidden Helper Function for equal width discretization (ew)

def _discretize_ew(x, num_bins):
    # Check if x is numeric and a vector (1D array)
    if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number):
        raise ValueError("x must be a numeric vector (1D array).")
    
    # Check if num_bins is a positive integer
    if not isinstance(num_bins, int) or num_bins <= 0:
        raise ValueError("num_bins must be a positive integer.")
    
    # Generate the cut points
    cut_points = np.linspace(np.min(x), np.max(x), num_bins + 1)
    
    # Replace the first and last cut points with -Inf and Inf respectively
    cut_points[0] = -np.inf
    cut_points[-1] = np.inf
    
    # Create the labels for the intervals
    interval_labels = [f"({cut_points[i]:.3f}, {cut_points[i+1]:.3f}]" for i in range(len(cut_points) - 1)]
    
    # Use pd.cut to assign each value to its corresponding interval
    x_discretized = pd.cut(x, bins=cut_points, labels=interval_labels, include_lowest=True, right=True)
        
    return x_discretized, cut_points

# Public utility functions
def discretize_ew(dataset: Dataset, num_bins: int, include_cut_points: bool = False) -> (Dataset, dict):
    '''
    Discretizes the numeric columns of a Dataset into specified number of bins using equal-width binning.
    Returns a new Dataset object with discretized data and optionally a dictionary of cut points.

    :param dataset: The Dataset object.
    :param num_bins: The number of bins to discretize each numeric column into.
    :param include_cut_points: If True, also returns a dictionary of cut points for each column.
    :return: A tuple containing a new Dataset object with discretized data and optionally a dictionary of cut points.
    '''
    df = dataset.get_dataframe()
    
    # Apply discretization to each column
    discretized_df = df.apply(lambda col: _discretize_ew(col.values, num_bins)[0] if np.issubdtype(col.dtype, np.number) else col)
        
    # Get cut points for each column
    cut_points_dict = {}
    if include_cut_points:
        cut_points_dict = {
            col: _discretize_ew(df[col].values, num_bins)[1]
            for col in df.columns if np.issubdtype(df[col].dtype, np.number)
        }
    
    # Create a new Dataset object with discretized data
    discretized_dataset = Dataset(discretized_df, class_column=dataset.class_column)
    
    if include_cut_points:
        return discretized_dataset, cut_points_dict
    else:
        return discretized_dataset

#=================================================
# EQUAL FREQUENCY DISCRETIZATION 
#=================================================
#Hidden Helper Function for equal frequency discretization (ef)

def _discretize_ef(x, num_bins):
    # Check that x is a numeric array
    if not isinstance(x, (np.ndarray, list)) or not np.issubdtype(np.array(x).dtype, np.number):
        raise ValueError("x must be a numeric array or list.")
    
    # Convert x to a NumPy array
    x = np.asarray(x)
    
    # Check that num_bins is a positive integer
    if not isinstance(num_bins, int) or num_bins <= 0:
        raise ValueError("num_bins must be a positive integer.")
    
    # Calculate cut points using quantiles
    quantiles = np.linspace(0, 1, num_bins + 1)
    cut_points = np.percentile(x, quantiles * 100)
    
    # Replace the first and last cut points with -Inf and Inf
    cut_points[0] = -np.inf
    cut_points[-1] = np.inf
    
    # Initialize the interval labels
    interval_labels = [f"({cut_points[i]:.3f}, {cut_points[i+1]:.3f}]" for i in range(len(cut_points) - 1)]
    
    # Assign each value to its corresponding interval
    x_discretized = pd.cut(x, bins=cut_points, labels=interval_labels, include_lowest=True, right=True)
    
    # Return the result
    return x_discretized, cut_points


# Public utility functions
def discretize_ef(dataset: Dataset, num_bins: int, include_cut_points: bool = False) -> (Dataset, dict):
    '''
    Discretizes the numeric columns of a Dataset into specified number of bins using equal-frequency binning.
    Returns a new Dataset object with discretized data and optionally a dictionary of cut points.

    :param dataset: The Dataset object.
    :param num_bins: The number of bins to discretize each numeric column into.
    :param include_cut_points: If True, also returns a dictionary of cut points for each column.
    :return: A tuple containing a new Dataset object with discretized data and optionally a dictionary of cut points.
    '''
    df = dataset.get_dataframe()
    
    # Apply discretization to each column
    discretized_df = df.apply(lambda col: _discretize_ef(col.values, num_bins)[0] if np.issubdtype(col.dtype, np.number) else col)
           
    # Get cut points for each column
    cut_points_dict = {}
    if include_cut_points:
        cut_points_dict = {
            col: _discretize_ef(df[col].values, num_bins)[1]
            for col in df.columns if np.issubdtype(df[col].dtype, np.number)
        }
    
    # Create a new Dataset object with discretized data
    discretized_dataset = Dataset(discretized_df, class_column=dataset.class_column)
    
    if include_cut_points:
        return discretized_dataset, cut_points_dict
    else:
        return discretized_dataset


def _calculate_entropy(series):
    """Calculate the normalized entropy and entropy value of a pandas Series."""
    if not isinstance(series.dtype, pd.CategoricalDtype):
        raise ValueError("The input should be a categorical Series.")
    
    # Create a frequency table
    freq = series.value_counts()
    total = freq.sum()
    
    # Calculate probabilities
    prob = freq / total

    # If all the values are the same, the entropy is 0
    if len(freq) == 1:
        return 0.0, 0.0
    
    # Calculate entropy
    entropy_value = -np.sum(prob * np.log2(prob))
    
    # Normalize entropy
    norm_entropy = entropy_value / np.log2(len(freq))
    
    return entropy_value, norm_entropy

# Public utility functions
def calculate_entropy(dataset, column_name=None):
    """
    Calculate entropy for a specified column or for all categorical columns.

    :param dataset: The Dataset object.
    :param column_name: Optional name of the column to calculate entropy for. If None, calculate for all categorical columns.
    :return: A dictionary with column names as keys and entropy values as values.
    """
    df = dataset.get_dataframe()

    result = {}
    
    if column_name:
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' is not in the dataset.")
        
        column = df[column_name]
        
        if not isinstance(column.dtype, pd.CategoricalDtype):
            # Convert to categorical temporarily
            column = column.astype('category')
            print(f"Column '{column_name}' is not categorical. Converting temporally to categorical for entropy calculation.")
        
        entropy_value, norm_entropy = _calculate_entropy(column)
        result[column_name] = {'entropy': entropy_value, 'normalized_entropy': norm_entropy}
    
    else:
        for col in df.columns:
            column = df[col]
            if not isinstance(column.dtype, pd.CategoricalDtype):
                # Convert to categorical temporarily
                column = column.astype('category')
                print(f"Column '{col}' is not categorical. Converting temporally to categorical for entropy calculation.")
            
            if isinstance(column.dtype, pd.CategoricalDtype):
                entropy_value, norm_entropy = _calculate_entropy(column)
                result[col] = {'entropy': entropy_value, 'normalized_entropy': norm_entropy}
    
    return result


def calculate_roc_auc(dataset: Dataset, prob_col: str, label_col: str):
    """
    Function to calculate TPR and FPR from a Dataset object with columns: probabilities and true labels.
    Returns a dictionary with roc_points and AUC.
    
    :param dataset: The Dataset object.
    :param prob_col: Name of the column containing the predicted probabilities.
    :param label_col: Name of the column containing the true labels.
    """
    # Extract the DataFrame from the Dataset object
    df = dataset.get_dataframe()

    # Extract the relevant columns
    probabilities = df[prob_col].values
    true_labels = df[label_col].values
    
    thresholds = np.unique(np.sort(probabilities)[::-1])
    roc_points = []

    for threshold in thresholds:
        predicted = np.where(probabilities >= threshold, 1, 0)
                        
        TP = np.sum((predicted == 1) & (true_labels == 1))
        FN = np.sum((predicted == 0) & (true_labels == 1))
        FP = np.sum((predicted == 1) & (true_labels == 0))
        TN = np.sum((predicted == 0) & (true_labels == 0))
        
        TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
        
        roc_points.append([FPR, TPR])
    
    roc_points = np.array(roc_points)
    
    # Sort by FPR and TPR
    sorted_indices = np.lexsort((roc_points[:, 1], roc_points[:, 0]))
    roc_points = roc_points[sorted_indices]
    
    # Vectorized AUC calculation
    x_diff = np.diff(roc_points[:, 0])
    y_avg = (roc_points[1:, 1] + roc_points[:-1, 1]) / 2
    auc = np.sum(x_diff * y_avg)
    ##eso es igual a 
    #auc = 0
    #for i in range(1, len(roc_points)):
       # auc += (roc_points[i, 0] - roc_points[i - 1, 0]) * (roc_points[i, 1] + roc_points[i - 1, 1]) / 2
    print('Roc points: [FPR, TPR]')
    return {'roc_points': roc_points, 'auc': auc}