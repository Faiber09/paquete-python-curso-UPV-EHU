import pytest
import numpy as np
import pandas as pd
import seaborn as sns
from datasetpack import Dataset, normalize, standardize, column_variance, discretize_ew, discretize_ef
from datasetpack import calculate_entropy, calculate_roc_auc
from datasetpack import plot_entropy, plot_roc_curve, plot_correlation_matrix

@pytest.fixture
def dataset():
    """Set up a basic Dataset object for testing."""
    data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [5, 4, 3, 2, 1],
        'C': ['cat', 'dog', 'cat', 'cat', 'dog'],
        'D': [0.1, 0.2, 0.3, 0.4, 0.5]
    })
    return Dataset(data, class_column='C')

def test_normalize(dataset):
    """Test normalization of numeric columns."""
    normalized_dataset = normalize(dataset)
    normalized_data = normalized_dataset.get_dataframe()
    assert normalized_data['A'].min() == pytest.approx(0.0)
    assert normalized_data['A'].max() == pytest.approx(1.0)
    assert normalized_data['B'].min() == pytest.approx(0.0)
    assert normalized_data['B'].max() == pytest.approx(1.0)

def test_standardize(dataset):
    """Test standardization of numeric columns."""
    standardized_dataset = standardize(dataset)
    standardized_data = standardized_dataset.get_dataframe()
    assert standardized_data['A'].mean() == pytest.approx(0.0, abs=1e-6)
    assert standardized_data['A'].std(ddof=0) == pytest.approx(1.0, abs=1e-6)

def test_column_variance(dataset):
    """Test variance calculation for a specific column."""
    variance = column_variance(dataset, column_name='A')
    expected_variance = np.var(dataset.get_dataframe()['A'])
    assert variance == pytest.approx(expected_variance)

def test_discretize_ew(dataset):
    """Test equal-width discretization."""
    discretized_dataset, cut_points = discretize_ew(dataset, num_bins=3, include_cut_points=True)
    discretized_data = discretized_dataset.get_dataframe()
    
    # Check if the column 'A' has been discretized
    assert isinstance(discretized_data['A'].dtype, pd.CategoricalDtype)
    
    # Verify that the column 'A' has categories (bins) after discretization
    categories = discretized_data['A'].cat.categories
    assert len(categories) == 3  # Expecting 4 categories (bins), including the intervals with -inf and inf
    
    # Simple check for cut points length and boundary values
    assert len(cut_points['A']) == 4
    assert len(cut_points) == 3  # 3 bins should produce 4 cut points
    assert cut_points['A'][0] == -np.inf
    assert cut_points['A'][-1] == np.inf

def test_discretize_ef(dataset):
    """Test equal-frequency discretization."""
    discretized_dataset, cut_points = discretize_ef(dataset, num_bins=3, include_cut_points=True)
    discretized_data = discretized_dataset.get_dataframe()
    
    # Check if the column 'A' has been discretized
    assert isinstance(discretized_data['A'].dtype, pd.CategoricalDtype)
    
    # Verify that the column 'A' has categories (bins) after discretization
    categories = discretized_data['A'].cat.categories
    assert len(categories) > 0  # Ensure there are categories
    
    # Verify cut points - they should include -inf and inf as per the function logic
    assert np.all(np.isfinite(cut_points['A'][1:-1]))  # Check that cut points excluding -inf and inf are finite
    assert cut_points['A'][0] == -np.inf
    assert cut_points['A'][-1] == np.inf


def test_calculate_entropy(dataset):
    """Test entropy calculation."""
    entropy_results = calculate_entropy(dataset)
    assert 'C' in entropy_results
    assert entropy_results['C']['entropy'] > 0

def test_calculate_roc_auc():
    """Test ROC AUC calculation."""
    data = pd.DataFrame({
        'prob': [0.1, 0.4, 0.35, 0.8],
        'label': [0, 0, 1, 1]
    })
    dataset = Dataset(data, class_column='label')
    roc_results = calculate_roc_auc(dataset, prob_col='prob', label_col='label')
    assert 'auc' in roc_results
    assert 0.0 <= roc_results['auc'] <= 1.0

def test_plot_entropy(dataset):
    """Test plotting entropy (plot does not raise exceptions)."""
    result = plot_entropy(dataset, normalized=True)
    assert result['message'] == "Plot generated successfully."

def test_plot_roc_curve():
    """Test plotting ROC curve (plot does not raise exceptions)."""
    data = pd.DataFrame({
        'prob': [0.1, 0.4, 0.35, 0.8],
        'label': [0, 0, 1, 1]
    })
    dataset = Dataset(data, class_column='label')
    result = plot_roc_curve(dataset, prob_col='prob', label_col='label')
    assert result['message'] == "ROC curve plotted successfully."

def test_plot_correlation_matrix(dataset):
    """Test plotting correlation matrix (plot does not raise exceptions)."""
    plot_correlation_matrix(dataset)
    # No assertion needed, just ensure it does not raise exceptions