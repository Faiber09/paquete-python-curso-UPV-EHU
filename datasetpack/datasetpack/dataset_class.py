import pandas as pd

class Dataset:
    """The Dataset class enables the creation, management and operations of dataframes
    or dictionaries. It contains an optional attribute which can be the label of a supervised set. 
    We are going to implement the database preprocessing functions from the tutorials within the course"""
    def __init__(self, data, class_column: str = None):
        """
        Initializes the Dataset object.

        :param data: A pandas DataFrame or a dictionary that can be converted to a DataFrame.
        :param class_column: The name of the column that is considered the special class variable.
        """
        # Check if data is a dictionary and convert it to a DataFrame if it is
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        elif not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected 'data' to be a pandas DataFrame or a dictionary, \
            but got {type(data).__name__} instead.")
        
        self.data = data
        self.class_column = class_column

        if class_column and class_column not in data.columns:
            raise ValueError(f"'{class_column}' is not a column in the provided DataFrame.")

    def describe(self):
        """
        Returns descriptive statistics for the dataset.
        :return: A pandas DataFrame containing descriptive statistics.
        """
        return self.data.describe()

    def size(self):
        """
        Returns the size of the dataset (number of rows, number of columns).
        :return: A tuple containing the number of rows and columns.
        """
        return self.data.shape

    def columns(self):
        """
        Returns the column names of the dataset.
        :return: A list of column names.
        """
        return self.data.columns.tolist()

    def get_dataframe(self):
        """
        Returns the underlying pandas DataFrame.
        :return: The pandas DataFrame contained in the Dataset.
        """
        return self.data

    def get_class_column(self):
        """
        Returns the class column data if it exists.
        :return: A pandas Series containing the class column, or None if no class column is set.
        """
        if self.class_column:
            return self.data[self.class_column]
        return None

    def set_class_column(self, class_column: str):
        """
        Sets the class column for the Dataset.
        :param class_column: The name of the new class column.
        """
        if class_column not in self.data.columns:
            raise ValueError(f"'{class_column}' is not a column in the provided DataFrame.")
        self.class_column = class_column