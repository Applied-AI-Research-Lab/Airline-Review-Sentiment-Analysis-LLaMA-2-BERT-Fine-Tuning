import os
import re
import pandas as pd
import csv
from sklearn.model_selection import train_test_split


class DatasetMethods:
    def __init__(self, dataset_path=''):
        self.dataset_path = dataset_path
        self.pre_path = '../Datasets/'

    """
    The method takes an array of column names (column_names) as input and removes empty rows.
    If the array is empty, each and every column is checked.
    Caution! The original dataset will be renamed to _original1,
         while the most current dataset will take the name of the original dataset
    """

    def remove_rows_with_empty_fields(self, column_names):
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(self.pre_path + self.dataset_path)

        # If column_names is empty, check for empty fields in all columns
        if not column_names:
            # Check for empty fields in all columns and remove corresponding rows
            df = df.dropna(how='any')
        else:
            # Check for empty fields in specified columns and remove corresponding rows
            df = df.dropna(subset=column_names, how='any')

        # Remove the .csv extension from the input file name
        file_name_without_extension = os.path.splitext(os.path.basename(self.dataset_path))[0]

        # Rename the original file by appending '_original' to its name
        original_file_path = self.pre_path + file_name_without_extension + '_original1.csv'
        os.rename(self.pre_path + self.dataset_path, original_file_path)

        # Save the modified DataFrame to a new CSV file
        df.to_csv(self.pre_path + self.dataset_path, index=False)

        return {"status": True, "data": 'Empty rows removed'}

    """    
    The method takes an array of column names (columns_to_remove) as input and removes them entirely.
    Caution! The original dataset will be renamed to _original2,
         while the most current dataset will take the name of the original dataset
    """

    def remove_columns_and_save(self, columns_to_remove):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(self.pre_path + self.dataset_path)

        # Remove the specified columns
        df = df.drop(columns=columns_to_remove, errors='ignore')

        # Remove the .csv extension from the input file name
        file_name_without_extension = os.path.splitext(os.path.basename(self.dataset_path))[0]

        # Rename the original file by appending '_original' to its name
        original_file_path = self.pre_path + file_name_without_extension + '_original2.csv'
        os.rename(self.pre_path + self.dataset_path, original_file_path)

        # Save the modified DataFrame to a new CSV file
        df.to_csv(self.pre_path + self.dataset_path, index=False)

        return {"status": True, "data": 'The provided columns removed'}

    """
    Display the unique labels in a specific column (column_name)
    """

    def display_unique_values(self, column_name):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(self.pre_path + self.dataset_path)

        # Get the unique labels
        unique_values = df[column_name].unique()

        print(f"Unique values in column '{column_name}' ({len(unique_values)}):")
        for value in unique_values:
            print(value)

    """
    The method removes rows containing a specific value (value_to_remove) in a given column (column_name)
    Caution! The original dataset will be renamed to _original4,
         while the most current dataset will take the name of the original dataset
    """

    def remove_rows_by_value(self, column_name, value_to_remove):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(self.pre_path + self.dataset_path)

        # Use boolean indexing to filter out rows with the specified value
        filtered_dataframe = df[df[column_name] != value_to_remove]

        # Remove the .csv extension from the input file name
        file_name_without_extension = os.path.splitext(os.path.basename(self.dataset_path))[0]

        # Rename the original file by appending '_original' to its name
        original_file_path = self.pre_path + file_name_without_extension + '_original4.csv'
        os.rename(self.pre_path + self.dataset_path, original_file_path)

        # Save the modified DataFrame to a new CSV file
        filtered_dataframe.to_csv(self.pre_path + self.dataset_path, index=False)

        return {"status": True, "data": f"Fields having value '{value_to_remove}' removed"}

    """
    This method cleans and standardizes the values of each row in all columns
    Caution! The original dataset will be renamed to _original3,
         while the most current dataset will take the name of the original dataset
    """

    def standardize_text(self, text):
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and punctuation
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        # Remove extra whitespaces and newline characters
        text = re.sub(r'\s+', ' ', text)

        # Remove newline characters specifically
        text = text.replace('\n', ' ').strip()

        return text

    def standardize_and_write_csv(self):
        # Rename the original dataset
        file_name_without_extension = os.path.splitext(os.path.basename(self.dataset_path))[0]
        original_file_path = f'../Datasets/{file_name_without_extension}_original3.csv'
        os.rename(f'../Datasets/{self.dataset_path}', original_file_path)

        # Write the standardized data to a new CSV file, excluding empty rows
        # Open the original file for reading and the new file for writing
        with open(original_file_path, 'r', encoding='utf-8') as infile, \
                open(f'../Datasets/{self.dataset_path}', 'w', newline='', encoding='utf-8') as outfile:

            csv_reader = csv.reader(infile)
            csv_writer = csv.writer(outfile)

            # Read and write the header
            header = next(csv_reader)
            csv_writer.writerow(header)

            # Process and write the remaining rows
            for row in csv_reader:
                standardized_row = [self.standardize_text(field) for field in row]

                # Check if the row is not empty after standardization
                if any(standardized_row):
                    csv_writer.writerow(standardized_row)

        return {"status": True, "data": "Standardization completed"}

    """
    This method creates a subset (total_rows) of the original dataset,
    ensuring the appropriate distribution of the (stratified_column) values
    Caution! The original dataset will be renamed to _original5,
         while the most current dataset will take the name of the original dataset
    """

    def create_stratified_subset(self, total_rows, stratified_column):
        # Load the dataset
        df = pd.read_csv(self.pre_path + self.dataset_path)

        # Check the unique values in the stratified column
        unique_values = df[stratified_column].unique()

        # Create an empty DataFrame to store the subset
        subset_df = pd.DataFrame()

        # Define the number of rows you want for each value in the stratified column
        rows_per_value = total_rows // len(unique_values)

        # Loop through each unique value and sample rows
        for value in unique_values:
            value_subset = df[df[stratified_column] == value].sample(rows_per_value, random_state=42)
            subset_df = pd.concat([subset_df, value_subset])

        # If the total number of rows is less than the specified total, sample the remaining rows from the entire dataset
        remaining_rows = total_rows - len(subset_df)
        remaining_subset = df.sample(remaining_rows, random_state=42)
        subset_df = pd.concat([subset_df, remaining_subset])

        # Optionally, you can shuffle the final subset
        subset_df = subset_df.sample(frac=1, random_state=42)

        # Remove the .csv extension from the input file name
        file_name_without_extension = os.path.splitext(os.path.basename(self.dataset_path))[0]

        # Rename the original file by appending '_original' to its name
        original_file_path = self.pre_path + file_name_without_extension + '_original5.csv'
        os.rename(self.pre_path + self.dataset_path, original_file_path)

        # Save the modified DataFrame to a new CSV file
        subset_df.to_csv(self.pre_path + self.dataset_path, index=False)

        return {"status": True, "data": "Subset created"}

    """
    Split the dataset into train, validation, and test sets.
    By providing the stratify_column argument, the stratify function ensures that
    the distribution of labels or classes is maintained in both sets.
    """

    def split_dataset(self, stratify_column=''):
        train_file_path = 'train_set.csv'
        valid_file_path = 'validation_set.csv'
        test_file_path = 'test_set.csv'

        df = pd.read_csv(self.pre_path + self.dataset_path, on_bad_lines='skip')  # Read the cleaned dataset CSV file

        # Split the dataset into train, validation, and test sets while stratifying by the stratify_column
        if stratify_column:  # If stratify_column is provided, then stratify
            train_valid, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[stratify_column])
            train, valid = train_test_split(train_valid, test_size=0.2, random_state=42,
                                            stratify=train_valid[stratify_column])
        else:  # Split the dataset without stratifying
            train_valid, test = train_test_split(df, test_size=0.2, random_state=42)
            train, valid = train_test_split(train_valid, test_size=0.2, random_state=42)

        # Save the split datasets to separate CSV files
        train.to_csv(self.pre_path + train_file_path, index=False)
        valid.to_csv(self.pre_path + valid_file_path, index=False)
        test.to_csv(self.pre_path + test_file_path, index=False)

        return {"status": True, "data": "Splitting succeed"}


# Example Usage
# Instantiate the DatasetMethods class by providing the (dataset_path)
DTS = DatasetMethods(dataset_path='Airline_review.csv')

# # Identify the unique labels in a specific column (column_name) to understand your dataset
# DTS.display_unique_values(column_name='Overall_Rating')  # 'Airline Name'

# # Remove rows where the value in the column (column_name) is equal to the specific value (value_to_remove)
# print(DTS.remove_rows_by_value(column_name='Overall_Rating', value_to_remove='n'))

# # Remove rows with empty values by providing specific column names or
# # by providing an empty array [] to check all columns
# print(DTS.remove_rows_with_empty_fields(column_names=['Airline Name', 'Overall_Rating', 'Review_Title', 'Review']))

# # Remove unnecessary columns by providing the array (columns_to_remove)
# print(DTS.remove_columns_and_save(columns_to_remove=
#                             ['Unnamed: 0', 'Review Date', 'Verified', 'Aircraft', 'Type Of Traveller', 'Seat Type',
#                              'Route', 'Date Flown',
#                              'Seat Comfort', 'Cabin Staff Service', 'Food & Beverages', 'Ground Service',
#                              'Inflight Entertainment',
#                              'Wifi & Connectivity', 'Value For Money', 'Recommended']))

# # Clean and standardize each row and value in your dataset
# print(DTS.standardize_and_write_csv())

# # Obtain a subset of the dataset with a specific number of rows (total_rows),
# # while ensuring appropriate label distribution by stratifying a specific column (stratified_column)
# print(DTS.create_stratified_subset(total_rows=2000, stratified_column='Overall_Rating'))

# # Split the dataset into training, validation, and test sets.
# # Provide the column name (stratify_column) as an argument if you need to control the distribution
# print(DTS.split_dataset(stratify_column='Overall_Rating'))
