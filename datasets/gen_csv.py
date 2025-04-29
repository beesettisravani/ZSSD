import pandas as pd
import os

# Load the CSV file
file_path = '/mnt/data/updated_sorted_data.csv'
output_dir = '/mnt/data/grouped_files/'
os.makedirs(output_dir, exist_ok=True)

def group_and_save(csv_path, output_dir, group_column):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Group by the specified column
    grouped = df.groupby(group_column)

    # Save each group to a separate CSV file
    for name, group in grouped:
        output_file = os.path.join(output_dir, f'{name}.csv')
        group.to_csv(output_file, index=False)
        print(f'Saved group "{name}" to {output_file}')

# Customize the group column here
group_column = 'target'  # Change to the correct column name from your file
group_and_save(c:\\Users\\HP\esktop\ezsd\updated_sorted_data.csv, output_dir, group_column)
