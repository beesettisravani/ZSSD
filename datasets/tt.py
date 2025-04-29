import pandas as pd

# Load the existing SemEval dataset
semeval_df = pd.read_csv(r"C:\Users\HP\Desktop\ezsd\SemEval-2016.csv")

# Load the Donald Trump dataset
donald_trump_df = pd.read_excel(r"C:\Users\HP\Desktop\ezsd\donald_trump_stance_300.xlsx")
donald_trump_df['id'] = 4  # Assign id 4

# Load the Climate Change dataset
climate_change_df = pd.read_excel(r"C:\Users\HP\Desktop\ezsd\climate_change_stance_300.xlsx")
climate_change_df['id'] = 5  # Assign id 5

# Make sure columns match
donald_trump_df = donald_trump_df[['Tweet', 'Target', 'Stance', 'id']]
climate_change_df = climate_change_df[['Tweet', 'Target', 'Stance', 'id']]

# Merge all datasets
final_df = pd.concat([semeval_df, donald_trump_df, climate_change_df], ignore_index=True)

# Save the final combined dataset
final_df.to_csv(r"C:\Users\HP\Desktop\ezsd\final_combined_dataset.csv", index=False)

print("✅ Final dataset created successfully!")
