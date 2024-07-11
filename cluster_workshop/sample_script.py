import pandas as pd

# Create a simple data dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}

# Convert the dictionary to a Pandas DataFrame
df = pd.DataFrame(data)

# Define the output CSV file name
output_csv = '* put directory here * /output.csv' #delete the "*" after providing directory

# Save the DataFrame to a CSV file
df.to_csv(output_csv, index=False)