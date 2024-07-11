import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create a simple data dictionary
data = {
    'Category': ['A', 'B', 'C', 'D', 'E'],
    'Values': [10, 20, 15, 25, 30]
}

# Convert the dictionary to a Pandas DataFrame
df = pd.DataFrame(data)

# Set the style of the plot
sns.set(style="whitegrid")

# Create a bar plot
plt.figure(figsize=(10, 6))
bar_plot = sns.barplot(x='Category', y='Values', data=df, palette='viridis')

# Add titles and labels
plt.title('Simple Bar Plot', fontsize=16)
plt.xlabel('Category', fontsize=14)
plt.ylabel('Values', fontsize=14)

# Save the plot to a file
output_plot = 'plot.png'
plt.savefig(output_plot)

# Show the plot
plt.show()

print(f"The plot has been saved to {output_plot}")
