import pandas as pd
import matplotlib.pyplot as plt

# Load the preprocessed data from a .txt file
def load_preprocessed_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 2:  # Ensure there are tokens present
                sender = parts[0]
                timestamp = parts[1]
                tokens = parts[2:]  # The rest are tokens
                data.append({'sender': sender, 'timestamp': timestamp, 'tokens': tokens})
    return pd.DataFrame(data)

# Load data
df = load_preprocessed_data('C:\\Users\\User\\Desktop\\10Acadamy\\Week-Five\\preprocessed_data.txt')

# Summary calculations
total_messages = df.shape[0]
total_tokens = df['tokens'].apply(len).sum()  # Count total tokens
average_tokens_per_message = df['tokens'].apply(len).mean()  # Calculate average

# Create summary data
summary = {
    'Total Messages': total_messages,
    'Total Tokens': total_tokens,
    'Average Tokens per Message': average_tokens_per_message
}

# Create a DataFrame for the summary
summary_df = pd.DataFrame(summary, index=[0])

# Print the summary table
print(summary_df)

# Plotting the summary table as an image
fig, ax = plt.subplots()
ax.axis('tight')
ax.axis('off')
table_data = summary_df.values.tolist()
table = ax.table(cellText=table_data, colLabels=summary_df.columns, cellLoc='center', loc='center')

# Save the table as a JPG image
plt.savefig('summary_table.jpg', bbox_inches='tight', pad_inches=0.1, dpi=300)

# Show the plot (optional)
plt.show()

