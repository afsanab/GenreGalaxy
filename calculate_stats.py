import pandas as pd

# Load the dataset
genre_pairs_df = pd.read_csv('goodreads_genre_pairs.csv')

# Calculate the number of occurrences for each genre pair
counts = genre_pairs_df.groupby(['Genre1', 'Genre2']).size()

# Get descriptive statistics
stats = counts.describe()

# Print the statistics
print(stats)

# You might want to also check the maximum value specifically
max_count = counts.max()
print(f"The maximum number of appearances for any genre pair is: {max_count}")
