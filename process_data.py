import pandas as pd
import ast
from itertools import combinations

# Load the dataset
file_path = 'goodreads_data.csv'  # Update this to your file's actual path
data = pd.read_csv(file_path)

# Clean the data
## Remove unnecessary columns
data_cleaned = data.drop(columns=['URL', 'Description', 'Unnamed: 0'])

## Convert 'Genres' from string representation to a list
data_cleaned['Genres'] = data_cleaned['Genres'].apply(ast.literal_eval)

## Drop rows with missing 'Genres' (if any)
data_cleaned = data_cleaned.dropna(subset=['Genres'])

# Extract unique genres
unique_genres = set()
data_cleaned['Genres'].apply(unique_genres.update)

# Create genre pairs for each book
genre_pairs = []
for genres_list in data_cleaned['Genres']:
    for pair in combinations(genres_list, 2):
        genre_pairs.append(pair)

# Convert genre pairs into a DataFrame
genre_pairs_df = pd.DataFrame(genre_pairs, columns=['Genre1', 'Genre2'])

# Optional: Save the cleaned data and genre pairs to new CSV files
data_cleaned.to_csv('cleaned_goodreads_data.csv', index=False)
genre_pairs_df.to_csv('goodreads_genre_pairs.csv', index=False)

print("Data cleaning and processing complete. Check the current directory for the output CSV files.")
