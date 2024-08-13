import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import networkx as nx
import plotly.graph_objs as go
import ast
import re

app = dash.Dash(__name__)

# Load your data
genre_pairs_df = pd.read_csv('goodreads_genre_pairs.csv')
books_data = pd.read_csv('goodreads_data.csv')
books_data['Genres'] = books_data['Genres'].apply(ast.literal_eval)  # Convert string list to actual list

book_options = [{'label': book, 'value': book} for book in books_data['Book'].unique()]

app.layout = html.Div([
    dcc.Dropdown(
        id='search-bar',
        options=book_options,
        placeholder='Search for a book...',
        search_value='',
        style={'width': '300px'}  # Adjust width as needed
    ),
    dcc.Graph(id='genre-graph')
])

# Initialize the graph
G = nx.Graph()

threshold = 10  

# Populate genre popularity
genre_count = {}
for genres in books_data['Genres']:
    for genre in genres:
        genre_count[genre] = genre_count.get(genre, 0) + 1

# Filter genres to include only those with at least 5 books
filtered_genres = {genre: count for genre, count in genre_count.items() if count >= threshold}

# Add nodes with genre popularity as size, only for genres with at least 5 books
for genre, count in filtered_genres.items():
    G.add_node(genre, size=count)


# Function to determine color based on co-occurrence count
def get_edge_color(count):
    # Each color now has an alpha value for transparency
    if count <= 10:
        return 'rgba(247, 252, 185, 0.5)'  # Yellow with transparency
    elif count <= 25:
        return 'rgba(255, 204, 102, 0.5)'  # Light Orange with transparency
    elif count <= 50:
        return 'rgba(255, 153, 204, 0.5)'  # Pink with transparency
    elif count <= 100:
        return 'rgba(255, 102, 255, 0.5)'  # Magenta with transparency
    else:
        return 'rgba(139, 0, 255, 0.5)'  # Dark Purple with transparency

# Add edges from the pairs DataFrame considering the threshold
genre_pair_counts = genre_pairs_df.groupby(['Genre1', 'Genre2']).size()
for (genre1, genre2), count in genre_pair_counts.items():
    if count >= threshold:
        if G.has_node(genre1) and G.has_node(genre2):
            G.add_edge(genre1, genre2, count=count)

# Position the nodes using a layout
pos = nx.kamada_kawai_layout(G)  # This layout is generally better for larger graphs

# Edge traces setup with initial colors based on co-occurrences
edge_traces = []
edge_colors = []  # Store initial edge colors based on co-occurrences

for edge in G.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    color = get_edge_color(edge[2]['count'])
    edge_colors.append(color)  # Store the color for dynamic updating later
    edge_traces.append(go.Scatter(
        x=[x0, x1, None],
        y=[y0, y1, None],
        line=dict(width=2, color=color),
        mode='lines',
        hoverinfo='none'
    ))


# Node traces
node_x = [pos[node][0] for node in G.nodes()]
node_y = [pos[node][1] for node in G.nodes()]
node_text = [f'{node} ({G.nodes[node]["size"]})' for node in G.nodes()]
node_size = [10 + G.nodes[node]['size'] / 100 for node in G.nodes()]

node_trace = go.Scatter(
    x=node_x, 
    y=node_y,
    text=node_text,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        size=node_size,
        color='#7f7f7f',  # Uniform gray color for nodes
        line_width=2
    )
)

# Function to highlight the genres of a selected book
def highlight_book_genres(book_title, books_data, G):
    # Find the book and its genres in the DataFrame
    # Adjusting the search to account for possible variations in series notation
    book_regex = r'^' + re.escape(book_title) + r'(\s\(\w+\s#\d+\))?$'
    book_row = books_data[books_data['Book'].str.contains(book_regex, case=False, na=False, regex=True)]
    
    if book_row.empty:
        return None  # Book not found, return None to indicate no update is necessary

    # Extract the genres for the found book
    book_genres = book_row['Genres'].iloc[0]

    # Create a set of colors for the nodes. All genres related to the book will be highlighted.
    node_colors = {node: ('#FFFF00' if node in book_genres else '#7f7f7f') for node in G.nodes()}
    
    return node_colors

legend_traces = []
co_occurrence_categories = [(10, 'rgba(247, 252, 185, 0.6)'), 
                            (25, 'rgba(255, 204, 102, 0.6)'), 
                            (50, 'rgba(255, 153, 204, 0.6)'), 
                            (100, 'rgba(255, 102, 255, 0.6)'), 
                            (float('inf'), 'rgba(139, 0, 255, 0.6)')]

for count, color in co_occurrence_categories:
    legend_traces.append(go.Scatter(
        x=[None],
        y=[None],
        mode='lines',
        line=dict(color=color, width=10),
        name=f'{count if count != float("inf") else "100+"} co-occurrences'
    ))

@app.callback(
    Output('genre-graph', 'figure'),
    [Input('search-bar', 'value')]
)
def update_graph(search_value):
    # Start with the original edge colors based on co-occurrences
    current_edge_colors = list(edge_colors)  # Copy the original edge colors

    # Default all node colors to grey
    node_colors = ['#7f7f7f' for _ in G.nodes()]

    if search_value:
        # Attempt to fetch the genres associated with the book
        book_genres_colors = highlight_book_genres(search_value, books_data, G)
        if book_genres_colors:
            # Update node colors for genres related to the book
            node_colors = [book_genres_colors.get(node, '#7f7f7f') for node in G.nodes()]

            # Set all edges to a more transparent gray to ensure we don't incorrectly color unrelated edges
            for i, edge in enumerate(G.edges()):
                current_edge_colors[i] = 'rgba(211, 211, 211, 0.1)'  # More transparent gray for non-highlighted edges

            # Highlight edges between the book's genres
            for i, edge in enumerate(G.edges(data=True)):
                if edge[0] in book_genres_colors and edge[1] in book_genres_colors:
                    if book_genres_colors[edge[0]] == '#FFFF00' and book_genres_colors[edge[1]] == '#FFFF00':
                        current_edge_colors[i] = '#FFFF00'  # Bright yellow for highlighted edges
        else:
            # If no genres found for the book, ensure all edges are more transparent
            current_edge_colors = ['rgba(211, 211, 211, 0.1)' for _ in G.edges()]

    # Generate edge traces with updated colors
    updated_edge_traces = []
    for i, edge in enumerate(G.edges(data=True)):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        updated_edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=2, color=current_edge_colors[i]),
            mode='lines',
            hoverinfo='none'
        ))

    # Update the node trace with new colors
    node_trace.marker.color = node_colors

    # Create the figure with updated traces
    fig = go.Figure(data=updated_edge_traces + [node_trace], layout=go.Layout(
        title='Network Graph of Literary Genres',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    ))

    return fig

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
