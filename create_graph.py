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
books_data = pd.read_csv('cleaned_goodreads_data.csv')
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
    G.add_node(genre, size=count, genres=[genre])


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
    mode='markers+text',
    hoverinfo='text',
    marker=dict(
        size=node_size,
        color='#7f7f7f',  # Default color, will be updated interactively
        line_width=2
    ),
    customdata=list(G.nodes())  # Add node names as custom data for easy access
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

# Correctly maintain node sizes based on the genre popularity
node_sizes = [10 + G.nodes[node]['size'] / 100 for node in G.nodes()]  # Scale size appropriately

@app.callback(
    Output('genre-graph', 'figure'),
    [Input('genre-graph', 'clickData'), Input('search-bar', 'value')]
)
def update_graph(clickData, search_value):
    node_colors = ['#7f7f7f' for _ in G.nodes()]  # Default gray color
    edge_colors = [get_edge_color(edge[2]['count']) for edge in G.edges(data=True)]  # Original function to set color
    node_sizes = [10 + G.nodes[node]['size'] / 100 for node in G.nodes()]  # Maintain size based on popularity

    if clickData:
        # Get the node name from clickData
        node_name = clickData['points'][0]['customdata']
        connected_nodes = list(nx.all_neighbors(G, node_name)) + [node_name]

        # Update node colors
        node_colors = ['#FFFF00' if node in connected_nodes else '#7f7f7f' for node in G.nodes()]

        # Update edge colors only for edges between connected nodes
        for i, edge in enumerate(G.edges(data=True)):
            if edge[0] in connected_nodes and edge[1] in connected_nodes:
                edge_colors[i] = '#FFFF00'  # Highlight with a bright yellow

    elif search_value:
        # Handle search-based highlighting
        book_genres_colors = highlight_book_genres(search_value, books_data, G)
        if book_genres_colors:
            node_colors = [book_genres_colors.get(node, '#7f7f7f') for node in G.nodes()]
            edge_colors = ['rgba(211, 211, 211, 0.1)' if not (book_genres_colors.get(edge[0]) == '#FFFF00' and book_genres_colors.get(edge[1]) == '#FFFF00') else '#FFFF00' for edge in G.edges()]

    # Create the figure
    fig = go.Figure(layout=go.Layout(
        title='Network Graph of Literary Genres',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    ))

    # Add edge traces first
    for i, edge in enumerate(G.edges(data=True)):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        fig.add_trace(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=2, color=edge_colors[i]),
            mode='lines',
            hoverinfo='none'
        ))

    # Add node trace last
    node_trace = go.Scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        text=[f'{node} ({G.nodes[node]["size"]})' for node in G.nodes()],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line_width=2
        ),
        customdata=list(G.nodes())
    )

    fig.add_trace(node_trace)

    return fig


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
