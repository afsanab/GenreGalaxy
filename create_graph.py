from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import networkx as nx
import plotly.graph_objs as go
import ast
import re
import os
import subprocess

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# Check if processed files exist
if not os.path.exists('cleaned_goodreads_data.csv') or not os.path.exists('goodreads_genre_pairs.csv'):
    print("Processed data files not found. Running process_data.py...")
    subprocess.run(['python', 'process_data.py'], check=True)
else:
    print("Processed data files found. Skipping data processing.")

# Load your data
genre_pairs_df = pd.read_csv('goodreads_genre_pairs.csv')
books_data = pd.read_csv('cleaned_goodreads_data.csv')
books_data['Genres'] = books_data['Genres'].apply(ast.literal_eval)  # Convert string list to actual list

book_options = [{'label': book, 'value': book} for book in books_data['Book'].unique()]
app.layout = html.Div([
    dcc.Location(id='url', refresh=True),
    dbc.Row([  # Create a row to contain the dropdown and the button
        dbc.Col(dcc.Dropdown(
            id='search-bar',
            options=book_options,
            placeholder='Search for a book...',
            style={'width': '100%'}  # Make the dropdown take the full column width
        ), width=9),  # Assign 9 out of 12 columns width to the dropdown
        dbc.Col(html.Button("Reset", id="reset-button", n_clicks=0, className="btn btn-warning"), 
                width=3, style={'text-align': 'right'})  # Assign 3 columns and align to the right
    ]),
    dcc.Graph(id='genre-graph'),
    html.Div(id='node-info', style={'padding': '40px'}),
    dbc.ListGroup(id='genre-list', className='list-group-flush')  # For displaying ranked genres
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
        return 'rgba(128, 0, 128, 0.5)'  # Dark Purple with transparency

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
        hoverinfo='none',
        showlegend=False
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
    customdata=list(G.nodes()),
    showlegend=False
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
                            (100, 'rgba(255, 0, 255, 0.6)'), 
                            (float('inf'), 'rgba(128, 0, 128, 0.6)')]

for count, color in co_occurrence_categories:
    legend_traces.append(go.Scatter(
        x=[None],
        y=[None],
        mode='lines',
        line=dict(color=color, width=10),
        name=f'{count if count != float("inf") else "100+"}'
    ))

# Correctly maintain node sizes based on the genre popularity
node_sizes = [10 + G.nodes[node]['size'] / 100 for node in G.nodes()]  # Scale size appropriately
from dash import callback_context


@app.callback(
    Output('url', 'href'),  # This will update the URL
    [Input('reset-button', 'n_clicks')],  # Triggered by the reset button click
    prevent_initial_call=True  # Prevents the callback from firing upon initialization
)
def reset_page(n_clicks):
    # This function triggers a reload by changing the URL endpoint.
    if n_clicks > 0:
        return '/'  # Assuming your Dash app's root is at "/", this reloads the page

# Color categories defined globally for easier access in multiple parts of the app
co_occurrence_categories = [
    (10, 'rgba(247, 252, 185, 0.5)'),
    (25, 'rgba(255, 204, 102, 0.5)'),
    (50, 'rgba(255, 153, 204, 0.5)'),
    (100, 'rgba(255, 0, 255, 0.5)'),
    (float('inf'), 'rgba(139, 0, 255, 0.5)')
]

def get_edge_color(count):
    for threshold, color in co_occurrence_categories:
        if count <= threshold:
            return color
    return co_occurrence_categories[-1][1]  # Return the color for the highest threshold

    
@app.callback(
    [Output('genre-graph', 'figure'),
     Output('genre-list', 'children')],  # Ensure 'genre-list' is the ID for the Div where the list is displayed.
    [Input('genre-graph', 'clickData'),
     Input('search-bar', 'value')]
)
def update_graph_and_list(clickData, search_value):
    # Initialize outputs for the genre list and default colors for nodes and edges
    node_colors = ['#7f7f7f' for _ in G.nodes()]  # Default grey for all nodes
    edge_colors = [get_edge_color(edge[2]['count']) for edge in G.edges(data=True)]  # Default colors
    node_sizes = [10 + G.nodes[node]['size'] / 100 for node in G.nodes()]  # Maintain size based on popularity
    genre_list_children = []  # Default empty list for genre rankings

    # Handle node click events
    if clickData:
        node_name = clickData['points'][0]['customdata']
        connected_nodes = list(nx.all_neighbors(G, node_name)) + [node_name]
        
        # Update node colors for connected nodes and grey out others
        node_colors = ['#FFFF00' if node in connected_nodes else 'rgba(211, 211, 211, 0.5)' for node in G.nodes()]
        
        # Update edge colors for connections between these nodes, grey out others
        for i, edge in enumerate(G.edges(data=True)):
            if edge[0] in connected_nodes and edge[1] in connected_nodes:
                edge_colors[i] = '#FFFF00'  # Highlight color for edges
            else:
                edge_colors[i] = 'rgba(211, 211, 211, 0.1)'  # Grey out unrelated edges

        # Gather and sort the co-occurrences to update the genre list
        related_genres = [(other, G[node_name][other]['count']) for other in connected_nodes if other != node_name]
        related_genres.sort(key=lambda x: x[1], reverse=True)  # Sort by count, descending

        # Format and number the list items
        genre_list_children = html.Ol([
            html.Li(f"{genre} - {count}") for genre, count in related_genres
        ], style={'padding-left': '40px', 'list-style-type': 'decimal'})

    # Handle search events
    elif search_value:
        book_genres_colors = highlight_book_genres(search_value, books_data, G)
        if book_genres_colors:
            node_colors = [book_genres_colors.get(node, 'rgba(211, 211, 211, 0.5)') for node in G.nodes()]
            edge_colors = ['rgba(211, 211, 211, 0.1)' if not (book_genres_colors.get(edge[0]) == '#FFFF00' and book_genres_colors.get(edge[1]) == '#FFFF00') else '#FFFF00' for edge in G.edges()]

    # Create the figure and layout configuration
    fig = go.Figure(layout=go.Layout(
        title={
            'text': 'Genre Galaxy',
            'y': 0.97,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {
                'family': 'Verdana',  # Choose a font family that suits your design
                'size': 24,  # Increase the font size for better visibility
                'color': 'black'  # Set the color of the title text
            }
        },
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),  # Adjust top margin if needed to make space for title
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    ))


    # Add edge traces
    for i, edge in enumerate(G.edges(data=True)):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        fig.add_trace(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=2, color=edge_colors[i]),
            mode='lines',
            hoverinfo='none',
            showlegend=False
        ))

    # Add node trace
    fig.add_trace(go.Scatter(
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
        customdata=list(G.nodes()),
        showlegend=False
    ))

    for threshold, color in co_occurrence_categories:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],  # No actual data points
            mode='lines',
            line=dict(color=color, width=10),
            name=f'{threshold if threshold != float("inf") else "100+"} co-occurrences'
        ))

    return fig, genre_list_children


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
