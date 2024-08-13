import pandas as pd
import networkx as nx
import plotly.graph_objs as go
import ast

# Load your data
genre_pairs_df = pd.read_csv('goodreads_genre_pairs.csv')
books_data = pd.read_csv('goodreads_data.csv')
books_data['Genres'] = books_data['Genres'].apply(ast.literal_eval)  # Convert string list to actual list

# Initialize the graph
G = nx.Graph()

# Populate genre popularity
genre_count = {}
for genres in books_data['Genres']:
    for genre in genres:
        genre_count[genre] = genre_count.get(genre, 0) + 1

# Add nodes with genre popularity as size
for genre, count in genre_count.items():
    G.add_node(genre, size=count)

# Define a threshold for including edges
threshold = 10  # Only consider genre pairs with at least this many co-occurrences

# Function to determine color based on co-occurrence count
def get_edge_color(count):
    if count < 10:
        return '#f7fcb9'  # Yellow
    elif count <= 25:
        return '#ffcc66'  # Light Orange
    elif count <= 50:
        return '#ff99cc'  # Pink
    elif count <= 100:
        return '#ff66ff'  # Magenta
    else:
        return '#8b00ff'  # Dark Purple

# Add edges from the pairs DataFrame considering the threshold
genre_pair_counts = genre_pairs_df.groupby(['Genre1', 'Genre2']).size()
for (genre1, genre2), count in genre_pair_counts.items():
    if count >= threshold:
        if G.has_node(genre1) and G.has_node(genre2):
            G.add_edge(genre1, genre2, count=count)

# Position the nodes using a layout
pos = nx.kamada_kawai_layout(G)  # This layout is generally better for larger graphs

# Edge traces
edge_traces = []
for edge in G.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    color = get_edge_color(edge[2]['count'])
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

# Create the figure and add the traces
fig = go.Figure(data=edge_traces + [node_trace], layout=go.Layout(
    title='Network Graph of Literary Genres',
    showlegend=False,
    hovermode='closest',
    margin=dict(b=20, l=5, r=5, t=40),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
))

fig.show()
