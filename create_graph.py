import pandas as pd
import networkx as nx
import plotly.graph_objs as go

# Load the genre pairs data
genre_pairs_df = pd.read_csv('goodreads_genre_pairs.csv')

# Initialize a graph
G = nx.Graph()

# Add edges between genres from the genre pairs DataFrame with a threshold
threshold = 5  # Only consider genre pairs with at least this many co-occurrences
genre_pair_counts = genre_pairs_df.groupby(['Genre1', 'Genre2']).size()
filtered_edges = genre_pair_counts[genre_pair_counts >= threshold].index

for genre1, genre2 in filtered_edges:
    G.add_edge(genre1, genre2)

# Use a more efficient layout algorithm if 'spring_layout' is too slow
pos = nx.kamada_kawai_layout(G)  # This layout algorithm is generally faster

# Prepare traces for Plotly
edge_trace = go.Scatter(
    x=[],
    y=[],
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_trace['x'] += tuple([x0, x1, None])
    edge_trace['y'] += tuple([y0, y1, None])

node_trace = go.Scatter(
    x=[],
    y=[],
    text=[],
    mode='markers',
    hoverinfo='text',
    marker=dict(
        color=[],
        size=10,
        line_width=2))

for node in G.nodes():
    x, y = pos[node]
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])
    node_trace['text'] += tuple([node])

# Create the figure and add the traces
fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Network graph of literary genres',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )

# Display the figure
fig.show()
