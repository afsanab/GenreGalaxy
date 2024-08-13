import pandas as pd
import networkx as nx
import plotly.graph_objs as go
import ast

# Load your data
# Assuming you have a file named 'goodreads_genre_pairs.csv' and another for genre popularity
genre_pairs_df = pd.read_csv('goodreads_genre_pairs.csv')
books_data = pd.read_csv('goodreads_data.csv')
books_data['Genres'] = books_data['Genres'].apply(ast.literal_eval)  # convert string of list to actual list

# Initialize the graph
G = nx.Graph()

# Populate genre popularity
genre_count = {}
for genres in books_data['Genres']:
    for genre in genres:
        if genre in genre_count:
            genre_count[genre] = genre_count[genre] + 1
        else:
            genre_count[genre] = 1

# Add nodes with genre popularity as size
for genre, count in genre_count.items():
    G.add_node(genre, size=count)

# Add edges from the pairs DataFrame
for index, row in genre_pairs_df.iterrows():
    if G.has_node(row['Genre1']) and G.has_node(row['Genre2']):
        G.add_edge(row['Genre1'], row['Genre2'])

# Network layout
pos = nx.spring_layout(G)  # Kamada-Kawai layout for large graphs if necessary

# Edge traces
x_edges = []
y_edges = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    x_edges.extend([x0, x1, None])
    y_edges.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=x_edges,
    y=y_edges,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

# Node traces
x_nodes = []
y_nodes = []
node_texts = []
node_sizes = []
for node in G.nodes():
    x_nodes.append(pos[node][0])
    y_nodes.append(pos[node][1])
    node_texts.append(f'{node} ({G.nodes[node]["size"]})')
    node_sizes.append(10 + G.nodes[node]['size'] / 100)  # Adjust node size scaling as needed

node_trace = go.Scatter(
    x=x_nodes,
    y=y_nodes,
    text=node_texts,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        size=node_sizes,
        color=[],
        line=dict(width=2)))

# Create the figure and add the traces
fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
    title='Network Graph of Literary Genres',
    showlegend=False,
    hovermode='closest',
    margin=dict(b=20, l=5, r=5, t=40),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

fig.show()
