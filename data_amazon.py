# -*- coding: utf-8 -*-

import pandas as pd
import os

# Load the dataset
file_path = 'amazon_suggestions_graphrnn.csv'
data = pd.read_csv(file_path)

# Output directory
output_dir = 'datasets/amazon/'
os.makedirs(output_dir, exist_ok=True)

# Decide how to group the data into graphs.
if data['Initial_Node'].nunique() > 1:
    # Group by "Initial_Node" and unpack the tuple (key, group)
    group_iter = data.groupby('Initial_Node')
    # group_iter is an iterable of (group_key, group DataFrame)
else:
    # If there's only one unique initial node, split into fixed-size chunks.
    chunk_size = 50  # Adjust chunk size as needed.
    group_iter = [(i, data.iloc[i:i+chunk_size]) for i in range(0, len(data), chunk_size)]

# Initialize lists for file outputs
amazon_A = []                   # Edge list: "source, target"
amazon_graph_indicator = []     # For each node, the graph it belongs to
amazon_node_labels = []         # Numeric label for each node (local ID)
amazon_graph_labels = []        # One label per graph

graph_id = 1  # Graph counter

# Process each group
# If group_iter comes from groupby, iterate as: for key, group in group_iter:
# If it's from chunking, it is already a tuple (key, group)
for key, group in group_iter:
    node_map = {}  # maps each unique query/suggestion to its local numeric node ID
    node_id = 1    # reset numbering for each graph

    # Process the "Query" column and assign node IDs
    for query in group['Query']:
        if query not in node_map:
            node_map[query] = node_id
            amazon_graph_indicator.append(f"{graph_id}\n")
            amazon_node_labels.append(f"{node_id}\n")
            node_id += 1

    # Process the "Suggestions" column; each row may contain multiple suggestions separated by ", "
    for suggestions in group['Suggestions']:
        for suggestion in suggestions.split(', '):
            if suggestion not in node_map:
                node_map[suggestion] = node_id
                amazon_graph_indicator.append(f"{graph_id}\n")
                amazon_node_labels.append(f"{node_id}\n")
                node_id += 1

    # Record edges: for each row, create an edge from the query node to each suggestion node
    for _, row in group.iterrows():
        query_id = node_map[row['Query']]
        for suggestion in row['Suggestions'].split(', '):
            suggestion_id = node_map[suggestion]
            amazon_A.append(f"{query_id}, {suggestion_id}\n")
    
    # Record a graph-level label (optional) – here we simply use the graph ID.
    amazon_graph_labels.append(f"{graph_id}\n")
    
    graph_id += 1

# Write the output files
with open(os.path.join(output_dir, 'amazon_A.txt'), 'w') as f:
    f.writelines(amazon_A)

with open(os.path.join(output_dir, 'amazon_graph_indicator.txt'), 'w') as f:
    f.writelines(amazon_graph_indicator)

with open(os.path.join(output_dir, 'amazon_node_labels.txt'), 'w') as f:
    f.writelines(amazon_node_labels)

with open(os.path.join(output_dir, 'amazon_graph_labels.txt'), 'w') as f:
    f.writelines(amazon_graph_labels)

print(f"Files have been saved in {output_dir}")
