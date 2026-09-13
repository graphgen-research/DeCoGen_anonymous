import os
os.environ["DGL_SKIP_GRAPHBOLT"] = "1"

import warnings
# Suppress PyTorch FutureWarnings about weights_only
warnings.filterwarnings(
    "ignore", category=FutureWarning, module=r"torch\.storage"
)

from model import create_model
import random
import time
import pickle
from torch.utils.data import DataLoader

from args import Args
from utils import create_dirs
from datasets.process_dataset import create_graphs
from datasets.preprocess import calc_max_prev_node, dfscodes_weights
from baselines.dgmg.data import DGMG_Dataset_from_file
from baselines.graph_rnn.data import Graph_Adj_Matrix_from_file
from graphgen.data import Graph_DFS_code_from_file
from model import create_model
from train import train

if __name__ == '__main__':
    args = Args()
    args = args.update_args()

    create_dirs(args)

    random.seed(2)

    # Load all graphs using the process_dataset function.
    graphs = create_graphs(args)
    random.shuffle(graphs)

    # Split graphs into 80% train, 10% validation, and 10% test.
    total = len(graphs)
    train_end = int(0.80 * total)
    valid_end = int(0.90 * total)
    
    graphs_train = graphs[:train_end]
    graphs_validate = graphs[train_end:valid_end]
    graphs_test = graphs[valid_end:]

    # Display the split sizes.
    print('Model:', args.note)
    print('Device:', args.device)
    print('Graph type:', args.graph_type)
    print('Training set: {}, Validation set: {}, Test set: {}'.format(
        len(graphs_train), len(graphs_validate), len(graphs_test)
    ))

    # Load the feature map.
    # (Ideally, args.current_dataset_path is set by create_graphs or elsewhere.)
    feature_map_path = os.path.join(args.current_dataset_path, 'map.dict')
    with open(feature_map_path, 'rb') as f:
        feature_map = pickle.load(f)

    print('Max number of nodes: {}'.format(feature_map['max_nodes']))
    print('Max number of edges: {}'.format(feature_map['max_edges']))
    print('Min number of nodes: {}'.format(feature_map['min_nodes']))
    print('Min number of edges: {}'.format(feature_map['min_edges']))
    print('Max degree of a node: {}'.format(feature_map['max_degree']))
    print('No. of node labels: {}'.format(len(feature_map['node_forward'])))
    print('No. of edge labels: {}'.format(len(feature_map['edge_forward'])))

    # Set model-specific parameters
    if args.note == 'GraphRNN':
        start = time.time()
        if args.max_prev_node is None:
            args.max_prev_node = max(calc_max_prev_node(args.current_processed_dataset_path))
        args.max_head_and_tail = None
        print('max_prev_node:', args.max_prev_node)
        end = time.time()
        print('Time taken to calculate max_prev_node = {:.3f}s'.format(end - start))

    # For DFScodeRNN, update feature_map with dfscodes weights if required.
    if args.note == 'DFScodeRNN' and args.weights:
        feature_map = {
            **feature_map,
            **dfscodes_weights(args.min_dfscode_path, graphs_train, feature_map, args.device)
        }

    # Create dataset objects from the splits.
    if args.note == 'GraphRNN':
        random_bfs = True
        dataset_train = Graph_Adj_Matrix_from_file(args, graphs_train, feature_map, random_bfs)
        dataset_validate = Graph_Adj_Matrix_from_file(args, graphs_validate, feature_map, random_bfs)
        dataset_test = Graph_Adj_Matrix_from_file(args, graphs_test, feature_map, random_bfs)
    elif args.note == 'DFScodeRNN':
        dataset_train = Graph_DFS_code_from_file(args, graphs_train, feature_map)
        dataset_validate = Graph_DFS_code_from_file(args, graphs_validate, feature_map)
        dataset_test = Graph_DFS_code_from_file(args, graphs_test, feature_map)
    elif args.note == 'DGMG':
        dataset_train = DGMG_Dataset_from_file(args, graphs_train, feature_map)
        dataset_validate = DGMG_Dataset_from_file(args, graphs_validate, feature_map)
        dataset_test = DGMG_Dataset_from_file(args, graphs_test, feature_map)

    # Create DataLoaders.
    if args.note == 'DGMG':
        dataloader_train = DataLoader(
            dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
            num_workers=args.num_workers, collate_fn=dataset_train.collate_batch)
        dataloader_validate = DataLoader(
            dataset_validate, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=dataset_validate.collate_batch)
        # Optionally, create a DataLoader for the test set.
        dataloader_test = DataLoader(
            dataset_test, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=dataset_test.collate_batch)
    else:
        dataloader_train = DataLoader(
            dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
            num_workers=args.num_workers)
        dataloader_validate = DataLoader(
            dataset_validate, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers)
        # Optionally, create a DataLoader for the test set.
        dataloader_test = DataLoader(
            dataset_test, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers)

    # Create and train the model.
    model = create_model(args, feature_map)
    
    train(args, dataloader_train, model, feature_map, dataloader_validate)

   
    
