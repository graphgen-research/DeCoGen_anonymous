import os
import shutil
import pickle
import torch

def mkdir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)

def load_graphs(graphs_path, graphs_identifiers=None):
    """
    Returns a list of graphs given a graphs directory and optionally a list of identifiers.
    If graphs_identifiers is not provided, all files ending in '.dat' in the directory will be loaded.
    If graphs_identifiers is provided:
      - If its first element is an integer, it assumes the files are named as 'graph<identifier>.dat'.
      - If its first element is a string (typically ending in '.dat'), it uses the filenames as provided.
    """
    graphs = []
    if graphs_identifiers is None:
        for name in os.listdir(graphs_path):
            if not name.endswith('.dat'):
                continue
            with open(os.path.join(graphs_path, name), 'rb') as f:
                graphs.append(pickle.load(f))
    else:
        # Check the type of the first element.
        if isinstance(graphs_identifiers[0], int):
            for ind in graphs_identifiers:
                fname = os.path.join(graphs_path, 'graph' + str(ind) + '.dat')
                with open(fname, 'rb') as f:
                    graphs.append(pickle.load(f))
        elif isinstance(graphs_identifiers[0], str):
            for fname in graphs_identifiers:
                with open(os.path.join(graphs_path, fname), 'rb') as f:
                    graphs.append(pickle.load(f))
        else:
            raise ValueError("graphs_identifiers must be a list of integers or a list of strings ending with '.dat'")
    return graphs

def save_graphs(graphs_path, graphs):
    """
    Save networkx graphs to a directory with indexing starting from 0.
    Files will be saved as 'graph0.dat', 'graph1.dat', etc.
    """
    for i in range(len(graphs)):
        with open(os.path.join(graphs_path, 'graph' + str(i) + '.dat'), 'wb') as f:
            pickle.dump(graphs[i], f)

# Create Directories for outputs
def create_dirs(args):
    if args.clean_tensorboard and os.path.isdir(args.tensorboard_path):
        shutil.rmtree(args.tensorboard_path)

    if args.clean_temp and os.path.isdir(args.temp_path):
        shutil.rmtree(args.temp_path)

    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)

    if not os.path.isdir(args.temp_path):
        os.makedirs(args.temp_path)

    if not os.path.isdir(args.tensorboard_path):
        os.makedirs(args.tensorboard_path)

    if not os.path.isdir(args.current_temp_path):
        os.makedirs(args.current_temp_path)

def save_model(epoch, args, model, optimizer=None, scheduler=None, **extra_args):
    if not os.path.isdir(args.current_model_save_path):
        os.makedirs(args.current_model_save_path)

    fname = args.current_model_save_path + args.fname + '_' + str(epoch) + '.dat'
    checkpoint = {'saved_args': args, 'epoch': epoch}

    save_items = {'model': model}
    if optimizer:
        save_items['optimizer'] = optimizer
    if scheduler:
        save_items['scheduler'] = scheduler

    for name, d in save_items.items():
        save_dict = {}
        for key, value in d.items():
            save_dict[key] = value.state_dict()
        checkpoint[name] = save_dict

    if extra_args:
        for arg_name, arg in extra_args.items():
            checkpoint[arg_name] = arg

    torch.save(checkpoint, fname)

def load_model(path, device, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(path, map_location=device)

    for name, d in {'model': model, 'optimizer': optimizer, 'scheduler': scheduler}.items():
        if d is not None:
            for key, value in d.items():
                value.load_state_dict(checkpoint[name][key])

        if name == 'model':
            for _, value in d.items():
                value.to(device=device)

def get_model_attribute(attribute, path, device):
    fname = path
    checkpoint = torch.load(fname, map_location=device)
    return checkpoint[attribute]
