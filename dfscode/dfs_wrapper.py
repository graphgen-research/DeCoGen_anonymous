import os
import subprocess
import tempfile
import pickle
import networkx as nx
import torch

def get_min_dfscode(G, temp_path=tempfile.gettempdir()):
    input_fd, input_path = tempfile.mkstemp(dir=temp_path)
    output_fd, output_path = tempfile.mkstemp(dir=temp_path)

    try:
        with open(input_path, 'w') as f:
            vcount = len(G.nodes)
            f.write(str(vcount) + '\n')
            i = 0
            d = {}
            for x in G.nodes:
                d[x] = i
                i += 1
                # Write label as integer if possible; otherwise, write as is.
                try:
                    label_int = int(G.nodes[x]['label'])
                    f.write(str(label_int) + '\n')
                except Exception:
                    f.write(str(G.nodes[x]['label']) + '\n')
            ecount = len(G.edges)
            f.write(str(ecount) + '\n')
            for (u, v) in G.edges:
                edge_label = G[u][v].get('label', "DEFAULT_LABEL")
                f.write(f"{d[u]} {d[v]} {edge_label}\n")
    except Exception as e:
        os.close(input_fd)
        os.close(output_fd)
        return None

    dfscode_bin_path = 'bin/dfscode'
    try:
        with open(input_path, 'r') as f_in:
            ret = subprocess.call([dfscode_bin_path, output_path, '2'], stdin=f_in)
            if ret != 0:
                raise RuntimeError(f"DFS binary failed with exit code {ret}")
    except Exception as e:
        os.close(input_fd)
        os.close(output_fd)
        try:
            os.remove(input_path)
            os.remove(output_path)
        except OSError:
            pass
        return None

    dfs_sequence = []
    try:
        with open(output_path, 'r') as dfsfile:
            lines = dfsfile.readlines()
            if not lines:
                raise ValueError("Empty DFS output")
            for row in lines:
                splited_row = row.split()
                if len(splited_row) < 10:
                    raise ValueError("DFS output format error")
                tokens = [splited_row[2 * i + 1] for i in range(5)]
                dfs_sequence.append(tokens)
    except Exception as e:
        dfs_sequence = None

    os.close(input_fd)
    os.close(output_fd)
    try:
        os.remove(input_path)
        os.remove(output_path)
    except OSError:
        pass

    return dfs_sequence

def graph_from_dfscode(dfscode):
    graph = nx.Graph()
    for dfscode_edge in dfscode:
        try:
            i, j, l1, e, l2 = dfscode_edge
            i = int(i)
            j = int(j)
        except Exception:
            continue
        if not graph.has_node(i):
            graph.add_node(i, label=l1)
        if not graph.has_node(j):
            graph.add_node(j, label=l2)
        graph.add_edge(i, j, label=e)
    return graph

def dfscode_to_tensor(dfscode, feature_map):
    """
    Converts a DFS code (list of lists of tokens) into tensor representations.
    Assumes that node labels in the DFS code are numeric strings.
    This version looks up the node labels as strings (e.g. "1", "2", etc.)
    to match the feature_map's node_forward dictionary keys.
    """
    max_nodes, max_edges = feature_map['max_nodes'], feature_map['max_edges']
    node_forward_dict = feature_map['node_forward']
    edge_forward_dict = feature_map['edge_forward']
    num_nodes_feat = len(node_forward_dict)
    num_edges_feat = len(edge_forward_dict)

    dfscode_tensors = {
        't1': (max_nodes + 1) * torch.ones(max_edges + 1, dtype=torch.long),
        't2': (max_nodes + 1) * torch.ones(max_edges + 1, dtype=torch.long),
        'v1': (num_nodes_feat + 1) * torch.ones(max_edges + 1, dtype=torch.long),
        'e': (num_edges_feat + 1) * torch.ones(max_edges + 1, dtype=torch.long),
        'v2': (num_nodes_feat + 1) * torch.ones(max_edges + 1, dtype=torch.long),
        'len': len(dfscode)
    }

    for i, code in enumerate(dfscode):
        dfscode_tensors['t1'][i] = int(code[0])
        dfscode_tensors['t2'][i] = int(code[1])
        # Look up node labels as strings.
        v1_key = int(code[2])  # e.g. "1"
        v2_key = int(code[4])  # e.g. "2"
        edge_label = code[3]  # remains as string (e.g., "DEFAULT_LABEL")
        dfscode_tensors['v1'][i] = node_forward_dict[v1_key]
        dfscode_tensors['e'][i] = edge_forward_dict[edge_label]
        dfscode_tensors['v2'][i] = node_forward_dict[v2_key]

    dfscode_tensors['t1'][len(dfscode)] = max_nodes
    dfscode_tensors['t2'][len(dfscode)] = max_nodes
    dfscode_tensors['v1'][i] = node_forward_dict[v1_key]
    dfscode_tensors['v2'][i] = node_forward_dict[v2_key]
    dfscode_tensors['e'][len(dfscode)] = num_edges_feat

    return dfscode_tensors

def dfscode_from_file_to_tensor_to_file(min_dfscode_file, min_dfscodes_path, min_dfscode_tensors_path, feature_map):
    with open(os.path.join(min_dfscodes_path, min_dfscode_file), 'rb') as f:
        min_dfscode = pickle.load(f)
    dfscode_tensors = dfscode_to_tensor(min_dfscode, feature_map)
    with open(os.path.join(min_dfscode_tensors_path, min_dfscode_file), 'wb') as f:
        pickle.dump(dfscode_tensors, f)

def min_dfscodes_to_tensors(min_dfscodes_path, min_dfscode_tensors_path, feature_map):
    min_dfscodes = [filename for filename in os.listdir(min_dfscodes_path) if filename.endswith(".dat")]
    from multiprocessing import Pool
    MAX_WORKERS = 48
    def process_file(filename):
        try:
            dfscode_from_file_to_tensor_to_file(filename, min_dfscodes_path, min_dfscode_tensors_path, feature_map)
            return True
        except Exception as ex:
            print("Error processing file", filename, ":", ex)
            return False

    with Pool(processes=MAX_WORKERS) as pool:
        results = list(pool.imap_unordered(process_file, min_dfscodes, chunksize=16))

    success_count = sum(1 for r in results if r)
    print('Done creating dfscode tensors. Successfully processed:', success_count, "out of", len(min_dfscodes))
    return success_count

if __name__ == '__main__':
    graph_path = os.path.expanduser('~/MTP/data/dataset/ENZYMES/graphs/graph180.dat')
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    dfs_code = get_min_dfscode(G)
    if dfs_code is not None:
        print("Length of DFS code:", len(dfs_code), "Graph edges:", G.number_of_edges())
        for code in dfs_code:
            print(code)
        reconstructed = graph_from_dfscode(dfs_code)
        print("Reconstructed graph: nodes =", reconstructed.number_of_nodes(),
              "edges =", reconstructed.number_of_edges())
    else:
        print("DFS code extraction failed.")
