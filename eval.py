import os
import random
import shutil
from statistics import mean
import torch
import networkx as nx

from graphgen.train import predict_graphs as gen_graphs_dfscode_rnn
from baselines.graph_rnn.train import predict_graphs as gen_graphs_graph_rnn
from baselines.dgmg.train import predict_graphs as gen_graphs_dgmg
from utils import get_model_attribute, load_graphs, save_graphs

import metrics.stats

# ===== gSpan wrapper for minimal DFS codes (used for FW/BW analysis) =====
from dfscode.dfs_wrapper import get_min_dfscode

LINE_BREAK = '----------------------------------------------------------------------\n'

# temp dir for DFS code extraction
DFS_TEMP_DIR = os.path.join('tmp_dfscode')
os.makedirs(DFS_TEMP_DIR, exist_ok=True)


class ArgsEvaluate():
    def __init__(self):
        # Can manually select the device too
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # === UPDATE THIS TO YOUR MODEL ===
        self.model_path = '/DeCoGen/model_save/DFScodeRNN_cora_2026-07-15 19:12:22/DFScodeRNN_cora_368.dat'

        self.num_epochs = get_model_attribute('epoch', self.model_path, self.device)

        # Whether to generate networkx graphs from the trained model
        self.generate_graphs = True

        self.count = 2560
        self.batch_size = 32  # Must divide count
        self.metric_eval_batch_size = 40

        # DFScodeRNN
        self.max_num_edges = 109

        # GraphRNN (kept for compatibility if you switch model)
        self.min_num_node = 9
        self.max_num_node = 99

        self.train_args = get_model_attribute('saved_args', self.model_path, self.device)

        self.graphs_save_path = 'graphs/'
        self.current_graphs_save_path = (
            self.graphs_save_path
            + self.train_args.fname + '_' + self.train_args.time + '/'
            + str(self.num_epochs) + '/'
        )


def patch_graph(graph: nx.Graph) -> nx.Graph:
    """Standardize node label field (strip suffix after first '-')."""
    for _, data in graph.nodes(data=True):
        raw_label = data.get('label', '')
        data['label'] = str(raw_label).split('-', 1)[0]
    return graph


# ========= minimal-DFS helpers (for consistent FW/BW orientation) =========
def _to_undirected_copy(G: nx.Graph) -> nx.Graph:
    """Return a simple undirected labeled copy (gSpan expects undirected)."""
    H = nx.Graph()
    H.add_nodes_from((n, d.copy()) for n, d in G.nodes(data=True))
    H.add_edges_from((u, v, d.copy()) for u, v, d in G.edges(data=True))
    return H


def _safe_min_dfscode(H: nx.Graph):
    """Get minimal DFS code; return [] on failure/None."""
    try:
        code = get_min_dfscode(H, DFS_TEMP_DIR)
        if code is None:
            return []
        return list(code)
    except Exception:
        return []


def analyze_edge_directions(graphs):
    """
    Compute forward/backward counts via minimal DFS code (t1<t2 forward).
    Robust to missing/invalid codes. Works the same for Test and Generated.
    """
    forward_counts, backward_counts, total_counts = [], [], []
    skipped_graphs = 0

    for G in graphs:
        H = _to_undirected_copy(G)
        dfscode = _safe_min_dfscode(H)

        if not dfscode:
            skipped_graphs += 1
            forward_counts.append(0)
            backward_counts.append(0)
            total_counts.append(0)
            continue

        fwd = bwd = 0
        # Each code item is typically [t1, t2, elabel, l1, l2] or similar; we only need t1,t2
        for code in dfscode:
            if len(code) >= 2:
                t1, t2 = code[0], code[1]
            else:
                continue
            # be tolerant to string/integer forms
            try:
                t1, t2 = int(t1), int(t2)
            except Exception:
                pass
            if t1 < t2:
                fwd += 1
            elif t1 > t2:
                bwd += 1

        total = fwd + bwd
        forward_counts.append(fwd)
        backward_counts.append(bwd)
        total_counts.append(total)

    avg_forward = mean(forward_counts) if forward_counts else 0
    avg_backward = mean(backward_counts) if backward_counts else 0
    avg_total = mean(total_counts) if total_counts else 0

    forward_ratio = (avg_forward / avg_total) if avg_total > 0 else 0.0
    backward_ratio = (avg_backward / avg_total) if avg_total > 0 else 0.0

    return {
        'forward_count': avg_forward,
        'backward_count': avg_backward,
        'total_count': avg_total,
        'forward_ratio': forward_ratio,
        'backward_ratio': backward_ratio,
        'individual_forward': forward_counts,
        'individual_backward': backward_counts,
        'skipped_graphs': skipped_graphs,
    }
# ==========================================================================


def generate_graphs(eval_args):
    """
    Generate graphs (networkx format) from a trained generative model and save them.
    """
    train_args = eval_args.train_args

    if train_args.note == 'GraphRNN':
        gen_graphs = gen_graphs_graph_rnn(eval_args)
    elif train_args.note == 'DFScodeRNN':
        gen_graphs = gen_graphs_dfscode_rnn(eval_args)
    elif train_args.note == 'DGMG':
        gen_graphs = gen_graphs_dgmg(eval_args)
    else:
        raise ValueError(f"Unknown model note: {train_args.note}")

    if os.path.isdir(eval_args.current_graphs_save_path):
        shutil.rmtree(eval_args.current_graphs_save_path)
    os.makedirs(eval_args.current_graphs_save_path)

    save_graphs(eval_args.current_graphs_save_path, gen_graphs)


def print_stats(
    node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref,
    edge_count_avg_pred, degree_mmd, clustering_mmd, orbit_mmd,
    forward_edge_avg_ref, forward_edge_avg_pred, backward_edge_avg_ref,
    backward_edge_avg_pred, forward_ratio_avg_ref, forward_ratio_avg_pred,
    backward_ratio_avg_ref, backward_ratio_avg_pred
):
    print('Node count avg: Test - {:.6f}, Generated - {:.6f}'.format(
        mean(node_count_avg_ref), mean(node_count_avg_pred)))
    print('Edge count avg: Test - {:.6f}, Generated - {:.6f}'.format(
        mean(edge_count_avg_ref), mean(edge_count_avg_pred)))

    print('Forward edge count avg: Test - {:.6f}, Generated - {:.6f}'.format(
        mean(forward_edge_avg_ref), mean(forward_edge_avg_pred)))
    print('Backward edge count avg: Test - {:.6f}, Generated - {:.6f}'.format(
        mean(backward_edge_avg_ref), mean(backward_edge_avg_pred)))

    print('Forward edge ratio avg: Test - {:.6f}, Generated - {:.6f}'.format(
        mean(forward_ratio_avg_ref), mean(forward_ratio_avg_pred)))
    print('Backward edge ratio avg: Test - {:.6f}, Generated - {:.6f}'.format(
        mean(backward_ratio_avg_ref), mean(backward_ratio_avg_pred)))

    print('MMD Degree - {:.6f}, MMD Clustering - {:.6f}, MMD Orbits - {:.6f}'.format(
        mean(degree_mmd), mean(clustering_mmd), mean(orbit_mmd)))

    print(LINE_BREAK)


if __name__ == "__main__":
    eval_args = ArgsEvaluate()
    train_args = eval_args.train_args

    print('Evaluating {}, run at {}, epoch {}'.format(
        train_args.fname, train_args.time, eval_args.num_epochs))

    # Generate graphs from the trained model (if requested)
    if eval_args.generate_graphs:
        generate_graphs(eval_args)

    random.seed(123)

    # Build dataset indices
    graphs = []
    for name in os.listdir(train_args.current_dataset_path):
        if name.endswith('.dat'):
            graphs.append(len(graphs))

    random.shuffle(graphs)
    split = int(0.90 * len(graphs))
    graphs_test_indices = graphs[split:]
    graphs_train_indices = graphs[:split]

    # Generated graph indices
    if not eval_args.generate_graphs:
        graphs_pred_indices = [
            i for i, name in enumerate(os.listdir(eval_args.current_graphs_save_path))
            if name.endswith('.dat')
        ]
    else:
        graphs_pred_indices = [i for i in range(eval_args.count)]

    print('Evaluating {}, run at {}, epoch {}'.format(
        train_args.fname, train_args.time, eval_args.num_epochs))

    print('Graphs generated - {}'.format(len(graphs_pred_indices)))
    print(len(graphs_test_indices))

    # Running aggregates
    node_count_avg_ref, node_count_avg_pred = [], []
    edge_count_avg_ref, edge_count_avg_pred = [], []
    forward_edge_avg_ref, forward_edge_avg_pred = [], []
    backward_edge_avg_ref, backward_edge_avg_pred = [], []
    forward_ratio_avg_ref, forward_ratio_avg_pred = [], []
    backward_ratio_avg_ref, backward_ratio_avg_pred = [], []

    degree_mmd, clustering_mmd, orbit_mmd = [], [], []

    total_pred = len(graphs_pred_indices)
    for i in range(0, total_pred, eval_args.metric_eval_batch_size):
        batch_cap = min(eval_args.metric_eval_batch_size, total_pred - i)
        if not graphs_test_indices:
            raise RuntimeError("No test graphs found in the dataset split.")
        batch_size = min(batch_cap, len(graphs_test_indices))

        # Reference (real) graphs for this batch
        graphs_ref_indices = random.sample(graphs_test_indices, batch_size)
        graphs_ref = load_graphs(train_args.current_dataset_path, graphs_ref_indices)
        graphs_ref = [patch_graph(g) for g in graphs_ref]

        # Generated graphs for this batch
        graphs_pred = load_graphs(eval_args.current_graphs_save_path, graphs_pred_indices[i: i + batch_size])
        graphs_pred = [patch_graph(g) for g in graphs_pred]

        # Basic metrics
        node_count_avg_ref.append(mean([len(G.nodes()) for G in graphs_ref]))
        node_count_avg_pred.append(mean([len(G.nodes()) for G in graphs_pred]))
        edge_count_avg_ref.append(mean([len(G.edges()) for G in graphs_ref]))
        edge_count_avg_pred.append(mean([len(G.edges()) for G in graphs_pred]))

        # DFS-based FW/BW analysis (consistent across models using minimal DFS code)
        ref_edge_stats = analyze_edge_directions(graphs_ref)
        pred_edge_stats = analyze_edge_directions(graphs_pred)

        if ref_edge_stats.get('skipped_graphs', 0) or pred_edge_stats.get('skipped_graphs', 0):
            print(f"[DFS] Skipped graphs (empty/invalid DFS code): "
                  f"ref={ref_edge_stats.get('skipped_graphs', 0)}, "
                  f"pred={pred_edge_stats.get('skipped_graphs', 0)}")

        forward_edge_avg_ref.append(ref_edge_stats['forward_count'])
        forward_edge_avg_pred.append(pred_edge_stats['forward_count'])
        backward_edge_avg_ref.append(ref_edge_stats['backward_count'])
        backward_edge_avg_pred.append(pred_edge_stats['backward_count'])
        forward_ratio_avg_ref.append(ref_edge_stats['forward_ratio'])
        forward_ratio_avg_pred.append(pred_edge_stats['forward_ratio'])
        backward_ratio_avg_ref.append(ref_edge_stats['backward_ratio'])
        backward_ratio_avg_pred.append(pred_edge_stats['backward_ratio'])

        # MMDs
        degree_mmd.append(metrics.stats.degree_stats(graphs_ref, graphs_pred))
        clustering_mmd.append(metrics.stats.clustering_stats(graphs_ref, graphs_pred))
        orbit_mmd.append(metrics.stats.orbit_stats_all(graphs_ref, graphs_pred))

        print('Running average of metrics:\n')
        print_stats(
            node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref, edge_count_avg_pred,
            degree_mmd, clustering_mmd, orbit_mmd,
            forward_edge_avg_ref, forward_edge_avg_pred, backward_edge_avg_ref,
            backward_edge_avg_pred, forward_ratio_avg_ref, forward_ratio_avg_pred,
            backward_ratio_avg_ref, backward_ratio_avg_pred
        )

    print('Evaluating {}, run at {}, epoch {}'.format(
        train_args.fname, train_args.time, eval_args.num_epochs))

    print_stats(
        node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref, edge_count_avg_pred,
        degree_mmd, clustering_mmd, orbit_mmd,
        forward_edge_avg_ref, forward_edge_avg_pred, backward_edge_avg_ref,
        backward_edge_avg_pred, forward_ratio_avg_ref, forward_ratio_avg_pred,
        backward_ratio_avg_ref, backward_ratio_avg_pred
    )
