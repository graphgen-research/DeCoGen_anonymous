import os
import subprocess as sp
import numpy as np

COUNT_START_STR = 'orbit counts: \n'


def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for (u, v) in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges


def orca(graph):
    """
    Run ORCA to compute node orbit counts for the given NetworkX graph.
    Returns a NumPy array of shape (num_nodes, num_orbits).
    """
    tmp_fname = 'tmp.txt'
    # Write edge list in ORCA format
    with open(tmp_fname, 'w') as f:
        f.write(f"{graph.number_of_nodes()} {graph.number_of_edges()}\n")
        for (u, v) in edge_list_reindexed(graph):
            f.write(f"{u} {v}\n")

    # Call ORCA binary
    output = sp.check_output(['./orca/orca', 'node', '4', tmp_fname, 'std'])
    text = output.decode('utf8')

    # Extract the orbit counts section
    idx = text.find(COUNT_START_STR)
    if idx == -1:
        raise RuntimeError("ORCA output did not contain orbit counts header")
    counts_section = text[idx + len(COUNT_START_STR):].strip()

    # Parse lines robustly, skipping non-numeric tokens
    rows = []
    for line in counts_section.splitlines():
        tokens = line.strip().split()
        ints = []
        for tok in tokens:
            # Remove trailing punctuation and ensure it's numeric
            clean = tok.rstrip(':,')
            if clean.lstrip('-').isdigit():
                ints.append(int(clean))
        if ints:
            rows.append(ints)

    node_orbit_counts = np.array(rows, dtype=int)

    # Clean up temporary file
    try:
        os.remove(tmp_fname)
    except OSError:
        pass

    return node_orbit_counts