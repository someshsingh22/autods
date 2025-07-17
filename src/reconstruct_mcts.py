import os
import json
from glob import glob


def load_nodes(node_dir):
    node_files = glob(os.path.join(node_dir, 'mcts_node_*.json'))
    nodes = {}
    for f in node_files:
        with open(f, 'r') as fp:
            node = json.load(fp)
            key = (node['level'], node['node_idx'])
            nodes[key] = node
    return nodes


def collect_logs_for_resumption(node_dir, output_file=None):
    nodes = load_nodes(node_dir)
    # Create fake root node (0,0) if missing
    if (0, 0) not in nodes:
        nodes[(0, 0)] = {
            "level": 0,
            "node_idx": 0,
            "parent_idx": None,
            "query": None,
            "visits": 0,
            "value": 0,
            "surprising": None,
            "hypothesis": None,
            "prior": None,
            "posterior": None,
            "messages": [],
            "creation_index": 0
        }
    # Build parent and children relationships
    parent_map = {}
    children = {}
    for key, node in nodes.items():
        parent_idx = node.get('parent_idx')
        parent_level = node.get('level', 0) - 1
        if parent_idx is not None:
            parent_key = (parent_level, parent_idx)
            children.setdefault(parent_key, []).append(key)
            parent_map[key] = parent_key
        else:
            parent_map[key] = None
    # Sort nodes by creation_index ascending for backprop order
    node_keys_by_creation = sorted(nodes.keys(), key=lambda k: nodes[k].get('creation_index', 0))
    # Reset visits and value
    for k in nodes:
        nodes[k]['visits'] = 0
        nodes[k]['value'] = 0
    # Backpropagate visits and value as in run_mcts
    for key in node_keys_by_creation:
        node = nodes[key]
        # Surprising: treat None as 0, True as 1, False as 0
        reward = int(bool(node.get('surprising', False)))
        n = key
        while n is not None:
            nodes[n]['visits'] += 1
            nodes[n]['value'] += reward
            n = parent_map[n]
    # Output as list sorted by creation_index descending
    node_list = sorted(nodes.values(), key=lambda n: n.get('creation_index', 0), reverse=True)
    if output_file is None:
        return node_list
    with open(output_file, 'w') as fp:
        json.dump(node_list, fp, indent=2)


# Example usage:
# reconstruct_mcts('mcts_gene/20250428-184216', 'mcts_gene/20250428-184216/mcts_nodes.json')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reconstruct MCTS nodes from directory.")
    parser.add_argument('node_dir', type=str, help='Directory containing MCTS node files.')
    parser.add_argument('output_file', type=str, help='Output file for reconstructed nodes.')
    args = parser.parse_args()

    collect_logs_for_resumption(args.node_dir, args.output_file)
