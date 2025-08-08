import json
import csv
import argparse

from src.utils import try_loading_dict


class ArgParser(argparse.ArgumentParser):
    def __init__(self, group=None):
        super().__init__(description='Get surprising nodes from MCTS logs')
        self.add_argument('--in_fpath', type=str, required=True,
                          help='mcts_nodes.json file path or directory containing mcts_node_*.json files')
        self.add_argument('--out_fpath', type=str, required=True, help='output CSV file path')


def nodes_to_csv(nodes_or_json_path, out_fpath):
    from src.mcts_utils import get_nodes, get_node_level_idx  # Import here to avoid circular import issues
    mcts_nodes = get_nodes(nodes_or_json_path)

    csv_list = []
    for node in mcts_nodes:
        csv_node = {}
        node_level, node_idx = get_node_level_idx(node)

        if node_level in [0, 1]:
            continue

        try:
            prior_mean = node["prior"]["mean"]
            posterior_mean = node["posterior"]["mean"]
        except:
            prior_mean = None
            posterior_mean = None

        csv_node['id'] = node['id'].replace('node_', '')
        csv_node['success'] = node.get('success', False)

        csv_node['surprisal'] = node.get('surprising', None)
        csv_node['degree_of_surprisal'] = abs(
            posterior_mean - prior_mean) if prior_mean is not None and posterior_mean is not None else None
        csv_node['direction_of_surprisal'] = None
        if prior_mean is not None and posterior_mean is not None:
            csv_node['direction_of_surprisal'] = 'negative' if posterior_mean < prior_mean else (
                'unchanged' if posterior_mean == prior_mean else 'positive')
        csv_node['prior_belief'] = prior_mean
        csv_node['posterior_belief'] = posterior_mean

        csv_node['hypothesis'] = node['hypothesis']
        experiment_plan = node['experiment_plan']
        csv_node['experiment_plan'] = f"Objective: {experiment_plan.get('objective', 'N/A')}\n" \
                                      f"Steps: {experiment_plan.get('steps', 'N/A')}\n" \
                                      f"Deliverables: {experiment_plan.get('deliverables', 'N/A')}"
        csv_node['analysis'] = node.get('analysis', 'N/A')
        csv_node['review'] = node.get('review', 'N/A')

        csv_list.append(csv_node)
    csv_list.sort(key=lambda x: x['degree_of_surprisal'] if x['degree_of_surprisal'] is not None else float('-inf'),
                  reverse=True)

    with open(out_fpath, 'w', newline='') as csv_file:
        fieldnames = ['id', 'success', 'hypothesis', 'surprisal', 'degree_of_surprisal', 'direction_of_surprisal',
                      'prior_belief', 'posterior_belief', 'experiment_plan', 'analysis', 'review']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_list:
            row = {k: (v if v is not None else '') for k, v in row.items()}
            writer.writerow(row)

    print(f"[CSV] MCTS nodes (n={len(csv_list)}; skipping root) saved to {out_fpath}.\n")


if __name__ == '__main__':
    parser = ArgParser()
    args = parser.parse_args()
    nodes_to_csv(args.in_fpath, args.out_fpath)
