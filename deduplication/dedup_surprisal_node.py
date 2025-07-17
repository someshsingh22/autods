import os
import json


def dedup_surprisal_node(input_file, output_file):
    # load original data
    with open(input_file, "r") as f:
        nodes = json.load(f)

    # filter for surprising == True
    print(f"Count of surprisal nodes: {len(nodes)}")
    filtered = [n for n in nodes if n.get("surprising") is True]

    # save filtered data
    with open(output_file, "w") as f:
        json.dump(filtered, f, indent=2)

    print(f"Saved {len(filtered)} nodes to {output_file}\n")


if __name__ == "__main__":
    input_dir = "proxy_data_2/interesting_mcts_nodes"
    output_dir = "proxy_data_2/interesting_surprisal_mcts_nodes"
    for filename in os.listdir(input_dir):
        input_file = input_dir + os.sep + filename
        output_file = os.path.join(output_dir, filename)
        # make sure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        dedup_surprisal_node(input_file, output_file)
