import json
import re
import os

def dict_to_str(d):
    """
    Convert a dictionary to a string without certain characters.

    Args:
        d (dict): The dictionary to convert.

    Returns:
        str: The formatted string representation of the dictionary.
    """
    s = json.dumps(d)
    s = re.sub(r'[\[\]\{\}\"]', '', s)  # Remove [ ] { } "
    s = s.replace(",", ";")  # Replace commas with semicolons
    return s


def load_node_logs(directory, level=None, node_idx=None):
    """
    Loads and parses log files in a directory that match the format
    'node_<level>_<index>.log', saving the contents in a dictionary.

    Args:
        directory (str): The path to the directory containing the log files.
        level (int, optional): The node level to filter by. Only logs for this level are loaded.
        node_idx (int, optional): The node index to filter by. Only logs for this node are loaded.

    Returns:
        dict: A dictionary where keys are tuples (level, index) and values are the extracted node messages.
              Returns an empty dict if no matching files are found or an error occurs.
    """
    result = {}
    node_lst = []
    try:
        for filename in os.listdir(directory):
            # Valid log file names should match the pattern node_<level>_<index>.log
            match = re.match(r"node_(\d+)_(\d+)\.json", filename)  # Updated extension

            if match:
                node_lst.append(filename.split(".")[0])
                file_level = int(match.group(1))
                file_index = int(match.group(2))
                # Skip files that don't match the given level or node_idx
                if (level is not None and file_level != level) or (node_idx is not None and file_index != node_idx):
                    continue
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, "r") as f:
                        # Load the entire JSON content from the file
                        log_content = json.load(f)
                        # Extract messages from the node log and save them in the result dictionary
                        result[(file_level, file_index)] = extract_node_messages(log_content)
                except OSError as e:
                    print(f"Error reading {filename}: {e}")
        return node_lst, result
    except FileNotFoundError:
        print(f"Directory not found: {directory}")
        return {}
    except OSError as e:
        print(f"OS error occurred: {e}")
        return {}

def get_structured_hypothesis_mcts(node):

    # extracted_info = []
    # for i, node in enumerate(nodes):
    level = node.get("level")
    node_idx = node.get("node_idx")


    h = node.get("hypothesis", {})
    if h:
        # print(f"\n-- Node_{level}_{node_idx} --")
        # print("Hypothesis:", h.get("hypothesis"))

        dims = h.get("dimensions", {})
        contexts = dims.get("contexts", [])
        variables = dims.get("variables", [])
        relationships = dims.get("relationships", [])

        relationships_str = ""
        for idx, rel in enumerate(relationships):
            relationships_str += f"Relationship set {idx}: {dict_to_str(rel)}\n"

        return(h.get("hypothesis"), {
        "node_id": f"node_{level}_{node_idx}",
        "hypothesis": h.get("hypothesis"),
        "variables": ", ".join(variables),
        "relationships": relationships_str,
        "contexts": dict_to_str(contexts),
        })
    return (None, None)

def convert_log_to_structured_hyp_mcts(mcts_nodes_json_path, out_path, dedup_out_path):
    with open(mcts_nodes_json_path) as f:
        raw_nodes_lst = json.load(f)   # assume this is a list of dicts

    all_hyp = []
    node_lst = []

    # Iterate over each log entry in the node logs dictionary
    for node in raw_nodes_lst:
        # print(node, type(node))
        hyp, node_obj = get_structured_hypothesis_mcts(node)
        if node_obj:
            node_lst.append(node_obj)

    save_hyp_mcts(node_lst, out_path, dedup_out_path)
    return

def get_structured_hypothesis(node_logs):
    """
    Extract structured hypothesis from node logs, by looking for messages from the "hypothesis_generator"
    to extract hypothesis details, including variables, relationships, and contexts.

    Args:
        node_logs (list): A list of dictionaries representing node log messages.

    Returns:
        list: A list of extracted hypothesis information as dictionaries. Each dictionary contains:
            - "hypothesis": The hypothesis text.
            - "variables": A string of variables separated by commas.
            - "relationships": A formatted string of relationships.
            - "contexts": A string representation of contexts.
    """
    if not node_logs:
        return None, []  # Return if there are no logs

    messages = node_logs

    # Find the last occurrence of a message with the name "user_proxy" by iterating in reverse.
    start_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("name") == "user_proxy":
            start_idx = i
            break

    if start_idx is None:
        return None, []  # Return if no "user_proxy" message is found

    node_messages = messages[start_idx:]

    # Initialize variables to store the hypothesis and extracted information.
    current_hypothesis = None
    extracted_info = []
    for msg in node_messages:
        if msg.get("name") == "hypothesis_generator":
            try:
                # Parse the content of the hypothesis_generator message
                content = json.loads(msg["content"])
                hypothesis = content.get("hypothesis", "")
                dimensions = content.get("dimensions", {})
                variables = dimensions.get("variables", [])
                relationships = dimensions.get("relationships", [])
                contexts = dimensions.get("contexts", [])

                # Build a string for relationships by iterating over each relationship
                relationships_str = ""
                for idx, rel in enumerate(relationships):
                    relationships_str += f"Relationship set {idx}: {dict_to_str(rel)}\n"

                # Append the extracted hypothesis info as a dictionary
                extracted_info.append({
                    "hypothesis": hypothesis,
                    "variables": ", ".join(variables),
                    "relationships": relationships_str,
                    "contexts": dict_to_str(contexts),
                })
                # If a hypothesis is already found, stop further iterations to avoid duplicates.
                if current_hypothesis:
                    break
            except (json.JSONDecodeError, TypeError):
                continue

    return extracted_info


def extract_node_messages(json_data):
    """
    Extract messages starting from the last occurrence of a message with name 'user_proxy'.

    Args:
        json_data (list): A list of dictionaries representing JSON data from a log file.

    Returns:
        list: A list of dictionaries containing messages from the last "user_proxy" message onward.
              Returns an empty list if no "user_proxy" messages are found.
    """
    if not isinstance(json_data, list):
        return []  # Return empty list if input is not a list

    # Identify all indices where the agent name is "user_proxy"
    user_proxy_indices = [i for i, msg in enumerate(json_data) if msg.get("name") == "user_proxy"]

    if not user_proxy_indices:
        return []  # Return empty list if no user_proxy messages are found

    last_user_proxy_index = user_proxy_indices[-1]
    # Return all messages from the last user_proxy message onward
    return json_data[last_user_proxy_index:]


def extract_experiment_info(filename):
    """
    Extract hypothesis and related experiment information from a log file.

    The function opens a log file, loads its JSON content,
    filters messages generated by the "hypothesis_generator", and extracts hypothesis details.

    Args:
        filename (str): The path to the log file.

    Returns:
        list: A list of dictionaries containing extracted experiment information. Each dictionary includes:
            - "hypothesis": The hypothesis text.
            - "variables": A string of variables separated by commas.
            - "relationships": A formatted string of relationships.
            - "contexts": A string representation of contexts.
    """
    with open(filename, 'r') as f:
        # Assume the file contains a JSON list of message objects.
        messages = json.load(f)

    # Filter for messages with from "hypothesis_generator"
    exp_generator_msgs = [msg for msg in messages if msg.get("name") == "hypothesis_generator"]

    extracted_info = []

    for msg in exp_generator_msgs:
        content = msg.get("content", "")
        # print("="*50, "Content", "="*50)
        # print(content)
        try:
            # Parse the content as JSON
            data = json.loads(content)
        except json.JSONDecodeError:
            continue

        # print("="*50, "Content parsed to json", "="*50)
        # print(data)
        # Extract hypothesis and its dimensions from the parsed JSON data
        hypothesis = data.get("hypothesis", "")
        dimensions = data.get("dimensions", {})
        variables = dimensions.get("variables", [])
        relationships = dimensions.get("relationships", [])
        contexts = dimensions.get("contexts", [])

        # print("RELATIONSHIPS:\n", relationships)

        # Build a string representation for relationships
        relationships_str = ""
        for idx, rel in enumerate(relationships):
            relationships_str += f"Relationship set {idx}: {dict_to_str(rel)}\n"

        # Append the extracted experiment info as a dictionary
        extracted_info.append({
            "hypothesis": hypothesis,
            "variables": ", ".join(variables),
            "relationships": relationships_str,
            "contexts": dict_to_str(contexts),
        })

    return extracted_info


def save_hyp_mcts(hypothesis, out_path, dedup_out_path):
    """
    Save hypothesis information to a file in JSONL format.

    This function writes each hypothesis dictionary as a separate JSON line in the output file.

    Args:
        hypothesis (list): A list of dictionaries containing hypothesis information.
        out_path (str): The file path where the hypothesis data will be saved.

    Returns:
        None
    """
    # Write each hypothesis dictionary as a JSON line to the output file.
    # for hyp in

    dedup_hyp, dedup_node_list = deduplicate_dicts(hypothesis)
    # print(node_lst)
    print("-------------------------------------- STRUCTURED HYPOTHESIS ----------------------------------------")
    print(f"Dedup structured output location: {dedup_out_path}")
    print(f"Total number of deduplicated structured hypotheses = {len(dedup_hyp)}")
    print(f"Structured output location: {out_path}")
    print(f"Original number of structured hypotheses = {len(hypothesis)}")
    print("-----------------------------------------------------------------------------------------------------")
    with open(out_path, "w") as outfile:
        for item in hypothesis:
            json_line = json.dumps(item)
            outfile.write(json_line + "\n")

    with open(dedup_out_path, "w") as outfile:
        for item in dedup_hyp:
            json_line = json.dumps(item)
            outfile.write(json_line + "\n")
    return

def save_hyp(hypothesis, node_lst, out_path, dedup_out_path):
    """
    Save hypothesis information to a file in JSONL format.

    This function writes each hypothesis dictionary as a separate JSON line in the output file.

    Args:
        hypothesis (list): A list of dictionaries containing hypothesis information.
        out_path (str): The file path where the hypothesis data will be saved.

    Returns:
        None
    """
    # Write each hypothesis dictionary as a JSON line to the output file.
    # for hyp in

    dedup_hyp, dedup_node_list = deduplicate_dicts(hypothesis)
    print("-------------------------------------- STRUCTURED HYPOTHESIS ----------------------------------------")
    print(f"Dedup structured output location: {dedup_out_path}")
    print(f"Total number of structured hypotheses = {len(dedup_hyp)}")
    print(f"Structured output location: {out_path}")
    print(f"Total number of structured hypotheses = {len(hypothesis)}")
    print("-----------------------------------------------------------------------------------------------------")
    with open(out_path, "w") as outfile:
        for item, node_id in zip(hypothesis, node_lst):
            item["node_id"] = node_id
            json_line = json.dumps(item)
            outfile.write(json_line + "\n")

    with open(dedup_out_path, "w") as outfile:
        for item, node_id in zip(dedup_hyp, dedup_node_list):
            item["node_id"] = node_id
            json_line = json.dumps(item)
            outfile.write(json_line + "\n")


def deduplicate_dicts(dict_list):
    """
    Remove duplicate dictionaries from a list.

    The function converts each dictionary to a JSON string (with sorted keys) to identify duplicates.
    Only unique dictionaries are retained.

    Args:
        dict_list (list): A list of dictionaries.

    Returns:
        list: A list of unique dictionaries with duplicates removed.
    """
    seen = set()
    unique_list = []
    unique_node_list = []
    for d in dict_list:
        dict_str = d["hypothesis"]#json.dumps(d, sort_keys=True)
        if dict_str not in seen:
            seen.add(dict_str)
            unique_list.append(d)
            unique_node_list.append(d["node_id"])
    return unique_list, unique_node_list


def convert_log_to_structured_hyp(node_log_dir, out_path, dedup_out_path):
    """
    Convert node log files to a structured hypothesis format and save the result.

    Args:
        node_log_dir (str): The directory containing node log files.
        out_path (str): The file path where the structured hypothesis data will be saved.

    Returns:
        None
    """
    structured_hypothesis = []
    # Load all node logs from the directory
    node_lst, node_logs = load_node_logs(node_log_dir)

    all_hyp = []
    # Iterate over each log entry in the node logs dictionary
    for key, node_id in zip(node_logs, node_lst):
        hypothesis_obj = get_structured_hypothesis(node_logs[key])
        # print(f"HYPOTHESIS OBJECT = {hypothesis_obj}")
        hypothesis_obj[0]["node_id"] = node_id
        all_hyp += hypothesis_obj
    # print(f"Number of hypotheses = {len(all_hyp)}")
    # print(f"Number of nodes = {len(node_lst)}")
    # Deduplicate the extracted hypothesis information and save to the output file
    # print(f"OUTPUT PATH = {out_path}")
    save_hyp(all_hyp, node_lst, out_path, dedup_out_path)
    return


if __name__ == "__main__":
    # Set the directory for node log files and the output file path for structured hypotheses.
    # log_dir = "/home/mchakravorty_umass_edu/eval_metric/autodv/clusterllm_granularity/datasets/data/n4_k2/20250326-150831"
    # out_path = "/home/mchakravorty_umass_edu/eval_metric/autodv/clusterllm_granularity/structured_hypothesis_data/20250326-150831_w_node_id.jsonl"
    # dedup_out_path = "/home/mchakravorty_umass_edu/eval_metric/autodv/clusterllm_granularity/structured_hypothesis_data/dedup_20250326-150831_w_node_id.jsonl"
    # convert_log_to_structured_hyp(log_dir, out_path, dedup_out_path)

    #Updated parameters
    log_path = "/home/mchakravorty_umass_edu/eval_metric/autodv/clusterllm_granularity/mcts_data/mcts_nodes.json"
    out_path = "/home/mchakravorty_umass_edu/eval_metric/autodv/clusterllm_granularity/mcts_results/sample_structured_data_output.jsonl"
    dedup_out_path = "/home/mchakravorty_umass_edu/eval_metric/autodv/clusterllm_granularity/mcts_results/sample_dedup_data_output.jsonl"
    convert_log_to_structured_hyp_mcts(log_path, out_path, dedup_out_path)