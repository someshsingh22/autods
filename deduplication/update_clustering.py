import os
import re
import json
import argparse
import random
import numpy as np
import warnings
import time
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
from openai import OpenAI
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from openai_embedding import save_embeddings, get_embedding

import logging
import os
from datetime import datetime
from extract_hypothesis_from_log import convert_log_to_structured_hyp_mcts


def setup_logger(log_dir):
    """
    Set up and configure a custom logger that logs messages to both a file and the console.

    Args:
        log_dir (str): The directory path where the log file will be created.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger("clusterllm_logger")
    # Set the logger to capture all levels of log messages (DEBUG and above).
    logger.setLevel(logging.DEBUG)

    # Get the current date and time to generate a unique timestamp for the log file name
    current_datetime = datetime.now()
    fdt = current_datetime.strftime('%Y-%m-%d_%H-%M-%S')

    # Ensure the log directory exists; create it if it does not
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"clusterllm_log_{fdt}.log")
    # print("-"*100)
    # print(f"LOG PATH = {log_path}")
    # print("-"*100)

    # Create a file handler to write log messages to the log file.
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Define the log message format, including timestamp, log level, and message
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def np_converter(o):
    """Convert NumPy types to Python native types for JSON serialization."""
    if isinstance(o, (np.int64, np.int32)):
        return int(o)
    if isinstance(o, (np.float64, np.float32)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def load_jsonl_data(data_path):
    """
    Load data from a JSON Lines (JSONL) file.

    Args:
        data_path (str): The file path to the JSONL file.

    Returns:
        list[dict]: A list of dictionaries, each representing a record from the file.

    Raises:
        FileNotFoundError: If the file at data_path does not exist.
        json.JSONDecodeError: If any line in the file is not valid JSON.
    """
    with open(data_path, 'r') as f:
        return [json.loads(line) for line in f]


def hyp_dict_to_str(d):
    """
    Convert a hypothesis dictionary to a structured string format. The dictionary is expected to include the keys:
    'hypothesis', 'contexts', 'variables', and 'relationships'.

    Args:
        d (dict): A dictionary containing the hypothesis data with the keys:
                  'hypothesis', 'contexts', 'variables', and 'relationships'.

    Returns:
        str: A formatted string representing the hypothesis and its details.
    """
    s = f"""Hypothesis: {d["hypothesis"]}\n
Context: {d["contexts"]}\n\n
Variables: {d["variables"]}\n\n
Relationships: {d["relationships"]}"""
    return s


def retrieve_structured_hypothesis(args, sent):
    """
    Searches through a JSON Lines (JSONL) file for an entry where the 'hypothesis' key matches
    the provided sentence (sent) and retrieves it after formatting when a match has been found.

    Args:
        args: An object with an attribute 'structured_data_dir' that specifies the
              directory to the JSONL file containing structured hypothesis data.
        sent (str): The hypothesis sentence to search for in the file.

    Returns:
        str or None: A formatted string containing the hypothesis details if a matching
                     entry is found; otherwise, None.

    """
    structured_data_path = args.structured_data_dir + os.sep + f"dedup_{args.data_id}_{args.version}.jsonl"
    with open(structured_data_path, 'r') as f:
        for line in f:
            ele = json.loads(line)
            if ele["hypothesis"] == sent:
                return (f"Hypothesis: {ele['hypothesis']}\n"
                        f"Context: {ele['context']}\n"
                        f"Variables: {ele['variables']}\n"
                        f"Relationship: {ele['relationship']}")
    logger.debug(f"ERROR: Structured hypothesis not found for - {sent}")


def load_feat(args):
    """
    Load feature embeddings from an HDF5 file.

    Args:
        args: An object with an attribute 'feat_path' that specifies the path to the HDF5
              file containing the feature embeddings.

    Returns:
        numpy.ndarray: An array containing the feature embeddings extracted from the HDF5 file.
    """
    import h5py
    with h5py.File(args.feat_path, 'r') as f:
        X = np.asarray(f['embeds'])
    return X


def should_belong_to_same_cluster(sent1, sent2, n=30, threshold=0.2):
    """
    Determine if two hypothesis sets should be clustered together using an LLM.

    Args:
        sent1 (str): The first hypothesis set as a string.
        sent2 (str): The second hypothesis set as a string.
        n (int, optional): The number of LLM responses to generate. Defaults to 30.
        threshold (float, optional): The fraction threshold of "yes" responses required to
                                     return True. Defaults to 0.2.

    Returns:
        bool: True if the fraction of "yes" responses exceeds the threshold, otherwise False.

    """
    prompt = (
        f"""You are given two sets of hypotheses. Each set describes a context, the variables involved, and the statistical relationships between them. Your task is to determine if both sets indicate the same statistical behavior. Consider the following:

Context: The conditions or boundaries under which the relationship holds. Both sets must have identical contexts.
Variables: All variables must match. Even if their names differ, they must refer to the same concept.
Relationships: Each hypothesis may include one or more pairs of explanatory and response variables. The statistical relationship between these variables must be equivalent, regardless of how it is described.

Your answer should be either "Yes" or "No" with no additional explanation.

Hypothesis Set 1: 
{sent1}

Hypothesis Set 2: 
{sent2}

Answer:
"""
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        n=n,
        messages=[
            {"role": "system", "content": "You are a research scientist skilled at analyzing statistical hypotheses."},
            {"role": "user", "content": prompt}
        ]
    )
    answers = [choice.message.content.strip().lower() for choice in response.choices]
    logger.debug("LLM prompt:\n" + prompt)
    logger.debug(f"LLM responses: {answers}")
    yes_cnt = sum(1 for ans in answers if "yes" in ans)
    logger.debug(f"LLM decision: {yes_cnt} out of {n} responses indicated 'Yes' (threshold: {threshold})")
    return (yes_cnt / n) > threshold


def save_hac_dendrogram(Z, n, args):
    """
    Generate and save a dendrogram image from a Hierarchical Agglomerative Clustering (HAC) tree.

    Args:
        Z (array-like): The linkage matrix produced by hierarchical clustering.
        n (int): The starting index for labeling nodes. Labels will be assigned sequentially from n.
        data_id (str): An identifier for the dataset, used in naming the output image file.
        img_out_dir (str): The directory where the dendrogram image will be saved.

    Returns:
        None: The function saves the dendrogram image to the specified directory.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ddata = dendrogram(Z, ax=ax)

    # Store each merge row (icoord, dcoord) along with its "height"
    merges = []
    for i, (icoord, dcoord) in enumerate(zip(ddata['icoord'], ddata['dcoord'])):
        # Calculate the x,y coordinates for label placement
        x_mid = np.mean(icoord[1:3])
        y_mid = np.mean(dcoord[1:3])
        # Determine the merge "height" from the maximum dcoord
        height = max(dcoord)
        merges.append((i, x_mid, y_mid, height))

    # Sort the merges by ascending height so that lower merges get smaller IDs
    merges.sort(key=lambda x: x[3])

    # Label each merge from n upward based on the sorted order
    for rank, (i, x_mid, y_mid, height) in enumerate(merges):
        node_id = n + rank
        ax.text(x_mid, y_mid, f"{node_id}", va='center', ha='center', fontsize=8, color='red')

    ax.set_title("HAC Dendrogram with Node Annotations (Height-Sorted)")
    fig.tight_layout()

    dendro_path = os.path.join(args.img_out_dir, f"{args.data_id}_{args.version}_hac_dendrogram.png")
    plt.savefig(dendro_path)
    plt.close(fig)
    logger.info(f"HAC dendrogram saved to {dendro_path}")


def dict_to_str(d):
    """
    Convert a dictionary to a formatted string without specific punctuation.

    Args:
        d (dict): The dictionary to be converted.

    Returns:
        str: The formatted string representation of the dictionary.
    """
    s = json.dumps(d)
    s = re.sub(r'[\[\]\{\}\"]', '', s)
    s = s.replace(",", ";")
    return s


def relationships_to_str(relationships):
    """
    Convert a list of relationship dictionaries to a formatted string with corresponding relationship sets.

    Args:
        relationships (list[dict]): A list of dictionaries, each describing a set of relationships.

    Returns:
        str: A concatenated string with each relationship set labeled and formatted.
    """
    relationships_str = ""
    for idx, rel in enumerate(relationships):
        relationships_str += f"Relationship set {idx}: {dict_to_str(rel)}\n"
    return relationships_str


def generate_hac_llm_clustering(args):
    """
    Build a HAC tree from hypothesis embeddings (on deduplicated data) and re-evaluate each merge using LLM decisions.
    Only if the LLM approves is the merge performed.
    Returns the clustering for the deduplicated hypotheses as well as a mapping from original indices to deduplicated indices.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)

    # structured_data_path = args.structured_data_dir + os.sep + f"{args.data_id}_{args.version}.jsonl"
    structured_data_path = args.structured_data_dir + os.sep + f"dedup_{args.data_id}_{args.version}.jsonl"
    data = load_jsonl_data(structured_data_path)
    # Deduplicate: build list of unique structured hypothesis strings and mapping from original index to dedup index.
    dedup_inp = []  # list of unique hypothesis strings
    hyp_to_index = {}  # maps hypothesis string to dedup index
    orig_to_dedup = []  # for each original data point, store its dedup index
    dedup_hyp = []

    for d in data:
        # print(f"DATA POINT: {d}")
        hyp = d["hypothesis"]
        if hyp not in hyp_to_index:
            # rel = relationships_to_str(d.get("relationships",[]))
            dedup_hyp.append(hyp)
            dedup_inp.append(hyp_dict_to_str(d))
            hyp_to_index[hyp] = len(dedup_inp) - 1
        orig_to_dedup.append(hyp_to_index[hyp])

    logger.info(f"Deduplicated hypotheses: {len(dedup_inp)} unique out of {len(data)} total.")

    # X = np.array([get_embedding(text) for text in dedup_hyp])
    X = np.array(get_embedding(dedup_hyp))
    feat_path = args.feat_dir + os.sep + f"{args.data_id}_{args.version}.hdf5"
    save_embeddings(X, feat_path)
    n = len(dedup_inp)
    # print(f"Number of hypotheses = {len(dedup_hyp)}")
    # Compute HAC tree on unique hypotheses
    Z = linkage(X, method='ward')
    logger.debug("HAC tree (linkage matrix) computed on deduplicated hypotheses.")
    if args.save_visualization:
        save_hac_dendrogram(Z, n, args)

    # Initialize clusters, assignments, and mapping from HAC nodes to current clusters for deduped indices
    clusters = {i: [i] for i in range(n)}
    cluster_assignment = {i: i for i in range(n)}
    hac_to_current = {i: i for i in range(n)}
    # Store a representative (initially each index itself) for each cluster.
    cluster_rep = {i: i for i in range(n)}

    logger.info(f"Number of merges = {len(Z)}")
    for r, row in enumerate(Z):
        hac_node_id = n + r
        left_hac, right_hac = int(row[0]), int(row[1])
        left_current = hac_to_current.get(left_hac, None)
        right_current = hac_to_current.get(right_hac, None)

        # print(f"LEFT CURRENT = {left_current}")
        # print(f"RIGHT CURRENT = {right_current}")

        if left_current is None or right_current is None or left_current == right_current:
            logger.debug(f"Skipping HAC merge at node {hac_node_id} (clusters missing/already merged).")
            # print(f"Skipping HAC merge at node {hac_node_id} (clusters missing/already merged).")
            hac_to_current[hac_node_id] = left_current if left_current is not None else right_current
            continue
        # print(f"Node = {hac_node_id}")
        logger.debug(
            f"Evaluating HAC merge at node {hac_node_id}: candidate clusters {left_current} and {right_current}.")

        # Retrieve the stored representatives for each cluster.
        rep_left = cluster_rep[left_current]
        rep_right = cluster_rep[right_current]
        # Log candidate representations.
        logger.debug(
            f"Candidate representative for cluster {left_current}: index {rep_left}, text: {dedup_inp[rep_left]}...")
        logger.debug(
            f"Candidate representative for cluster {right_current}: index {rep_right}, text: {dedup_inp[rep_right]}...")

        # Retrieve structured hypotheses from our deduplicated list.
        struct_left = dedup_inp[rep_left]
        struct_right = dedup_inp[rep_right]

        llm_decision = should_belong_to_same_cluster(struct_left, struct_right,
                                                     n=args.n_decisions,
                                                     threshold=args.llm_decision_threshold)
        logger.debug(
            f"LLM decision for merging clusters {left_current} and {right_current}: {'ACCEPTED' if llm_decision else 'REJECTED'}.")
        # print(f"LLM decision for merging clusters {left_current} and {right_current}: {'ACCEPTED' if llm_decision else 'REJECTED'}.")

        if llm_decision:
            merged_cluster_id = min(left_current, right_current)
            other_cluster_id = max(left_current, right_current)
            clusters[merged_cluster_id] = clusters[left_current] + clusters[right_current]
            for idx in clusters[merged_cluster_id]:
                cluster_assignment[idx] = merged_cluster_id

            # Randomly choose one of the two candidate representatives as the new representative.
            merged_rep = random.choice([rep_left, rep_right])
            cluster_rep[merged_cluster_id] = merged_rep

            # Remove redundant cluster info.
            del clusters[other_cluster_id]
            del cluster_rep[other_cluster_id]

            hac_to_current[hac_node_id] = merged_cluster_id
            logger.debug(f"Merged clusters {left_current} and {right_current} into {merged_cluster_id}.")
            merged_info = [(idx, dedup_inp[idx][:30] + '...' if len(dedup_inp[idx]) > 30 else dedup_inp[idx]) for idx in
                           clusters[merged_cluster_id]]
            logger.debug(f"After merge, cluster {merged_cluster_id} contains: {merged_info}")
        else:
            hac_to_current[hac_node_id] = None
            logger.debug(f"Rejected HAC merge at node {hac_node_id}: clusters remain separate.")
        # print(f"HAC to current = {hac_to_current}")
    final_labels_dedup = [cluster_assignment[i] for i in range(n)]
    logger.debug(f"Final deduplicated cluster assignments: {final_labels_dedup}")
    logger.debug(f"Total unique clusters (deduped): {len(set(final_labels_dedup))}")
    print(f"Total unique clusters (deduped): {len(set(final_labels_dedup))}")

    if len(set(final_labels_dedup)) < 2 or len(set(final_labels_dedup)) == n:
        logger.debug("Clustering is trivial on deduplicated data. Skipping silhouette score calculation.")
    else:
        score = silhouette_score(X, final_labels_dedup, metric='euclidean')
        logger.debug(f"Silhouette score = {score:.3f} with {len(set(final_labels_dedup))} clusters (deduped).")

    # Map deduplicated cluster assignments back to original data.
    final_labels = [cluster_assignment[orig_to_dedup] for orig_to_dedup in orig_to_dedup]

    if args.save_clustering_output:
        save_text_to_cluster(args, data, final_labels)
        save_clusters_by_cluster_json(args, data, final_labels)
    return final_labels, clusters, cluster_assignment, orig_to_dedup


def save_clusters_by_cluster_json(args, data, final_labels):
    """
    Save another JSON file with each cluster number as key and a list of dictionaries
    (each containing node_id and text) as the value.
    """
    original_structured_data_path = args.structured_data_dir + os.sep + f"{args.data_id}_{args.version}.jsonl"
    org_hyps = load_jsonl_data(original_structured_data_path)
    seen = {}

    def add_to_cluster(cluster_id, node_id, text):
        key = (node_id, text)
        if key not in seen[cluster_id]:
            cluster_dict[cluster_id].append({"node_id": node_id, "text": text})
            seen[cluster_id].add(key)

    cluster_dict = {}
    for i, d in enumerate(data):
        cluster_id = final_labels[i]
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
            seen[cluster_id] = set()
        add_to_cluster(cluster_id, d["node_id"], d["hypothesis"])
        # cluster_dict[cluster_id].append({
        #         "node_id": d["node_id"],
        #         "text": d["hypothesis"]
        #     })
        for idx, dup in enumerate(org_hyps):
            if dup["hypothesis"] == d["hypothesis"]:
                # cluster_dict[cluster_id].append({
                #     "node_id": dup["node_id"],
                #     "text": dup["hypothesis"]
                # })
                add_to_cluster(cluster_id, dup["node_id"], dup["hypothesis"])

    clusters_save_path = os.path.join(
        args.out_dir,
        f"clusters_by_cluster_{args.data_id}_{args.version}.json"
    )
    with open(clusters_save_path, "w") as f:
        json.dump(cluster_dict, f, indent=2, default=np_converter)
    logger.info(f"Clusters by cluster mapping saved to {clusters_save_path}")


def save_text_to_cluster(args, data, final_labels):
    """Save the mapping of original input texts to their assigned clusters."""
    # For each original data point, use its hypothesis and the mapped cluster label.
    final_text_clusters = [{"node_id": d["node_id"], "text": d["hypothesis"], "cluster": final_labels[i]}
                           for i, d in enumerate(data)]
    text_clusters_save_path = os.path.join(
        args.out_dir,
        f"text_clusters_silhouette_{args.data_id}_{args.version}.json"
    )
    with open(text_clusters_save_path, "w") as f:
        json.dump(final_text_clusters, f, indent=2, default=np_converter)
    logger.info(f"Text with cluster assignments saved to {text_clusters_save_path}")


def main(args):
    start_time = time.time()

    logger.debug(f"------------------ Logger for Data id = {args.data_id} ------------------")
    # logger.debug(f"Saving = {args.save}")
    logger.debug(f"Run version = {args.version}")
    # logger.debug(f"Data path = {args.data_path}")
    original_structured_data_path = args.structured_data_dir + os.sep + f"{args.data_id}_{args.version}.jsonl"
    structured_data_path = args.structured_data_dir + os.sep + f"dedup_{args.data_id}_{args.version}.jsonl"
    feat_path = args.feat_dir + os.sep + f"{args.data_id}_{args.version}.hdf5"
    convert_log_to_structured_hyp_mcts(args.node_log_path, original_structured_data_path, structured_data_path)
    logger.debug(f"Embeddings path = {feat_path}")
    logger.debug(f"Structured hypothesis path = {structured_data_path}")
    logger.debug(f"Results will be saved in {args.out_dir}")
    # data_path = os.path.join(input_dir, f"{data_id}.jsonl")
    # data = load_data(structured)
    # if data:
    # logger.debug("Data loading successful.")
    # max_clusters = len(data) - 1
    # logger.debug(f"Max clusters possible = {max_clusters}")

    os.makedirs(args.out_dir, exist_ok=True)

    final_labels, clusters, cluster_assignment, orig_to_dedup = generate_hac_llm_clustering(args)

    end_time = time.time()
    print(f"Total time taken = {end_time - start_time:.2f} seconds")
    logger.debug(f"Total time taken = {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    # Initialize OpenAI client and suppress warnings
    client = OpenAI()
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default=1,
                        help="Version of the output being saved 1. This is to help you track multiple runs on the same data")

    parser.add_argument("--data_id", type=str, required=True,
                        help="Filename, without any extensions, to be used for saving all intermediate results")

    parser.add_argument("--dataset", type=str, default="autodv",
                        help="Name of the dataset being used. Default is 'autodv'.")

    parser.add_argument("--node_log_path", type=str, required=True,
                        help="File path where node logs are present. This argument is required.")

    parser.add_argument("--feat_dir", type=str, default=os.getcwd() + os.sep + "embeddings",
                        help="Directory where feature embeddings are to be saved.")

    parser.add_argument("--img_out_dir", type=str, default=os.getcwd() + os.sep + "visualizations",
                        help="Directory where dendrogram images will be saved.")

    parser.add_argument("--structured_data_dir", default=os.getcwd() + os.sep + "structured_hypotheses",
                        help="Directory where structured data file (in JSONL format) will be saved.")

    parser.add_argument("--log_dir", type=str, default=os.getcwd() + os.sep + "clusterllm_logs",
                        help="Directory for saving log files.")

    parser.add_argument("--save_clustering_output", type=str, default=True,
                        help="Flag to indicate whether to save the clustering output.")

    parser.add_argument("--save_visualization", type=str, default=True,
                        help="Flag to save visualization images.")

    parser.add_argument("--out_dir", type=str, default=os.getcwd() + os.sep + "clusterllm_output",
                        help="Output directory for final results.")

    # Uncomment and adjust these arguments if needed:
    # parser.add_argument("--min_clusters", type=int, default=1,
    #                     help="Minimum number of clusters for clustering algorithm. Default is 1.")
    # parser.add_argument("--max_clusters", type=int, default=max_clusters,
    #                     help="Maximum number of clusters for clustering algorithm.")

    parser.add_argument("--seed", type=int, default=100,
                        help="Seed for reproducibility.")

    parser.add_argument("--n_decisions", type=int, default=10,
                        help="Number of decisions (LLM completions) to generate for clustering decisions.")

    parser.add_argument("--llm_decision_threshold", type=float, default=0.2,
                        help="Threshold fraction of 'yes' responses required for LLM-based clustering decisions.")

    args = parser.parse_args()

    logger = setup_logger(args.log_dir)
    main(args)