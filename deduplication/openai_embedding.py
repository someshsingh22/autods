import argparse
import json
import os
import h5py
import numpy as np
from openai import OpenAI

client = OpenAI()
import time


def load_data(data_path):
    """
    Load and process data from a JSON Lines (JSONL) file.
    This function reads a file where each line is expected to be a JSON object.
    For each valid JSON object, it extracts the keys 'hypothesis', 'contexts',
    'variables', and 'relationships', then formats them into a structured string.
    Args:
        data_path (str): The file path to the JSONL file.
    Returns:
        list: A list of formatted strings, each containing hypothesis details.
    """
    texts = []
    # Open the file with UTF-8 encoding
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            try:
                # Attempt to parse the JSON data on the current line
                data = json.loads(line)
                # Format the extracted data into a structured string
                t = (
                    f"Hypothesis: {data['hypothesis']}\n"
                    f"Context: {data['contexts']}\n"
                    f"Variables: {data['variables']}\n"
                    f"Relationship: {data['relationships']}"
                )
                texts.append(t)
            except json.JSONDecodeError as e:
                # Print an error message if JSON decoding fails
                print(f"Skipping line due to JSON error: {e}")
    return texts


def get_embedding(texts, model="text-embedding-3-large", batch_size=128):
    """
    Compute embeddings for a list of texts using the OpenAI Embeddings API.
    Args:
        texts (list): A list of text strings to be embedded.
        model (str, optional): The identifier for the embedding model to use.
        batch_size (int, optional): The number of texts to process in one API call.
    Returns:
        numpy.ndarray: An array of embeddings for the input texts.
    """
    # print(f"Texts:\n{texts}")
    all_embeddings = []
    # Process the texts in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        # Request embeddings for the current batch from the API
        response = client.embeddings.create(input=batch, model=model)
        for item in response.data:
            # Convert the embedding to a NumPy array and add it to the list
            all_embeddings.append(np.array(item.embedding))
    return np.array(all_embeddings)


def save_embeddings(embeddings, out_path):
    """
    Save computed embeddings to an HDF5 file.
    Args:
        embeddings (numpy.ndarray): The array of embeddings to be saved.
        out_path (str): The file path where the HDF5 file will be created.
    Returns:
        None
    """
    # Open an HDF5 file in write mode
    with h5py.File(out_path, "w") as f:
        # Create a dataset named 'embeds' and store the embeddings00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
        f.create_dataset("embeds", data=embeddings)
    print(f"Embeddings saved to {out_path}")


def main(args):
    start_time = time.time()
    path = args.data_path
    out_dir = args.out_dir

    if not os.path.isfile(path):
        print(f"Input file at {path} does not exist!")
        return

    if not path.endswith(".jsonl"):
        print("Error: The input file must have a .jsonl extension.")
        return

    texts = load_data(path)
    if not texts:
        print("No valid texts found. Exiting.")
        return

    embeddings = get_embeddings(
        texts,
        model="text-embedding-ada-002",  # Use desired embedding model here
        batch_size=128
    )

    base_name = os.path.basename(path).replace(".jsonl", "")
    out_path = os.path.join(out_dir, base_name + "_structured_hypothesis_openai_embeds_v3.hdf5")
    save_embeddings(embeddings, out_path)
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str, help="Path to the input .jsonl file.")
    parser.add_argument("--out_dir", type=str, default=".", help="Directory to save the output HDF5 file.")
    args = parser.parse_args()

    main(args)
