import os
import json
import argparse
import warnings

# Import OpenAI client (make sure the openai package is installed and configured)
from openai import OpenAI

client = OpenAI()

warnings.filterwarnings('ignore')


def np_converter(o):
    """Helper function for JSON serialization of NumPy types (if any)."""
    if isinstance(o, (int, float, list, dict)):
        return o
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def load_data(file_path):
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def analyze_sentence(sentence):
    """
    Calls OpenAI GPT-4 API to break the given hypothesis into three parts:
    context, variables, and relationship. Returns a dictionary with the parsed results.
    """
    prompt = (
        "Break the given hypothesis, which describes a relationship between variables in a context, "
        "down into 3 parts. Context indicates boundary conditions (e.g., 'for men over the age of 30' or 'in Asia and Europe'). "
        "Variables are the set of concepts (e.g., gender, age, income) and Relationship describes how they interact (e.g., quadratic, inversely proportional).\n\n"
        "<Example 1>\n"
        "Hypothesis: There are three distinct clusters of individuals based on socio-economic status, ability, and class percentile, although these clusters show some overlap.\n\n"
        "Context: unbounded or full dataset\n\n"
        "Variables: socio-economic status, ability, class percentile\n\n"
        "Relationship: The variables form three distinct clusters with some overlap, suggesting additional factors may be involved.\n"
        "</Example 1>\n\n"
        "Now it's your turn - \n"
        f"Hypothesis: {sentence}\n"
        "Context:\n"
        "Variables:\n"
        "Relationship:"
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system",
             "content": "You are a research scientist, capable of understanding and analyzing hypotheses determined after statistical analysis of certain datasets."},
            {"role": "user", "content": prompt}
        ]
    )
    # Combine the original hypothesis with the GPT response.
    result_text = f"Hypothesis: {sentence}\n" + response.choices[0].message.content

    # Parse the result text into its components.
    analysis = {}
    for line in result_text.splitlines():
        if line.startswith("Hypothesis:"):
            analysis["hypothesis"] = line[len("Hypothesis:"):].strip()
        elif line.startswith("Context:"):
            analysis["context"] = line[len("Context:"):].strip()
        elif line.startswith("Variables:"):
            analysis["variables"] = line[len("Variables:"):].strip()
        elif line.startswith("Relationship:"):
            analysis["relationship"] = line[len("Relationship:"):].strip()
    return analysis


def save_analysis_results(analysis_results, output_file):
    """Save each analysis result as a separate JSON line in the output file."""
    with open(output_file, 'w') as f:
        for result in analysis_results:
            f.write(json.dumps(result, default=np_converter) + "\n")


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(
    #     description="Analyze hypotheses using GPT-4 and save the analysis to a JSONL file."
    # )
    # parser.add_argument("--input_file", type=str, required=True,
    #                     help="Path to the input JSONL file containing sentences (with an 'input' key)")
    # parser.add_argument("--output_file", type=str, default="/home/mchakravorty_umass_edu/eval_metric/autodv/clusterllm_granularity/results/structured_hypothesis")
    # args = parser.parse_args()

    # Load the input data from the JSONL file.
    input_dir = "/home/mchakravorty_umass_edu/eval_metric/autodv/clusterllm_granularity/datasets"
    out_dir = "/home/mchakravorty_umass_edu/eval_metric/autodv/clusterllm_granularity/results/structured_hypothesis"
    for filename in os.listdir(input_dir):
        if filename.endswith(".jsonl"):
            # name = filename.replace(".jsonl","")
            output_file = out_dir + os.sep + filename
            data = load_data(input_dir + os.sep + filename)

            analysis_results = []
            # For each entry, extract the sentence (assumed to be under the key "input")
            for entry in data:
                sentence = entry.get("input", "")
                if sentence:
                    analysis = analyze_sentence(sentence)
                    analysis_results.append(analysis)
                else:
                    print("No input found in entry:", entry)

            # Save the analysis results in JSONL format.
            save_analysis_results(analysis_results, output_file)
            # print(f"Analysis results saved to {args.output_file}")