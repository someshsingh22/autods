#!/usr/bin/env bash

version=1
dataset="blade"

# The path to the directory which contains the mcts_nodes.json files that need to be deduplicated
mcts_data_dir="./Experiments/data/mcts"

# Logs detailed outputs and error messages
log_dir="./results/log"
# Directory for saving structured hypotheses - context, variable, relationships
structured_data_dir="./results/structured_hypothesis"
# Directory for saving text embeddings for the hypotheses
feat_dir="./results/embeddings/openai"
# Directory for saving dendograms of the HAC
img_out_dir="./results/dendrograms"
# Directory for saving deduplicated results
out_dir="./results/dedup"

save_clustering_output=True
save_visualization=False
seed=100
n_decision=30
llm_decision_threshold=0.7

# create dirs once
mkdir -p "$out_dir" "$log_dir" "$structured_data_dir" "$feat_dir" "$img_out_dir"

for node_file in "$mcts_data_dir"/*.json; do
  [ -e "$node_file" ] || continue

  filename=$(basename "$node_file")
  data_id="${filename%_mcts_nodes.json}"

  echo "→ processing dataset: $data_id"

  python3 update_clustering.py \
    --data_id "$data_id" \
    --node_log_path "$node_file" \
    --feat_dir "$feat_dir" \
    --structured_data_dir "$structured_data_dir" \
    --img_out_dir "$img_out_dir" \
    --save_clustering_output "$save_clustering_output" \
    --save_visualization "$save_visualization" \
    --out_dir "$out_dir" \
    --log_dir "$log_dir" \
    --seed "$seed" \
    --n_decision "$n_decision" \
    --llm_decision_threshold "$llm_decision_threshold" \
    --version "$version" \
    --dataset "$dataset"
done
