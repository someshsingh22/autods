import os
import shutil

src_root = "proxy_data_2/interesting"
dst_dir = "proxy_data_2/interesting_mcts_nodes"

os.makedirs(dst_dir, exist_ok=True)

for root, _, files in os.walk(src_root):
    if 'mcts_nodes.json' in files:
        src_path = os.path.join(root, 'mcts_nodes.json')
        dir_name = os.path.basename(root)
        dst_name = f"{dir_name}_mcts_nodes.json"
        shutil.copy2(src_path, os.path.join(dst_dir, dst_name))
