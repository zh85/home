"""Convert LLM-generated data (data_dict format) to compute_metrics.py format,
then run metrics to evaluate LLM-only quality (before diffusion refinement).
"""
import sys, os, torch, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str,
                    default='results/llama3_sample_crossattn_mp_20_10000_data.pt')
parser.add_argument('--out_file', type=str,
                    default='results/llama3_sample_llmonly_mp_20_10000.pt')
parser.add_argument('--num_samples', type=int, default=10000)
args = parser.parse_args()

print(f"Loading LLM-generated data from: {args.data_file}")
collected_data = torch.load(args.data_file, map_location='cpu', weights_only=False)
print(f"Loaded {len(collected_data)} structures, using first {args.num_samples}")

structures = collected_data[:args.num_samples]

# Convert from data_dict to compute_metrics format
num_atoms_list = []
atom_types_list = []
frac_coords_list = []
lengths_list = []
angles_list = []

for s in structures:
    n_atom = int(s['n_atom']) if isinstance(s['n_atom'], (int, float)) else int(s['n_atom'][0])
    num_atoms_list.append(torch.LongTensor([n_atom]))
    atom_types_list.append(torch.LongTensor(s['a_type']).squeeze())
    frac_coords_list.append(torch.Tensor(s['x_coord']).squeeze())
    lengths_list.append(torch.Tensor(s['length']).squeeze())
    angles_list.append(torch.Tensor(s['angle']).squeeze())

# Pad/stack variable-length tensors
num_atoms_all = torch.cat(num_atoms_list)
atom_types_all = torch.cat(atom_types_list)
frac_coords_all = torch.cat(frac_coords_list)
lengths_all = torch.stack(lengths_list).reshape(-1, 3)
angles_all = torch.stack(angles_list).reshape(-1, 3)

print(f"Saving to: {args.out_file}")
torch.save({
    'frac_coords': frac_coords_all,
    'atom_types': atom_types_all,
    'num_atoms': num_atoms_all,
    'lengths': lengths_all,
    'angles': angles_all,
}, args.out_file)
print("Done. Now run: python compute_metrics.py --root_path ... --eval_model_name mp20 --tasks gen --gt_file data/mp_20/test.csv")
