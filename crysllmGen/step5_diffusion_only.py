"""Phase 2: Diffusion sampling from saved LLM data + features.
Run after step5_llm_gen_only.py completes.
"""
import sys, os, time, argparse, torch, numpy as np
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torch.utils.data import Dataset
from torch_geometric.data import Data, DataLoader

from exp_llm_crossattn.diffusion_crossattn import CrossAttnCSPDiffusion


def lattices_to_params_shape(lattices):
    lengths = torch.sqrt(torch.sum(lattices ** 2, dim=-1))
    angles = torch.zeros_like(lengths)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        cos_angle = torch.sum(lattices[..., j, :] * lattices[..., k, :], dim=-1) / \
            (lengths[..., j] * lengths[..., k])
        angles[..., i] = torch.clamp(cos_angle, -1.0, 1.0)
    angles = torch.arccos(angles) * 180.0 / np.pi
    return lengths, angles


class SampleDataset(Dataset):
    def __init__(self, structures):
        self.structures = structures
    def __len__(self):
        return len(self.structures)
    def __getitem__(self, idx):
        s = self.structures[idx]
        return Data(
            num_atoms=torch.LongTensor([s["n_atom"]]), num_nodes=s["n_atom"],
            num_bonds=s["edge_indices"].shape[0], lengths=s["length"],
            angles=s["angle"], frac_coords=torch.Tensor(s["x_coord"]),
            atom_types=torch.LongTensor(s["a_type"]),
            edge_index=torch.LongTensor(s["edge_indices"].T).contiguous(),
            to_jimages=torch.LongTensor(s["to_jimages"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chkpt_name", type=str,
                        default="out/mp_20/04052026/120614/model_199.pt")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--diff_steps", type=int, default=500)
    parser.add_argument("--num_evals", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--run-type", type=str, default="train")
    parser.add_argument("--out-prefix", type=str, default="results/llama3_sample_crossattn")
    parser.add_argument("--dataset", type=str, default="mp_20")
    parser.add_argument("--num_samples", type=int, default=10000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load saved intermediate data
    data_file = f"{args.out_prefix}_{args.dataset}_{args.num_samples}_data.pt"
    feat_file = f"{args.out_prefix}_{args.dataset}_{args.num_samples}_features.pt"
    print(f"Loading structures from: {data_file}")
    collected_data = torch.load(data_file, map_location='cpu', weights_only=False)
    print(f"Loading features from: {feat_file}")
    collected_features = torch.load(feat_file, map_location='cpu', weights_only=False)
    print(f"Loaded {len(collected_data)} structures, {len(collected_features)} features")

    # Load diffusion model
    print(f"Loading diffusion checkpoint from: {args.chkpt_name}")
    diffusion_model = CrossAttnCSPDiffusion(args.timesteps, args.run_type,
                                            use_llm_cond=True).to(device)
    checkpoint = torch.load(args.chkpt_name, map_location=device)
    diffusion_model.load_state_dict(checkpoint["model"])
    diffusion_model.eval()

    dataset = SampleDataset(collected_data[:args.num_samples])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    frac_coords_all, num_atoms_all = [], []
    atom_types_all, lattices_all = [], []
    start_time = time.time()

    for i, batch in enumerate(dataloader):
        print(f"Sampling batch {i}/{len(dataloader)}")
        batch = batch.to(device)
        batch.lengths = batch.lengths.to(torch.float32)
        batch.angles = batch.angles.to(torch.float32)
        batch.frac_coords = batch.frac_coords.to(torch.float32)
        if hasattr(batch, 'atom_types') and batch.atom_types.dtype == torch.float64:
            batch.atom_types = batch.atom_types.to(torch.int64)

        batch_size = batch.num_graphs
        llm_feat = None
        if collected_features:
            start_idx = i * dataloader.batch_size
            end_idx = min(start_idx + batch_size, len(collected_features))
            batch_feats = collected_features[start_idx:end_idx]
            if batch_feats:
                llm_feat = torch.stack(batch_feats).to(device)

        batch_frac, batch_num, batch_atom, batch_lat = [], [], [], []
        for _ in range(args.num_evals):
            outputs, _ = diffusion_model.sample(batch, diff_steps=args.diff_steps, llm_feat=llm_feat)
            batch_frac.append(outputs["frac_coords"].cpu())
            batch_num.append(outputs["num_atoms"].cpu())
            batch_atom.append(outputs["atom_types"].cpu())
            batch_lat.append(outputs["lattices"].cpu())

        frac_coords_all.append(torch.stack(batch_frac))
        num_atoms_all.append(torch.stack(batch_num))
        atom_types_all.append(torch.stack(batch_atom))
        lattices_all.append(torch.stack(batch_lat))

    frac_coords_all = torch.cat(frac_coords_all, dim=1)
    num_atoms_all = torch.cat(num_atoms_all, dim=1)
    atom_types_all = torch.cat(atom_types_all, dim=1)
    lattices_all = torch.cat(lattices_all, dim=1)
    lengths, angles = lattices_to_params_shape(lattices_all)

    print("Total generation time:", time.time() - start_time)

    output_file = f"{args.out_prefix}_{args.dataset}_{args.num_samples}.pt"
    torch.save({
        "frac_coords": frac_coords_all, "num_atoms": num_atoms_all,
        "atom_types": atom_types_all, "lengths": lengths,
        "angles": angles, "time": time.time() - start_time,
    }, output_file)
    print("Saved results to:", output_file)


if __name__ == "__main__":
    main()
