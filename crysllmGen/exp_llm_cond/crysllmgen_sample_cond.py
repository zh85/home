"""Crystal generation with LLM conditioning during diffusion refinement.

Usage:
    python exp_llm_cond/crysllmgen_sample_cond.py \
        --model_path exp/test_run \
        --chkpt_name out/mp_20/.../model_final.pt \
        --dataset mp_20
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
import torch
import numpy as np
from tqdm import tqdm

from transformers import modeling_utils
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from torch.utils.data import Dataset
from torch_geometric.data import Data, DataLoader

from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice

from config import config
from llm_finetune import MAX_LENGTH
from data_utils import process_one
from exp_llm_cond.diffusion_cond import CondCSPDiffusion
from exp_llm_cond.extract_llama_features import (
    BASE_PROMPT, get_crystal_string_deterministic,
)

if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or \
   modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = [
        "tp", "none", "colwise", "rowwise"]


def lattices_to_params_shape(lattices):
    lengths = torch.sqrt(torch.sum(lattices ** 2, dim=-1))
    angles = torch.zeros_like(lengths)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        cos_angle = torch.sum(
            lattices[..., j, :] * lattices[..., k, :], dim=-1) / \
            (lengths[..., j] * lengths[..., k])
        angles[..., i] = torch.clamp(cos_angle, -1.0, 1.0)
    angles = torch.arccos(angles) * 180.0 / np.pi
    return lengths, angles


def parse_generated_text(gen_text):
    lines = [line for line in gen_text.split("\n") if line.strip()]
    lengths_val = list(map(float, lines[0].split()))
    angles_val = list(map(float, lines[1].split()))
    species = lines[2::2]
    coords = [list(map(float, c.split())) for c in lines[3::2]]

    structure = Structure(
        lattice=Lattice.from_parameters(*(lengths_val + angles_val)),
        species=species,
        coords=coords,
        coords_are_cartesian=False,
    )
    return structure.to(fmt="cif")


def resize_tokenizer_and_embeddings(tokenizer, model, special_tokens: dict):
    """Resize tokenizer and initialize new embeddings if special tokens are added."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeds = model.get_input_embeddings().weight.data
        output_embeds = model.get_output_embeddings().weight.data
        input_avg = input_embeds[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_avg = output_embeds[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeds[-num_new_tokens:] = input_avg
        output_embeds[-num_new_tokens:] = output_avg


def prepare_model_and_tokenizer(args):
    print(f"Loading base model from: {args.llama_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.llama_model_path,
        model_max_length=MAX_LENGTH,
        padding_side="left",
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.llama_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Add special tokens used during LoRA adapter training
    special_tokens = {}
    if tokenizer.convert_tokens_to_ids("[PAD]") is None:
        special_tokens["pad_token"] = "[PAD]"
    if tokenizer.unk_token is None:
        special_tokens["unk_token"] = "<unk>"
    if special_tokens:
        resize_tokenizer_and_embeddings(tokenizer, model, special_tokens)

    print(f"Loading LoRA adapter from: {args.model_path}")
    model = PeftModel.from_pretrained(model, args.model_path)
    model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def encode_llm_feature(model, tokenizer, crystal_text, device):
    """Extract a single LLM feature vector for diffusion conditioning."""
    full_text = BASE_PROMPT + crystal_text + tokenizer.eos_token
    tokens = tokenizer(
        full_text, return_tensors="pt", truncation=True,
        max_length=MAX_LENGTH)
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        mask = tokens['attention_mask'].unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)

    return pooled.float().squeeze(0)


class SampleDataset(Dataset):
    def __init__(self, structures):
        self.structures = structures

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        s = self.structures[idx]
        return Data(
            num_atoms=torch.LongTensor([s["n_atom"]]),
            num_nodes=s["n_atom"],
            num_bonds=s["edge_indices"].shape[0],
            lengths=s["length"],
            angles=s["angle"],
            frac_coords=torch.Tensor(s["x_coord"]),
            atom_types=torch.LongTensor(s["a_type"]),
            edge_index=torch.LongTensor(s["edge_indices"].T).contiguous(),
            to_jimages=torch.LongTensor(s["to_jimages"]),
        )


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load LLM
    model, tokenizer = prepare_model_and_tokenizer(args)

    # Load diffusion model
    print(f"Loading diffusion checkpoint from: {args.chkpt_name}")
    diffusion_model = CondCSPDiffusion(
        args.timesteps, args.run_type,
        use_llm_cond=args.use_llm_cond,
    ).to(device)
    checkpoint = torch.load(args.chkpt_name, map_location=device)
    diffusion_model.load_state_dict(checkpoint["model"])

    prompts = [BASE_PROMPT] * args.num_samples

    # LLM Sampling
    collected_data = []
    collected_features = []
    progress = tqdm(total=args.num_samples,
                    desc="Generating LLM samples")

    idx = 0
    while idx < args.num_samples:
        batch_prompts = prompts[idx:idx + args.batch_size]
        tokenized = tokenizer(batch_prompts, return_tensors="pt")
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        try:
            output_ids = model.generate(
                **tokenized,
                do_sample=True,
                max_new_tokens=500,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
            decoded = tokenizer.batch_decode(
                output_ids, skip_special_tokens=True,
                clean_up_tokenization_spaces=False)
        except Exception:
            continue

        for gen_text, prompt in zip(decoded, batch_prompts):
            generated_part = gen_text.replace(prompt, "")

            try:
                cif_str = parse_generated_text(generated_part)
                (frac_coords, atom_types, lengths, angles,
                 num_atoms, _, _, data_dict) = process_one(
                     cif_str, True, False, "crystalnn", False, 0.01)
            except Exception:
                continue

            valid_atoms = ((atom_types >= 0) & (atom_types < 100)).all().item()
            if valid_atoms:
                collected_data.append(data_dict)

                # Extract LLM feature for this crystal
                if args.use_llm_cond:
                    crystal_text = get_crystal_string_deterministic(cif_str)
                    feat = encode_llm_feature(
                        model, tokenizer, crystal_text, device)
                    collected_features.append(feat)

        idx += len(collected_data) - idx
        progress.update(len(collected_data) - progress.n)

    print(f"LLM generated {len(collected_data)} valid samples")

    # Diffusion Model Sampling
    dataset = SampleDataset(collected_data)
    dataloader = DataLoader(
        dataset, batch_size=min(args.batch_size, len(collected_data)), shuffle=False)

    frac_coords_all, num_atoms_all = [], []
    atom_types_all, lattices_all = [], []

    start_time = time.time()

    for i, batch in enumerate(dataloader):
        print(f"Sampling batch {i}/{len(dataloader)}")
        batch = batch.to(device)

        batch.lengths = batch.lengths.to(torch.float32)
        batch.angles = batch.angles.to(torch.float32)
        batch.frac_coords = batch.frac_coords.to(torch.float32)
        if hasattr(batch, 'atom_types') and \
           batch.atom_types.dtype == torch.float64:
            batch.atom_types = batch.atom_types.to(torch.int64)

        # Get LLM features for this batch
        batch_size = batch.num_graphs
        llm_feat = None
        if args.use_llm_cond and collected_features:
            start_idx = i * dataloader.batch_size
            end_idx = min(start_idx + batch_size, len(collected_features))
            batch_feats = collected_features[start_idx:end_idx]
            if batch_feats:
                llm_feat = torch.stack(batch_feats).to(device)

        batch_frac, batch_num, batch_atom, batch_lat = [], [], [], []

        for _ in range(args.num_evals):
            outputs, _ = diffusion_model.sample(
                batch, diff_steps=args.diff_steps, llm_feat=llm_feat)
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
    torch.save(
        {
            "frac_coords": frac_coords_all,
            "num_atoms": num_atoms_all,
            "atom_types": atom_types_all,
            "lengths": lengths,
            "angles": angles,
            "time": time.time() - start_time,
        },
        output_file,
    )
    print("Saved results to:", output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="exp/test_run",
                        help="LoRA adapter checkpoint directory")
    parser.add_argument("--llama_model_path", type=str,
                        default="/zhdd/home/hengzhang/models/Meta-Llama-3-8B/LLM-Research/Meta-Llama-3-8B")
    parser.add_argument("--dataset", type=str, default="mp_20")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--chkpt_name", type=str,
                        default="out/mp_20/03052026/211726/model_final.pt")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--diff_steps", type=int, default=500)
    parser.add_argument("--num_evals", type=int, default=1)
    parser.add_argument("--run-type", type=str, default="train")
    parser.add_argument("--out-prefix", type=str,
                        default="results/llama3_sample_cond")
    parser.add_argument("--use_llm_cond", action="store_true", default=True,
                        help="Use LLM features for diffusion conditioning")
    args = parser.parse_args()

    main(args)
