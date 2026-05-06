# =========================
# Standard Library Imports
# =========================
import os
import time
import argparse
import random

# =========================
# Third-Party Imports
# =========================
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import LlamaForCausalLM, LlamaTokenizer, modeling_utils
from peft import PeftModel
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, AutoTokenizer, TrainingArguments, AutoModelForCausalLM

from torch.utils.data import Dataset
from torch_geometric.data import Data, DataLoader

from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.io.cif import CifWriter

# =========================
# Project-Specific Imports
# =========================
from config import config
from llm_finetune import get_crystal_string, MAX_LENGTH
from templating import make_swap_table
from data_utils import process_one
from models_ddpm.diffusion import CSPDiffusion

# =========================
# Token Constants
# =========================
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

# Fix for older transformers versions
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]


# ======================================================
# Utility Functions
# ======================================================
def lattices_to_params_shape(lattices: torch.Tensor):
    """
    Convert lattice matrices to lengths and angles.

    Args:
        lattices (Tensor): Shape (..., 3, 3)

    Returns:
        lengths (Tensor): (..., 3)
        angles  (Tensor): (..., 3) in degrees
    """
    lengths = torch.sqrt(torch.sum(lattices ** 2, dim=-1))
    angles = torch.zeros_like(lengths)

    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        cos_angle = torch.sum(
            lattices[..., j, :] * lattices[..., k, :], dim=-1
        ) / (lengths[..., j] * lengths[..., k])
        angles[..., i] = torch.clamp(cos_angle, -1.0, 1.0)

    angles = torch.arccos(angles) * 180.0 / np.pi
    return lengths, angles


def parse_generated_text(gen_text: str) -> str:
    """
    Parse LLM-generated crystal text into a CIF string.

    Args:
        gen_text (str): Raw generated text

    Returns:
        cif_str (str): CIF-formatted structure
    """
    lines = [line for line in gen_text.split("\n") if line.strip()]

    lengths = list(map(float, lines[0].split()))
    angles = list(map(float, lines[1].split()))
    species = lines[2::2]
    coords = [list(map(float, c.split())) for c in lines[3::2]]

    structure = Structure(
        lattice=Lattice.from_parameters(*(lengths + angles)),
        species=species,
        coords=coords,
        coords_are_cartesian=False,
    )

    return structure.to(fmt="cif")


# ======================================================
# Model & Tokenizer Preparation
# ======================================================
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
    """Load LLaMA-3 base model from local path, apply LoRA adapter, and merge."""
    base_model_path = args.llama_model_path
    print(f"Loading base model from: {base_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        model_max_length=MAX_LENGTH,
        padding_side="left",
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
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
    model = PeftModel.from_pretrained(
        model,
        args.model_path,
    )

    model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


# ======================================================
# LLM Sampling (Unconditional)
# ======================================================
def unconditional_sample(args, model, tokenizer, prompts):
    """
    Generate crystal structures from the LLM.

    Returns:
        frac_coords, atom_types, lengths, angles, num_atoms, data_dicts
    """
    tokenized = tokenizer(prompts, return_tensors="pt")
    tokenized = {k: v.cuda() for k, v in tokenized.items()}

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
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    except Exception:
        return None, None, None, None, None, None

    frac_coords_list, atom_types_list = [], []
    lengths_list, angles_list, num_atoms_list = [], [], []
    data_dicts = []

    for gen_text, prompt in zip(decoded, prompts):
        generated_part = gen_text.replace(prompt, "")

        try:
            cif_str = parse_generated_text(generated_part)
            (
                frac_coords,
                atom_types,
                lengths,
                angles,
                num_atoms,
                _,
                _,
                data_dict,
            ) = process_one(cif_str, True, False, "crystalnn", False, 0.01)
        except Exception:
            continue

        valid_atoms = ((atom_types >= 0) & (atom_types < 100)).all().item()
        if valid_atoms:
            frac_coords_list.append(torch.tensor(frac_coords))
            atom_types_list.append(torch.tensor(atom_types))
            lengths_list.append(torch.tensor(lengths).view(1, 3))
            angles_list.append(torch.tensor(angles).view(1, 3))
            num_atoms_list.append(torch.tensor([num_atoms]))
            data_dicts.append(data_dict)

    return frac_coords_list, atom_types_list, lengths_list, angles_list, num_atoms_list, data_dicts


# ======================================================
# Dataset Definition
# ======================================================
class SampleDataset(Dataset):
    """Dataset wrapper for diffusion model sampling."""

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


# ======================================================
# Main Execution
# ======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="exp/test_run",
                        help="Path to the LoRA adapter checkpoint directory")
    parser.add_argument("--llama_model_path", type=str,
                        default="/zhdd/home/hengzhang/models/Meta-Llama-3-8B/LLM-Research/Meta-Llama-3-8B",
                        help="Path to the base LLaMA-3-8B model")
    parser.add_argument("--dataset", type=str, default="mp_20")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--chkpt_name", type=str,
                        default="out/mp_20/03052026/211726/model_final.pt",
                        help="Path to the trained diffusion model checkpoint")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--diff_steps", type=int, default=500)
    parser.add_argument("--num_evals", type=int, default=1)
    parser.add_argument("--run-type", type=str, default="train")
    parser.add_argument("--out-prefix", type=str, default="results/llama3_sample")
    args = parser.parse_args()

    device = config.device if torch.cuda.is_available() else "cpu"

    print(f"Base model: {args.llama_model_path}")
    print(f"LoRA adapter: {args.model_path}")
    print(f"Diffusion checkpoint: {args.chkpt_name}")
    print(f"Dataset: {args.dataset}")

    model, tokenizer = prepare_model_and_tokenizer(args)

    # -------- Prompt Construction --------
    base_prompt = (
        "Below is a description of a bulk material. "
        "Generate a description of the lengths and angles of the lattice vectors "
        "and then the element type and coordinates for each atom within the lattice:\n"
    )
    prompts = [base_prompt] * args.num_samples

    # -------- LLM Sampling --------
    frac_coords_llm, atom_types_llm, lengths_llm, angles_llm, num_atoms_llm = [], [], [], [], []
    collected_data = []
    progress = tqdm(total=args.num_samples, desc="Generating LLM samples")

    idx = 0
    while idx < args.num_samples:
        end_idx = min(idx + args.batch_size, args.num_samples)
        batch_prompts = prompts[idx:end_idx]
        frac_coords_list, atom_types_list, lengths_list, angles_list, num_atoms_list, data_dicts = \
            unconditional_sample(args, model, tokenizer, batch_prompts)

        if frac_coords_list is None:
            continue

        frac_coords_llm.extend(frac_coords_list)
        atom_types_llm.extend(atom_types_list)
        lengths_llm.extend(lengths_list)
        angles_llm.extend(angles_list)
        num_atoms_llm.extend(num_atoms_list)
        collected_data.extend(data_dicts)

        idx += len(data_dicts)
        progress.update(len(data_dicts))

    print(f"LLM generated {len(collected_data)} valid samples")

    # -------- Diffusion Model Sampling --------
    diffusion_model = CSPDiffusion(args.timesteps, args.run_type).to(device)
    checkpoint = torch.load(args.chkpt_name, map_location=device)
    diffusion_model.load_state_dict(checkpoint["model"])

    dataset = SampleDataset(collected_data)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    frac_coords_all, num_atoms_all = [], []
    atom_types_all, lattices_all = [], []

    start_time = time.time()

    for i, batch in enumerate(dataloader):
        print(f"Sampling batch {i}/{len(dataloader)} ")
        batch = batch.to(device)

        batch.lengths = batch.lengths.to(torch.float32)
        batch.angles = batch.angles.to(torch.float32)
        batch.frac_coords = batch.frac_coords.to(torch.float32)
        if hasattr(batch, 'atom_types'):
            if batch.atom_types.dtype == torch.float64:
                batch.atom_types = batch.atom_types.to(torch.int64)

        batch_frac, batch_num, batch_atom, batch_lat = [], [], [], []

        for _ in range(args.num_evals):
            outputs, _ = diffusion_model.sample(batch, diff_steps=args.diff_steps)
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
