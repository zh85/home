"""Extract LLaMA-3-8B last-layer hidden states for all mp_20 crystals.

Usage:
    python exp_llm_cond/extract_llama_features.py \
        --lora_path exp/test_run \
        --dataset mp_20
"""
import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pymatgen.core.structure import Structure

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MAX_LENGTH = 2048

BASE_PROMPT = (
    "Below is a description of a bulk material. "
    "Generate a description of the lengths and angles of the lattice vectors "
    "and then the element type and coordinates for each atom within the lattice:\n"
)


def get_crystal_string_deterministic(cif_str):
    """Convert CIF to crystal text without random translation."""
    structure = Structure.from_str(cif_str, fmt="cif")
    lengths = structure.lattice.parameters[:3]
    angles = structure.lattice.parameters[3:]
    atom_ids = structure.species
    frac_coords = structure.frac_coords

    crystal_str = \
        " ".join(["{0:.1f}".format(x) for x in lengths]) + "\n" + \
        " ".join([str(int(x)) for x in angles]) + "\n" + \
        "\n".join([
            str(t) + "\n" + " ".join(["{0:.2f}".format(x) for x in c])
            for t, c in zip(atom_ids, frac_coords)
        ])
    return crystal_str


def load_llama_model(args):
    """Load LLaMA-3-8B with merged LoRA adapter."""
    print(f"Loading base model from: {args.llama_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.llama_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.llama_model_path,
        model_max_length=MAX_LENGTH,
        padding_side="left",
        use_fast=True,
    )
    # Add special tokens to match fine-tuning setup
    special_tokens = {}
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = "[PAD]"
    if tokenizer.eos_token is None:
        special_tokens["eos_token"] = "</s>"
    if tokenizer.bos_token is None:
        special_tokens["bos_token"] = "<s>"
    if tokenizer.unk_token is None:
        special_tokens["unk_token"] = "<unk>"

    if special_tokens:
        num_new = tokenizer.add_special_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))
        if num_new > 0:
            input_embeds = model.get_input_embeddings().weight.data
            output_embeds = model.get_output_embeddings().weight.data
            input_avg = input_embeds[:-num_new].mean(dim=0, keepdim=True)
            output_avg = output_embeds[:-num_new].mean(dim=0, keepdim=True)
            input_embeds[-num_new:] = input_avg
            output_embeds[-num_new:] = output_avg

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading LoRA adapter from: {args.lora_path}")
    model = PeftModel.from_pretrained(model, args.lora_path)
    model = model.merge_and_unload()
    model.eval()

    return model, tokenizer


def extract_features_for_split(csv_path, model, tokenizer, device,
                               batch_size=16):
    """Extract mean-pooled last-layer hidden states for a CSV split."""
    df = pd.read_csv(csv_path)

    # Deduplicate by material_id to avoid redundant computation
    unique_df = df.drop_duplicates(subset=['material_id'])
    print(f"Total rows: {len(df)}, unique materials: {len(unique_df)}")

    feat_dict = {}
    inputs_batch = []
    mp_ids_batch = []

    for _, row in tqdm(unique_df.iterrows(), total=len(unique_df),
                       desc=f"Processing {os.path.basename(csv_path)}"):
        cif_str = row['cif']
        mp_id = row['material_id']

        crystal_text = get_crystal_string_deterministic(cif_str)
        full_text = BASE_PROMPT + crystal_text + tokenizer.eos_token

        inputs_batch.append(full_text)
        mp_ids_batch.append(mp_id)

        if len(inputs_batch) >= batch_size:
            _process_batch(model, tokenizer, device,
                           inputs_batch, mp_ids_batch, feat_dict)
            inputs_batch, mp_ids_batch = [], []

    if inputs_batch:
        _process_batch(model, tokenizer, device,
                       inputs_batch, mp_ids_batch, feat_dict)

    return feat_dict


def _process_batch(model, tokenizer, device, texts, mp_ids, feat_dict):
    tokens = tokenizer(
        texts, return_tensors="pt", padding=True,
        truncation=True, max_length=MAX_LENGTH,
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]  # [B, seq_len, 4096]

        # Mean pool over non-padding tokens
        mask = tokens['attention_mask'].unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)

    pooled = pooled.cpu().to(torch.float16)

    for mp_id, feat in zip(mp_ids, pooled):
        feat_dict[mp_id] = feat


def compute_scaler(feat_path, output_path):
    """Compute mean and std from training features."""
    feat_dict = torch.load(feat_path, map_location='cpu')
    features = torch.stack(list(feat_dict.values()))  # [N, 4096]
    mean = features.mean(dim=0, keepdim=True)  # [1, 4096]
    std = features.std(dim=0, keepdim=True) + 1e-5  # [1, 4096]
    scaler = {'mean': mean.half(), 'std': std.half()}
    torch.save(scaler, output_path)


def normalize_features(feat_path, scaler_path, output_path):
    """Normalize features using saved scaler."""
    feat_dict = torch.load(feat_path, map_location='cpu')
    scaler = torch.load(scaler_path, map_location='cpu')
    mean = scaler['mean'].squeeze(0)
    std = scaler['std'].squeeze(0)

    normalized = {k: ((v.float() - mean) / std).half()
                  for k, v in feat_dict.items()}
    torch.save(normalized, output_path)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_llama_model(args)

    data_base = f"data/{args.dataset}"
    out_base = f"{data_base}/llm_feat"

    for split in ['train', 'val', 'test']:
        csv_path = f"{data_base}/{split}.csv"
        out_path = f"{out_base}_{split}.pt"

        if not os.path.exists(csv_path):
            print(f"Skipping {csv_path} (not found)")
            continue

        print(f"\n{'='*60}")
        print(f"Extracting features for {split} split...")
        feat_dict = extract_features_for_split(
            csv_path, model, tokenizer, device,
            batch_size=args.batch_size)
        torch.save(feat_dict, out_path)

        del feat_dict
        torch.cuda.empty_cache()

    # Compute scaler from training set
    train_feat_path = f"{out_base}_train.pt"
    scaler_path = f"{out_base}_scaler.pt"
    print(f"\nComputing scaler from {train_feat_path}...")
    compute_scaler(train_feat_path, scaler_path)

    # Normalize all splits
    for split in ['train', 'val', 'test']:
        feat_path = f"{out_base}_{split}.pt"
        if os.path.exists(feat_path):
            print(f"Normalizing {split}...")
            normalize_features(
                feat_path, scaler_path,
                f"{out_base}_{split}_norm.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama_model_path", type=str,
                        default="/zhdd/home/hengzhang/models/Meta-Llama-3-8B/LLM-Research/Meta-Llama-3-8B")
    parser.add_argument("--lora_path", type=str,
                        default="exp/test_run")
    parser.add_argument("--dataset", type=str, default="mp_20")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    main(args)
