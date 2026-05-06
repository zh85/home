"""Phase 1: LLM generation + feature extraction → save to disk.
Run this once, then run phase 2 (diffusion sampling) separately.
"""
import sys, os, time, argparse, torch, numpy as np
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import modeling_utils
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice

from config import config
from llm_finetune import MAX_LENGTH
from data_utils import process_one
from exp_llm_cond.extract_llama_features import BASE_PROMPT, get_crystal_string_deterministic

if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or \
   modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]


def parse_generated_text(gen_text):
    lines = [line for line in gen_text.split("\n") if line.strip()]
    lengths_val = list(map(float, lines[0].split()))
    angles_val = list(map(float, lines[1].split()))
    species = lines[2::2]
    coords = [list(map(float, c.split())) for c in lines[3::2]]
    structure = Structure(
        lattice=Lattice.from_parameters(*(lengths_val + angles_val)),
        species=species, coords=coords, coords_are_cartesian=False)
    return structure.to(fmt="cif")


def resize_tokenizer_and_embeddings(tokenizer, model, special_tokens: dict):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeds = model.get_input_embeddings().weight.data
        output_embeds = model.get_output_embeddings().weight.data
        input_avg = input_embeds[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_avg = output_embeds[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeds[-num_new_tokens:] = input_avg
        output_embeds[-num_new_tokens:] = output_avg


def encode_llm_feature(model, tokenizer, crystal_text, device):
    full_text = BASE_PROMPT + crystal_text + tokenizer.eos_token
    tokens = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        mask = tokens['attention_mask'].unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
    return pooled.float().cpu().squeeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="exp/test_run")
    parser.add_argument("--llama_model_path", type=str,
                        default="/zhdd/home/hengzhang/models/Meta-Llama-3-8B/LLM-Research/Meta-Llama-3-8B")
    parser.add_argument("--dataset", type=str, default="mp_20")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--use_llm_cond", action="store_true", default=True)
    parser.add_argument("--out-prefix", type=str, default="results/llama3_sample_crossattn")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load LLM
    print(f"Loading base model from: {args.llama_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.llama_model_path, model_max_length=MAX_LENGTH,
                                              padding_side="left", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.llama_model_path, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
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

    prompts = [BASE_PROMPT] * args.num_samples
    collected_data = []
    collected_features = []
    progress = tqdm(total=args.num_samples, desc="Generating LLM samples")
    idx = 0

    while idx < args.num_samples:
        batch_prompts = prompts[idx:idx + args.batch_size]
        tokenized = tokenizer(batch_prompts, return_tensors="pt")
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        try:
            output_ids = model.generate(**tokenized, do_sample=True, max_new_tokens=500,
                                        temperature=args.temperature, top_p=args.top_p,
                                        pad_token_id=tokenizer.eos_token_id)
            decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True,
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
                if args.use_llm_cond:
                    crystal_text = get_crystal_string_deterministic(cif_str)
                    feat = encode_llm_feature(model, tokenizer, crystal_text, device)
                    collected_features.append(feat)

        idx += len(collected_data) - idx
        progress.update(len(collected_data) - progress.n)

    progress.close()
    print(f"LLM generated {len(collected_data)} valid samples")

    # Save intermediate results to disk
    out_prefix = f"{args.out_prefix}_{args.dataset}_{args.num_samples}"
    torch.save(collected_data, f"{out_prefix}_data.pt")
    torch.save(collected_features, f"{out_prefix}_features.pt")
    print(f"Saved {len(collected_data)} structures to {out_prefix}_data.pt")
    print(f"Saved {len(collected_features)} features to {out_prefix}_features.pt")

    del model, tokenizer
    torch.cuda.empty_cache()
    print("LLM model freed. Intermediate data saved. Run phase 2 next.")


if __name__ == "__main__":
    main()
