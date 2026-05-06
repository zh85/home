"""Diffusion model training with optional LLM conditioning.

Usage:
    # Standard training (no LLM conditioning)
    python exp_llm_cond/diff_train_cond.py --dataset mp_20 --epochs 500

    # With LLM conditioning
    python exp_llm_cond/diff_train_cond.py --dataset mp_20 --epochs 500 --use_llm_cond
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
from tqdm import tqdm
from time import time

from config import config
from torch_geometric.data import DataLoader
from exp_llm_cond.dataset_cond import CondMaterialDataset
from exp_llm_cond.diffusion_cond import CondCSPDiffusion
from utils import configure_save_path, save_model, delete_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(dataloader, model, use_llm_cond=False):
    iters = len(dataloader)
    diff_losses = np.empty(iters)
    coord_losses = np.empty(iters)
    lattice_losses = np.empty(iters)

    for i, batch in enumerate(dataloader):
        batch = batch.to(device)

        # Extract LLM features if available
        # PyG concatenates per-node tensors; reshape to [B, feat_dim]
        llm_feat = None
        if use_llm_cond and hasattr(batch, 'llm_feat'):
            llm_feat = batch.llm_feat.to(device).view(batch.num_graphs, -1)

        loss, loss_lattice, loss_coord = model(batch, llm_feat=llm_feat)
        model.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.)
        model.optim.step()

        diff_losses[i] = loss.item()
        coord_losses[i] = loss_coord.item()
        lattice_losses[i] = loss_lattice.item()

    return diff_losses.mean(), coord_losses.mean(), lattice_losses.mean()


def validate(dataloader, model, use_llm_cond=False):
    iters = len(dataloader)
    diff_losses = np.empty(iters)
    coord_losses = np.empty(iters)
    lattice_losses = np.empty(iters)

    for i, batch in enumerate(dataloader):
        batch = batch.to(device)

        llm_feat = None
        if use_llm_cond and hasattr(batch, 'llm_feat'):
            llm_feat = batch.llm_feat.to(device).view(batch.num_graphs, -1)

        loss, loss_lattice, loss_coord = model(batch, llm_feat=llm_feat)
        diff_losses[i] = loss.item()
        coord_losses[i] = loss_coord.item()
        lattice_losses[i] = loss_lattice.item()

    return diff_losses.mean(), coord_losses.mean(), lattice_losses.mean()


def train(train_dataloader, val_dataloader, test_dataloader,
          model, epochs, dataset, use_llm_cond=False):
    basedir = configure_save_path(dataset)
    path = os.path.join(basedir, config.file_name_model)
    figpath = os.path.join(basedir, config.file_name_plot)
    min_loss = 1e8
    best_epoch = None

    if not os.path.exists(figpath):
        os.makedirs(figpath)
    out = open(figpath + "/out.txt", "w")

    for epoch in tqdm(range(epochs)):
        t0 = time()
        loss, coord_loss, lattice_loss = train_epoch(
            train_dataloader, model, use_llm_cond)
        val_loss, val_coord_loss, val_lattice_loss = validate(
            val_dataloader, model, use_llm_cond)
        model.scheduler.step(loss)

        if loss < min_loss:
            save_model(model, f"{path}_{epoch:03d}.pt")
            if best_epoch is not None:
                delete_model(f"{path}_{best_epoch:03d}.pt")
            min_loss = loss
            best_epoch = epoch
            best_str = "NEW BEST"
        else:
            best_str = f"best: {min_loss:.4f} at {best_epoch}"

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} "
                  f"Loss: {loss:.4f}, "
                  f"Coord Loss: {coord_loss:.4f}, "
                  f"Lattice Loss: {lattice_loss:.4f} "
                  f"[{time() - t0:.2f}s]", best_str)
            print(f"Validation Coord Loss: {val_coord_loss:.4f}, "
                  f"Lattice Loss: {val_lattice_loss:.4f}, "
                  f"[{time() - t0:.2f}s]\n")

            results = ('Epoch :' + str(epoch)
                       + ' Train Diff Loss : ' + str(round(loss, 4))
                       + ' Train Coord Loss : ' + str(round(coord_loss, 4))
                       + ' Train Lattice Loss : ' + str(round(lattice_loss, 4))
                       + ' Valid Coord Loss : ' + str(round(val_coord_loss, 4))
                       + ' Valid Lattice Loss : ' + str(round(val_lattice_loss, 4))
                       + ' Time : ' + str(int(time() - t0)))
            out.writelines(results)
            out.writelines("\n\n")

    test_lattice_loss, test_coord_loss, test_type_loss = validate(
        test_dataloader, model, use_llm_cond)
    print(f"Test Coord Loss: {test_coord_loss:.4f}, "
          f"Lattice Loss: {test_lattice_loss:.4f}, "
          f"[{time() - t0:.2f}s]")

    test_results = ('Test Coord Loss : ' + str(round(test_coord_loss, 4))
                    + ' Test Lattice Loss : ' + str(round(test_lattice_loss, 4)))
    out.writelines(test_results)
    out.writelines("\n")
    save_model(model, f"{path}_final.pt")
    print("All saved at ", path)
    return model, path


def main(args):
    print(torch.__version__)
    print(f"Use LLM cond: {args.use_llm_cond}")

    torch.manual_seed(config.SEED)

    data_base = f"data/{args.dataset}"

    # Setup LLM feature paths
    llm_feat_dir = args.llm_feat_dir or data_base
    if args.use_llm_cond:
        scaler_path = f"{llm_feat_dir}/llm_feat_scaler.pt"
    else:
        scaler_path = None

    dataset_kwargs = {}
    if args.use_llm_cond:
        dataset_kwargs = {
            'train': {
                'llm_feat_path': f"{llm_feat_dir}/llm_feat_train_norm.pt",
                'llm_scaler_path': scaler_path,
            },
            'val': {
                'llm_feat_path': f"{llm_feat_dir}/llm_feat_val_norm.pt",
                'llm_scaler_path': scaler_path,
            },
            'test': {
                'llm_feat_path': f"{llm_feat_dir}/llm_feat_test_norm.pt",
                'llm_scaler_path': scaler_path,
            },
        }

    train_dataset = CondMaterialDataset(
        os.path.join(data_base, f"{config.train_data}.csv"),
        **dataset_kwargs.get('train', {}))
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        pin_memory=True)

    val_dataset = CondMaterialDataset(
        os.path.join(data_base, f"{config.eval_data}.csv"),
        **dataset_kwargs.get('val', {}))
    val_dataloader = DataLoader(
        val_dataset, batch_size=32, shuffle=True, pin_memory=True)

    test_dataset = CondMaterialDataset(
        os.path.join(data_base, f"{config.test_data}.csv"),
        **dataset_kwargs.get('test', {}))
    test_dataloader = DataLoader(
        test_dataset, batch_size=32, shuffle=True, pin_memory=True)

    model = CondCSPDiffusion(
        args.timesteps, args.run_type,
        use_llm_cond=args.use_llm_cond,
    ).to(device)

    train(train_dataloader, val_dataloader, test_dataloader,
          model, args.epochs, args.dataset,
          use_llm_cond=args.use_llm_cond)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--dataset', required=True, type=str, default='perov_5')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--expt_date', type=str)
    parser.add_argument('--expt_time', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--run-type', type=str, default='train')
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--use_llm_cond', action='store_true', default=False,
                        help='Use pre-extracted LLM features as conditioning')
    parser.add_argument('--llm_feat_dir', type=str, default=None,
                        help='Directory containing llm_feat_*.pt files')
    args = parser.parse_args()
    main(args)
