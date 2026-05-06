"""CondMaterialDataset: MaterialDataset with pre-extracted LLM features."""
import torch
import numpy as np
from torch_geometric.data import Data

from models_ddpm.dataset import MaterialDataset


class CondMaterialDataset(MaterialDataset):
    """MaterialDataset subclass that loads and aligns LLM features."""

    def __init__(self, path, llm_feat_path=None, llm_scaler_path=None,
                 niggli=True, primitive=False, graph_method='crystalnn',
                 preprocess_workers=30):
        super().__init__(path)

        self.llm_feat = None
        if llm_feat_path is not None:
            feat_dict = torch.load(llm_feat_path, map_location='cpu')

            # Load scaler if provided
            scaler = None
            if llm_scaler_path is not None:
                scaler = torch.load(llm_scaler_path, map_location='cpu')

            # Align features with cached_data order
            feat_list = []
            for data_dict in self.cached_data:
                mp_id = data_dict.get('mp_id', data_dict.get('material_id'))
                feat = feat_dict[mp_id]
                if scaler is not None:
                    feat = (feat - scaler['mean'].squeeze(0)) / \
                        scaler['std'].squeeze(0)
                feat_list.append(feat)

            self.llm_feat = torch.stack(feat_list)  # [N, 4096]

    def __getitem__(self, index):
        data = super().__getitem__(index)

        if self.llm_feat is not None:
            data.llm_feat = self.llm_feat[index]

        return data
