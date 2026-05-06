"""CondCSPNet: CSPNet with LLaMA-3 hidden state conditioning.

Per-atom additive injection after atom_latent_emb, before message passing.
"""
import torch
import torch.nn as nn
from torch_scatter import scatter

from models_ddpm.cspnet import CSPNet, SinusoidsEmbedding, CSPLayer, MAX_ATOMIC_NUM


class CondCSPNet(CSPNet):
    """CSPNet subclass that accepts LLM features as per-atom conditioning."""

    def __init__(self, llm_feat_dim=4096, **kwargs):
        super().__init__(**kwargs)

        self.llm_feat_dim = llm_feat_dim
        hidden_dim = kwargs.get('hidden_dim', 128)

        # Projection MLP: llm_feat_dim -> hidden_dim
        self.llm_proj = nn.Sequential(
            nn.Linear(llm_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t, atom_types, frac_coords, lattices, num_atoms,
                node2graph, llm_feat=None):
        edges, frac_diff = self.gen_edges(
            num_atoms, frac_coords, lattices, node2graph)
        edge2graph = node2graph[edges[0]]

        if self.smooth:
            node_features = self.node_embedding(atom_types)
        else:
            node_features = self.node_embedding(atom_types - 1)

        t_per_atom = t.repeat_interleave(num_atoms, dim=0)

        node_features = torch.cat([node_features, t_per_atom], dim=1)
        node_features = self.atom_latent_emb(node_features)

        # ---- LLM conditioning injection ----
        if llm_feat is not None:
            llm_proj = self.llm_proj(llm_feat.float())   # [B, hidden_dim]
            llm_per_atom = llm_proj[node2graph]           # [N, hidden_dim]
            node_features = node_features + llm_per_atom   # additive
        # -----------------------------------

        for i in range(0, self.num_layers):
            node_features = self._modules["csp_layer_%d" % i](
                node_features, frac_coords, lattices, edges,
                edge2graph, frac_diff=frac_diff)

        if self.ln:
            node_features = self.final_layer_norm(node_features)

        coord_out = self.coord_out(node_features)

        graph_features = scatter(
            node_features, node2graph, dim=0, reduce='mean')
        lattice_out = self.lattice_out(graph_features)
        lattice_out = lattice_out.view(-1, 3, 3)
        if self.ip:
            if self.run_type == 'sample':
                lattice_out = lattice_out.double()
                lattices = lattices.double()
            lattice_out = torch.einsum(
                'bij,bjk->bik', lattice_out, lattices)
        if self.pred_type:
            type_out = self.type_out(node_features)
            return lattice_out, coord_out, type_out

        return lattice_out, coord_out
