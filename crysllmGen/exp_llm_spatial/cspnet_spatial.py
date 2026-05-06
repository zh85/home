"""SpatialCrossAttnCSPNet: Cross-attention with sinusoidal coordinate encoding bias.

Each atom attends to 8 LLM condition tokens with a position-dependent bias,
so different spatial regions naturally attend to different condition tokens.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from models_ddpm.cspnet import CSPNet, MAX_ATOMIC_NUM


class SpatialCrossAttnCSPLayer(nn.Module):
    """Cross-attention layer with spatial bias from sinusoidal coordinate encoding."""

    def __init__(self, hidden_dim=128, num_tokens=8, num_freqs=4,
                 act_fn=nn.SiLU(), dis_emb=None, ln=False, ip=True):
        super().__init__()

        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.ip = ip
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        if dis_emb is not None:
            self.dis_dim = dis_emb.dim

        # Cross-attention QKV + output
        self.cross_q = nn.Linear(hidden_dim, hidden_dim)
        self.cross_k = nn.Linear(hidden_dim, hidden_dim)
        self.cross_v = nn.Linear(hidden_dim, hidden_dim)
        self.cross_out = nn.Linear(hidden_dim, hidden_dim)

        # Spatial encoder: frac_coords → per-token attention bias
        # Sinusoidal encoding: 3 coords × num_freqs × 2 (sin/cos) = 6*num_freqs
        self.num_freqs = num_freqs
        spatial_input_dim = 6 * num_freqs
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_input_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, num_tokens),
        )
        # Zero-init the final layer so spatial bias starts at zero
        nn.init.zeros_(self.spatial_encoder[-1].weight)
        nn.init.zeros_(self.spatial_encoder[-1].bias)

        # Register frequencies for sinusoidal encoding
        freqs = 2.0 ** torch.arange(0, num_freqs) * torch.pi
        self.register_buffer('freqs', freqs, persistent=True)

        # Edge and node MLPs
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 9 + self.dis_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn)

        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def _sinusoidal_encode(self, frac_coords):
        """Encode fractional coordinates with periodic sinusoidal features.

        Args:
            frac_coords: [..., 3] fractional coordinates in [0, 1)
        Returns:
            [..., 6*num_freqs] sin/cos features
        """
        x = frac_coords.unsqueeze(-1)  # [..., 3, 1]
        freqs = self.freqs.to(x.device)  # [num_freqs]
        phases = x * freqs  # [..., 3, num_freqs]
        sin_feats = torch.sin(phases).flatten(-2)  # [..., 3*num_freqs]
        cos_feats = torch.cos(phases).flatten(-2)  # [..., 3*num_freqs]
        return torch.cat([sin_feats, cos_feats], dim=-1)

    def _cross_attention(self, node_features, cond_tokens, frac_coords, node2graph):
        """Cross-attention with spatial bias on attention scores.

        Args:
            node_features: [N, d]
            cond_tokens:   [B, num_tokens, d]
            frac_coords:   [N, 3]
            node2graph:    [N] long
        Returns:
            [N, d]
        """
        N = node_features.shape[0]
        num_tokens = cond_tokens.shape[1]
        d = self.hidden_dim

        cond_per_atom = cond_tokens[node2graph]  # [N, num_tokens, d]

        Q = self.cross_q(node_features)          # [N, d]
        K = self.cross_k(cond_per_atom)          # [N, num_tokens, d]
        V = self.cross_v(cond_per_atom)          # [N, num_tokens, d]

        Q = Q.unsqueeze(1)                        # [N, 1, d]
        scores = Q @ K.transpose(-2, -1)          # [N, 1, num_tokens]
        scores = scores / math.sqrt(d)

        # Spatial bias: atom position → per-token attention preference
        spatial_encoded = self._sinusoidal_encode(frac_coords)  # [N, 6*freqs]
        spatial_bias = self.spatial_encoder(spatial_encoded)     # [N, num_tokens]
        scores = scores + spatial_bias.unsqueeze(1)              # [N, 1, num_tokens]

        attn = F.softmax(scores, dim=-1)
        attn_out = attn @ V          # [N, 1, d]
        attn_out = attn_out.squeeze(1)  # [N, d]

        return self.cross_out(attn_out)

    def edge_model(self, node_features, frac_coords, lattices,
                   edge_index, edge2graph, frac_diff=None):
        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        if frac_diff is None:
            xi, xj = frac_coords[edge_index[0]], frac_coords[edge_index[1]]
            frac_diff = (xj - xi) % 1.
        if self.dis_emb is not None:
            frac_diff = self.dis_emb(frac_diff)
        if self.ip:
            lattice_ips = lattices @ lattices.transpose(-1, -2)
        else:
            lattice_ips = lattices
        lattice_ips_flatten = lattice_ips.view(-1, 9)
        lattice_ips_flatten_edges = lattice_ips_flatten[edge2graph]
        edges_input = torch.cat(
            [hi, hj, lattice_ips_flatten_edges, frac_diff], dim=1)
        return self.edge_mlp(edges_input.float())

    def node_model(self, node_features, edge_features, edge_index):
        agg = scatter(edge_features, edge_index[0], dim=0, reduce='mean',
                      dim_size=node_features.shape[0])
        agg = torch.cat([node_features, agg], dim=1)
        return self.node_mlp(agg)

    def forward(self, node_features, cond_tokens, node2graph,
                frac_coords, lattices, edges, edge2graph, frac_diff=None):
        node_input = node_features

        if self.ln:
            node_features = self.layer_norm(node_input)

        if cond_tokens is not None:
            ca_out = self._cross_attention(
                node_features, cond_tokens, frac_coords, node2graph)
            node_features = node_features + ca_out

        edge_features = self.edge_model(
            node_features, frac_coords, lattices,
            edges, edge2graph, frac_diff)
        node_output = self.node_model(node_features, edge_features, edges)

        return node_input + node_output


class SpatialCrossAttnCSPNet(CSPNet):
    """CSPNet with per-layer spatial cross-attention over LLM condition tokens."""

    def __init__(self, llm_feat_dim=4096, num_tokens=8, spatial_num_freqs=4, **kwargs):
        super().__init__(**kwargs)

        self.num_tokens = num_tokens
        hidden_dim = kwargs.get('hidden_dim', 128)

        for i in range(self.num_layers):
            self._modules["csp_layer_%d" % i] = SpatialCrossAttnCSPLayer(
                hidden_dim, num_tokens, spatial_num_freqs, self.act_fn, self.dis_emb,
                ln=self.ln, ip=self.ip)

    def forward(self, t, atom_types, frac_coords, lattices, num_atoms,
                node2graph, llm_feat=None):
        edges, frac_diff = self.gen_edges(
            num_atoms, frac_coords, lattices, node2graph)
        edge2graph = node2graph[edges[0]]

        cond_tokens = None
        if llm_feat is not None:
            B = llm_feat.shape[0]
            d = llm_feat.shape[1]
            token_dim = d // self.num_tokens
            cond_tokens = llm_feat.float().view(B, self.num_tokens, token_dim)

        if self.smooth:
            node_features = self.node_embedding(atom_types)
        else:
            node_features = self.node_embedding(atom_types - 1)

        t_per_atom = t.repeat_interleave(num_atoms, dim=0)
        node_features = torch.cat([node_features, t_per_atom], dim=1)
        node_features = self.atom_latent_emb(node_features)

        for i in range(self.num_layers):
            node_features = self._modules["csp_layer_%d" % i](
                node_features, cond_tokens, node2graph,
                frac_coords, lattices, edges, edge2graph,
                frac_diff=frac_diff)

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
