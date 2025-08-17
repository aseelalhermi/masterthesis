import torch
import torch.nn as nn

class CrossAttentionDecoder(nn.Module):
    def __init__(self, feature_dim=32, time_dim=32, num_heads=4):
        super().__init__()
        self.time_proj = nn.Linear(time_dim, feature_dim)
        # Cross-attention: query from time_emb, key/value from x_feat
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(feature_dim)

        # Feed-forward decoder
        self.ff = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
            #nn.Sigmoid()
        )

    def forward(self, x_feat, time_emb):
        """
        x_feat: (batch, seq, feature_dim)   -> K, V
        time_emb: (batch, time, feature_dim) -> Q
        """
        # Cross-attention: time_emb queries the feature sequence
        attn_out, _ = self.cross_attn(query=time_emb, key=x_feat, value=x_feat)
        out = self.norm1(attn_out + time_emb)   # residual connection

        # Decoder head applied on each (batch, time, feature_dim)
        hazard_probs = self.ff(out).squeeze(-1)  # (batch, time)
        return hazard_probs


class SAINTWithTime(nn.Module):
    def __init__(self, base_saint_model, dim, time_points, time_embedding_type='learnable'):
        super().__init__()
        self.base_model = base_saint_model
        self.dim = dim
        raw_time_points = torch.tensor(time_points, dtype=torch.float32)
        self.register_buffer('time_points', (raw_time_points - raw_time_points.min()) / (raw_time_points.max() - raw_time_points.min()))
        self.num_time_points = len(self.time_points)
        self.time_embedding_type = time_embedding_type
        self.decoder = CrossAttentionDecoder(feature_dim=self.dim, num_heads=4)
        if time_embedding_type == 'learnable':
            self.time_embedding = nn.Embedding(self.num_time_points, dim)

    def forward(self, x_categ, x_cont, time_ids):
        x_feat = self.base_model.transformer(x_categ, x_cont)
        device = x_feat.device
        # Ensure time_points is on the correct device once
        time_points = self.time_points.to(device)
        time_ids = time_ids.to(device)
        if self.time_embedding_type == 'learnable':
            time_ids_idx = torch.searchsorted(time_points, time_ids.float()).long()
            time_ids_idx = time_ids_idx.to(device)
            time_emb = self.time_embedding(time_ids_idx.clamp(max=self.num_time_points - 1))
            time_emb = time_emb.to(device)
            x = x_feat
        
        if self.time_embedding_type == 'none':
            x = x_feat[:, None, :]

        feature_dim = x_feat.size(-1)
        hazard_probs = self.decoder(x_feat, time_emb)  # (256, 3)
        return hazard_probs
