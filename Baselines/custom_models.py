import torch
import torch.nn as nn

class TransformerSurvivalModel2(nn.Module):
    def __init__(self, hidden_size, num_heads, num_layers, dim_feedforward, dropout,
                 num_num_features, cat_cardinalities):
        super().__init__()

        # --- categorical embeddings (each = hidden_size) ---
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, hidden_size) for card in cat_cardinalities
        ])
        cat_total_dim = hidden_size * len(cat_cardinalities)

        # --- numeric projection to hidden_size ---
        self.num_proj = nn.Linear(num_num_features, hidden_size)

        # --- combine projected num + cat embeddings ---
        self.combine_proj = nn.Linear(cat_total_dim + hidden_size, hidden_size)

        # --- transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- final output ---
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_num, x_cat_tokens):
        # embed categorical features (shape: [batch, cat, hidden_size])
        embedded_cat = [emb(x_cat_tokens[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        x_cat = torch.cat(embedded_cat, dim=-1)  # shape: [batch, cat * hidden_size]

        # project numeric features
        x_num_proj = self.num_proj(x_num)  # shape: [batch, hidden_size]

        # combine numeric + categorical embeddings, project to hidden_size
        x = torch.cat([x_num_proj, x_cat], dim=-1)
        x = self.combine_proj(x)  # shape: [batch, hidden_size]

        # transformer expects (batch, seq_len, hidden_size)
        x = x.unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)

        return self.fc(x).squeeze(-1)



class DeepHitTransformerSurvivalModel2(nn.Module):
    def __init__(self, hidden_size, num_heads, num_layers, dim_feedforward, dropout,
                 num_num_features, cat_cardinalities,num_durations):
        super().__init__()

        # --- categorical embeddings (each = hidden_size) ---
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, hidden_size) for card in cat_cardinalities
        ])
        cat_total_dim = hidden_size * len(cat_cardinalities)

        # --- numeric projection to hidden_size ---
        self.num_proj = nn.Linear(num_num_features, hidden_size)

        # --- combine projected num + cat embeddings ---
        self.combine_proj = nn.Linear(cat_total_dim + hidden_size, hidden_size)

        # --- transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- final output ---
        self.fc = nn.Linear(hidden_size, num_durations)

    def forward(self, x_num, x_cat_tokens):
        # embed categorical features (shape: [batch, cat, hidden_size])
        embedded_cat = [emb(x_cat_tokens[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        x_cat = torch.cat(embedded_cat, dim=-1)  # shape: [batch, cat * hidden_size]

        # project numeric features
        x_num_proj = self.num_proj(x_num)  # shape: [batch, hidden_size]

        # combine numeric + categorical embeddings, project to hidden_size
        x = torch.cat([x_num_proj, x_cat], dim=-1)
        x = self.combine_proj(x)  # shape: [batch, hidden_size]

        # transformer expects (batch, seq_len, hidden_size)
        x = x.unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)

        return self.fc(x).squeeze(-1)

def get_deepsurv_net2(input_dim_num, cat_cardinalities, hidden_size, num_nodes1, num_nodes2, dropout):
    # --- categorical embeddings: each feature → hidden_size vector ---
    emb_layers = nn.ModuleList([
        nn.Embedding(card, hidden_size) for card in cat_cardinalities
    ])
    num_cats = len(cat_cardinalities)

    class DeepSurvModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = emb_layers

            # numeric features projection → hidden_size
            self.num_proj = nn.Linear(input_dim_num, hidden_size)

            # combine projection (numeric + categorical)
            self.combine_proj = nn.Linear((num_cats + 1) * hidden_size, hidden_size)

            # MLP backbone
            self.backbone = nn.Sequential(
                nn.Linear(hidden_size, num_nodes1),
                nn.ReLU(),
                nn.BatchNorm1d(num_nodes1),
                nn.Dropout(dropout),
                nn.Linear(num_nodes1, num_nodes2),
                nn.ReLU(),
                nn.BatchNorm1d(num_nodes2),
                nn.Dropout(dropout),
                nn.Linear(num_nodes2, 1)
            )

        def forward(self, x_num, x_cat_tokens):
            # embed categorical features
            embedded_cat = [emb(x_cat_tokens[:, i]) for i, emb in enumerate(self.embeddings)]
            x_cat = torch.cat(embedded_cat, dim=-1)  # shape = [batch, num_cats * hidden_size]

            # numeric projection
            x_num_proj = self.num_proj(x_num)        # shape = [batch, hidden_size]

            # combine numeric + categorical → hidden_size
            x = torch.cat([x_num_proj, x_cat], dim=-1)
            x = self.combine_proj(x)                 # shape = [batch, hidden_size]

            return self.backbone(x)

    return DeepSurvModel()

def get_deephit_net2(input_dim_num, cat_cardinalities, hidden_size, num_nodes, dropout, num_durations):
    # --- categorical embeddings: each feature → hidden_size vector ---
    emb_layers = nn.ModuleList([
        nn.Embedding(card, hidden_size) for card in cat_cardinalities
    ])
    num_cats = len(cat_cardinalities)

    class DeepHitModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = emb_layers

            # numeric projection → hidden_size
            self.num_proj = nn.Linear(input_dim_num, hidden_size)

            # combine projection (numeric + categorical)
            self.combine_proj = nn.Linear((num_cats + 1) * hidden_size, hidden_size)

            # backbone (predicting discrete hazard for each duration)
            self.backbone = nn.Sequential(
                nn.Linear(hidden_size, num_nodes),
                nn.ReLU(),
                nn.BatchNorm1d(num_nodes),
                nn.Dropout(dropout),
                nn.Linear(num_nodes, num_nodes),
                nn.ReLU(),
                nn.BatchNorm1d(num_nodes),
                nn.Dropout(dropout),
                nn.Linear(num_nodes, num_durations)
            )

        def forward(self, x_num, x_cat_tokens):
            # embed categorical features
            embedded_cat = [emb(x_cat_tokens[:, i]) for i, emb in enumerate(self.embeddings)]
            x_cat = torch.cat(embedded_cat, dim=-1)  # shape: [batch, num_cats * hidden_size]

            # numeric projection
            x_num_proj = self.num_proj(x_num)        # shape: [batch, hidden_size]

            # combine numeric + categorical → hidden_size
            x = torch.cat([x_num_proj, x_cat], dim=-1)
            x = self.combine_proj(x)                 # shape: [batch, hidden_size]

            return self.backbone(x)

    return DeepHitModel()

def get_optimizer(net, optimizer_type, lr, weight_decay=0.0):
    if optimizer_type == "Adam":
        return torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "AdamW":
        return torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("Unsupported optimizer type")

