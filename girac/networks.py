"""Temporal encoder and deterministic actor and critic networks."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class RotaryPositionalEncoding(nn.Module):
    """Rotary position embedding for queries and keys."""
    def __init__(self, d_model):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)

    def apply_rotary(self, x):
        """x: (batch, seq_len, dim), dim must be even."""
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        freqs = torch.outer(t, self.inv_freq.to(x.dtype))
        cos_f = freqs.cos().unsqueeze(0)
        sin_f = freqs.sin().unsqueeze(0)
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        out1 = x1 * cos_f - x2 * sin_f
        out2 = x2 * cos_f + x1 * sin_f
        return torch.cat([out1, out2], dim=-1)


class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.rope = RotaryPositionalEncoding(self.d_k)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # Rotate queries and keys independently for each head.
        Q = self.rope.apply_rotary(Q.reshape(B * self.n_heads, L, self.d_k))
        K = self.rope.apply_rotary(K.reshape(B * self.n_heads, L, self.d_k))
        Q = Q.view(B, self.n_heads, L, self.d_k)
        K = K.view(B, self.n_heads, L, self.d_k)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(out), attn


class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block."""
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = RoPEMultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h, attn_w = self.attn(self.ln1(x))
        x = x + h
        x = x + self.ffn(self.ln2(x))
        return x, attn_w


class GainInvariantEncoder(nn.Module):
    """Encode observation, control, output, and setpoint histories.

    Instance normalization and temporal differences describe the history.
    Relative errors and the appended setpoint retain tracking information.
    """

    def __init__(self, dim_o, dim_u, dim_y, window_len,
                 d_model, n_heads, n_layers, ffn_dim, dropout=0.0):
        super().__init__()
        self.dim_o = dim_o
        self.dim_u = dim_u
        self.dim_y = dim_y
        self.window_len = window_len
        self.token_dim = dim_o + dim_u + dim_y

        self.aug_dim = 2 * self.token_dim + dim_y

        self.input_proj = nn.Linear(self.aug_dim, d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)

        self.output_dim = d_model + dim_y

    def forward(self, flat_state):
        B = flat_state.shape[0]
        wl = self.window_len

        y_target = flat_state[:, -self.dim_y:]
        o_flat = flat_state[:, :wl * self.dim_o]
        u_flat = flat_state[:, wl * self.dim_o : wl * (self.dim_o + self.dim_u)]
        y_flat = flat_state[:, wl * (self.dim_o + self.dim_u) : wl * (self.dim_o + self.dim_u + self.dim_y)]

        o_win = o_flat.view(B, wl, self.dim_o)
        u_win = u_flat.view(B, wl, self.dim_u)
        y_win = y_flat.view(B, wl, self.dim_y)

        tokens = torch.cat([o_win, u_win, y_win], dim=-1)

        # Normalize each channel across the history window.
        mu = tokens.mean(dim=1, keepdim=True)
        sig = tokens.std(dim=1, keepdim=True, correction=1) + 1e-6
        tokens_norm = (tokens - mu) / sig

        tokens_diff = torch.zeros_like(tokens_norm)
        tokens_diff[:, 1:, :] = tokens_norm[:, 1:, :] - tokens_norm[:, :-1, :]

        # Keep tracking error and setpoint information outside normalization.
        y_tgt = y_target.unsqueeze(1).expand(-1, wl, -1)
        error_ratio = (y_win - y_tgt) / (torch.abs(y_win) + 0.1)

        aug = torch.cat([tokens_norm, tokens_diff, error_ratio], dim=-1)
        h = self.input_proj(aug)

        all_attn = []
        for layer in self.layers:
            h, attn_w = layer(h)
            all_attn.append(attn_w)
        h = self.final_ln(h)

        pooled = h.mean(dim=1)
        return torch.cat([pooled, y_target], dim=-1), all_attn


class GIRACActor(nn.Module):
    def __init__(self, encoder: GainInvariantEncoder, action_dim: int, hidden_size: int = 128):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder.output_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state, return_attn=False):
        features, attn = self.encoder(state)
        action = torch.tanh(self.head(features))
        if return_attn:
            return action, attn
        return action


class GIRACCritic(nn.Module):
    def __init__(self, encoder: GainInvariantEncoder, action_dim: int, hidden_size: int = 128):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder.output_dim + action_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        features, _ = self.encoder(state)
        return self.head(torch.cat([features, action], dim=-1))
