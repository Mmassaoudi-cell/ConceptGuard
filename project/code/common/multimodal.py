"""
Modality blocks for CrossGuard-RL (Candidate 7), built from Edge-IIoTset's genuine per-protocol
parser fields. A flow's http.*/mqtt.*/dns.*/... columns are NaN whenever that protocol isn't the
one active on the flow — this is REAL missingness from the packet-parsing process, not something
fabricated for this experiment. Presence masks below are computed from that natural NaN structure.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

MODALITY_BLOCKS = {
    "arp": ["arp.dst.proto_ipv4", "arp.opcode", "arp.hw.size", "arp.src.proto_ipv4"],
    "icmp": ["icmp.checksum", "icmp.seq_le", "icmp.transmit_timestamp", "icmp.unused"],
    "http": ["http.file_data", "http.content_length", "http.request.uri.query", "http.request.method",
             "http.referer", "http.request.full_uri", "http.request.version", "http.response", "http.tls_port"],
    "tcp": ["tcp.ack", "tcp.ack_raw", "tcp.checksum", "tcp.connection.fin", "tcp.connection.rst",
            "tcp.connection.syn", "tcp.connection.synack", "tcp.dstport", "tcp.flags", "tcp.flags.ack",
            "tcp.len", "tcp.options", "tcp.payload", "tcp.seq", "tcp.srcport"],
    "udp": ["udp.port", "udp.stream", "udp.time_delta"],
    "dns": ["dns.qry.name", "dns.qry.name.len", "dns.qry.qu", "dns.qry.type", "dns.retransmission",
            "dns.retransmit_request", "dns.retransmit_request_in"],
    "mqtt": ["mqtt.conack.flags", "mqtt.conflag.cleansess", "mqtt.conflags", "mqtt.hdrflags", "mqtt.len",
             "mqtt.msg_decoded_as", "mqtt.msg", "mqtt.msgtype", "mqtt.proto_len", "mqtt.protoname",
             "mqtt.topic", "mqtt.topic_len", "mqtt.ver"],
    "mbtcp": ["mbtcp.len", "mbtcp.trans_id", "mbtcp.unit_id"],
}
CONTEXT_COLS = ["ip.src_host", "ip.dst_host"]


def compute_presence_masks(df: pd.DataFrame) -> dict:
    """1.0 if ANY column in the block is non-null for that row, else 0.0 (all-null = protocol absent)."""
    masks = {}
    for name, cols in MODALITY_BLOCKS.items():
        present_cols = [c for c in cols if c in df.columns]
        if not present_cols:
            masks[name] = np.zeros(len(df), dtype=np.float32)
            continue
        masks[name] = (~df[present_cols].isna()).any(axis=1).astype(np.float32).values
    return masks


def block_to_numeric(df: pd.DataFrame, cols) -> np.ndarray:
    """Best-effort numeric coercion per block: numeric cols -> as-is, non-numeric (strings/hex ids)
    -> factorized to small integer codes. NaN -> 0 after coercion (masked out via presence mask
    anyway, so the fill value doesn't leak information). NOT scaled here -- raw fields like
    tcp.seq/tcp.ack are 32-bit magnitudes that would dominate an unscaled loss; use
    fit_block_scaler/apply_block_scaler (fit on train only) before feeding a model."""
    present_cols = [c for c in cols if c in df.columns]
    if not present_cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    out = []
    for c in present_cols:
        col = df[c]
        if pd.api.types.is_numeric_dtype(col):
            out.append(pd.to_numeric(col, errors="coerce").fillna(0.0).values.astype(np.float32))
        else:
            codes, _ = pd.factorize(col.astype(str))
            out.append(codes.astype(np.float32))
    return np.stack(out, axis=1)


def fit_block_scaler(feats_np: dict) -> dict:
    """feats_np: {block_name: (N, dim) array}, TRAIN split only. Returns {block_name: (mean, std)}."""
    scalers = {}
    for name, arr in feats_np.items():
        if arr.shape[1] == 0:
            scalers[name] = (np.zeros(0, dtype=np.float32), np.ones(0, dtype=np.float32))
            continue
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        std[std < 1e-6] = 1.0
        scalers[name] = (mean.astype(np.float32), std.astype(np.float32))
    return scalers


def apply_block_scaler(feats_np: dict, scalers: dict) -> dict:
    return {name: (arr - scalers[name][0]) / scalers[name][1] if arr.shape[1] > 0 else arr
            for name, arr in feats_np.items()}


class ModalityEncoder(nn.Module):
    def __init__(self, block_dims: dict, shared_dim=32, fused_dim=64):
        super().__init__()
        self.names = list(block_dims.keys())
        self.encoders = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(max(dim, 1), shared_dim), nn.ReLU(), nn.Linear(shared_dim, shared_dim))
            for name, dim in block_dims.items()
        })
        self.fuse = nn.Linear(shared_dim, fused_dim)

    def forward(self, block_feats: dict, block_masks: dict, dropout_p=0.0, training_dropout=False):
        """block_feats[name]: (B, dim_name) tensor. block_masks[name]: (B,) 1/0 presence.
        Returns fused (B, fused_dim) via masked mean over present (and non-dropped) modalities,
        plus per-modality embeddings (for contrastive loss)."""
        per_modality = {}
        B = next(iter(block_feats.values())).shape[0]
        device = next(iter(block_feats.values())).device
        eff_mask_sum = torch.zeros(B, device=device)
        fused_accum = torch.zeros(B, self.fuse.out_features, device=device)

        for name in self.names:
            emb = self.encoders[name](block_feats[name])  # (B, shared_dim)
            mask = block_masks[name].clone()
            if training_dropout and dropout_p > 0:
                drop = (torch.rand(B, device=device) < dropout_p).float()
                mask = mask * (1 - drop)  # randomly withhold a present modality
            proj = self.fuse(emb)  # (B, fused_dim) -- same space as `fused`, needed for contrastive loss
            per_modality[name] = proj
            fused_accum += proj * mask.unsqueeze(1)
            eff_mask_sum += mask

        fused = fused_accum / eff_mask_sum.clamp(min=1).unsqueeze(1)
        return fused, per_modality


def contrastive_alignment_loss(fused, per_modality: dict, block_masks: dict, temperature=0.2):
    """InfoNCE-style alignment: each present modality's embedding should be close to the SAME
    flow's fused embedding and far from other flows' fused embeddings in the batch."""
    fused_n = F.normalize(fused, dim=1)
    total_loss, n_terms = 0.0, 0
    for name, emb in per_modality.items():
        mask = block_masks[name]
        if mask.sum() < 2:
            continue
        emb_n = F.normalize(emb, dim=1)
        logits = emb_n @ fused_n.t() / temperature  # (B, B)
        targets = torch.arange(emb.shape[0], device=emb.device)
        loss_i = F.cross_entropy(logits, targets, reduction="none")
        loss_i = (loss_i * mask).sum() / mask.sum().clamp(min=1)
        total_loss = total_loss + loss_i
        n_terms += 1
    return total_loss / max(n_terms, 1)
