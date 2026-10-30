"""Aligns CNN re-ID features (OSNet / ResNetIBN) with the frozen CLIP text space.

The backbone feature is projected into CLIP's joint embedding space and trained against the
CLIP text embedding of each vehicle's caption (data/text_labels.json). At inference a text query
is encoded with the same CLIP text encoder and compared with the projected image features.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from utils.losses import SupConLoss


class TextAlignHead(nn.Module):
    def __init__(self, in_dim, clip_dim=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, clip_dim),
            nn.LayerNorm(clip_dim)
        )
        # CLIP's initial temperature, learnable as in CLIP
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))
        self.criterion = SupConLoss()

    def forward(self, feat):
        """Backbone feature [B, in_dim] -> L2-normalised CLIP-space embedding [B, clip_dim]"""
        return F.normalize(self.proj(feat), dim=-1)

    def loss(self, feat, caption_bank, caption_ids):
        """caption_bank: [C, clip_dim] embeddings of all training captions; caption_ids: [B] rows of it.

        Image -> text: cross-entropy over the whole bank, so every image is contrasted with all
        training captions instead of only the ~16 vehicles in its batch.
        In-batch SupCon: positives are matched by caption, not identity: different vehicles can share
        a caption, and their (identical) text embeddings must not be pushed apart.
        """
        labelled = caption_ids >= 0  # -1: image without a trusted caption
        if not labelled.any():
            return feat.sum() * 0
        feat, caption_ids = feat[labelled], caption_ids[labelled]
        scale = self.logit_scale.exp().clamp(max=100)
        image_embeds = self(feat)
        bank_loss = F.cross_entropy(scale * image_embeds @ caption_bank.t(), caption_ids)
        return bank_loss + self.criterion(image_embeds, caption_bank[caption_ids], caption_ids, scale)


def text_retrieval_map(image_embeds, image_queries, query_embeds):
    """mAP of text -> image retrieval.

    image_embeds:  [N, D] normalised image embeddings
    image_queries: length-N list, the query string each image is relevant to (None: no query for
                   that image; it is still ranked as a distractor)
    query_embeds:  dict query string -> [D] normalised text embedding
    Returns (mAP, R@1, R@5) averaged over the distinct queries.
    """
    image_queries = np.asarray(image_queries, dtype=object)
    image_embeds = torch.as_tensor(image_embeds).float()
    aps, r1, r5 = [], [], []
    for query in sorted(set(image_queries) - {None}):
        scores = image_embeds @ torch.as_tensor(query_embeds[query]).float()
        relevant = image_queries[scores.argsort(descending=True).numpy()] == query
        hits = np.cumsum(relevant)
        aps.append(float((hits / np.arange(1, len(relevant) + 1))[relevant].mean()))
        r1.append(float(relevant[:1].any()))
        r5.append(float(relevant[:5].any()))
    return float(np.mean(aps)), float(np.mean(r1)), float(np.mean(r5))


@torch.no_grad()
def build_caption_bank(captions, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
    """Encodes each unique caption once with the frozen CLIP text encoder.

    Returns (caption -> row index, tensor [N, clip_dim]). The CLIP model is released afterwards.
    """
    from CLIP_Query.clip_encoder import ClipEncoder

    unique = sorted(set(captions) - {''})
    encoder = ClipEncoder(model_name, pretrained)
    bank = torch.from_numpy(np.stack([encoder.encode_text(c) for c in unique])).float()
    del encoder
    return {c: i for i, c in enumerate(unique)}, bank
