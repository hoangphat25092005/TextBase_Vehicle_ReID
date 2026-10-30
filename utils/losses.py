import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """Batch-hard triplet loss for Re-ID training.

    Distances are computed on L2-normalised features (the same space used for cosine retrieval), so the
    margin is on the cosine-distance scale instead of being negligible next to raw feature distances.
    The anchor itself is not counted as a positive; anchors without a positive in the batch are skipped.
    """

    def __init__(self, margin=0.3, normalize=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.normalize = normalize
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, features, labels):
        """
        Args:
            features: feature matrix with shape (batch_size, feat_dim)
            labels: ground truth labels with shape (batch_size)
        """
        if self.normalize:
            features = F.normalize(features, p=2, dim=1)
        n = features.size(0)

        # Compute pairwise distance matrix
        dist = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist = dist.addmm(features, features.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()

        same = labels.expand(n, n).eq(labels.expand(n, n).t())
        eye = torch.eye(n, dtype=torch.bool, device=features.device)
        pos_mask = same & ~eye
        has_pos = pos_mask.any(dim=1)
        if not has_pos.any():
            return features.sum() * 0  # no positive pair in this batch

        # For each anchor, the hardest (farthest) positive and hardest (closest) negative
        dist_ap = dist.masked_fill(~pos_mask, 0).max(dim=1).values[has_pos]
        dist_an = dist.masked_fill(same, float('inf')).min(dim=1).values[has_pos]

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)


class CenterLoss(nn.Module):
    """Center Loss for Re-ID training"""
    
    def __init__(self, num_classes, feat_dim, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu and torch.cuda.is_available()  # Only use GPU if available

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
    
    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim)
            labels: ground truth labels with shape (batch_size)
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)
        
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        
        return loss


class SupConLoss(nn.Module):
    """Symmetric supervised contrastive loss between image and prompt embeddings (PromptSG Eq. 4-6).
    Every image/prompt of the same identity in the batch counts as a positive."""

    def forward(self, image_embeds, text_embeds, labels, logit_scale):
        image_embeds = F.normalize(image_embeds, dim=1)
        text_embeds = F.normalize(text_embeds, dim=1)
        logits = logit_scale * image_embeds @ text_embeds.t()
        positives = labels.unsqueeze(1).eq(labels.unsqueeze(0)).float()
        return self._sup_con(logits, positives) + self._sup_con(logits.t(), positives)

    @staticmethod
    def _sup_con(logits, positives):
        log_prob = F.log_softmax(logits, dim=1)
        return -((log_prob * positives).sum(dim=1) / positives.sum(dim=1)).mean()