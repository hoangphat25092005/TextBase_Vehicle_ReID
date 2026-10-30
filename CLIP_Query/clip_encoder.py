import numpy as np
import cv2
import torch
import open_clip
from PIL import Image
from typing import List, Optional


PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a CCTV image of a {}.",
    "a {} on the road.",
]


class ClipEncoder:
    """Wraps an open_clip model to embed vehicle crops and text queries into the same space."""

    def __init__(self,
                 model_name: str = "ViT-B-32",
                 pretrained: str = "laion2b_s34b_b79k",
                 device: Optional[str] = None,
                 batch_size: int = 32):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        print(f"[CLIP] Loaded {model_name} ({pretrained}) on {self.device}")

    @torch.no_grad()
    def encode_images(self, images_bgr: List[np.ndarray]) -> np.ndarray:
        """Embed OpenCV (BGR) crops. Returns L2-normalised array of shape [N, D]."""
        features = []
        for start in range(0, len(images_bgr), self.batch_size):
            chunk = images_bgr[start:start + self.batch_size]
            # OpenCV crops are BGR; CLIP was trained on RGB
            tensors = [
                self.preprocess(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                for img in chunk
            ]
            batch = torch.stack(tensors, dim=0).to(self.device)
            feats = self.model.encode_image(batch).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            features.append(feats.cpu().numpy())
        if not features:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.concatenate(features, axis=0)  # Shape something like this after concatenate [[], [], [], []]

    @torch.no_grad()
    def encode_text(self, query: str) -> np.ndarray:
        """Embed a text query with a prompt ensemble. Returns an L2-normalised vector [D]."""
        prompts = [template.format(query) for template in PROMPT_TEMPLATES]
        tokens = self.tokenizer(prompts).to(self.device) # Shape?
        feats = self.model.encode_text(tokens).float() # Shape?
        feats = feats / feats.norm(dim=-1, keepdim=True) # Shape?
        mean = feats.mean(dim=0)
        mean = mean / mean.norm()
        return mean.cpu().numpy() # Final Shape?

    @property
    def dim(self) -> int:
        return self.model.text_projection.shape[1]  # What is this use for
