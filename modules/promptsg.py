"""
PromptSG (Prompt-driven Semantic Guidance) adapted for vehicle Re-ID.

Yang et al., "A Pedestrian is Worth One Prompt: Towards Language Guidance Person
Re-Identification", CVPR 2024.

- CLIP ViT-B/16 visual encoder (fine-tuned) + CLIP text encoder (frozen).
- An inversion network maps the global image embedding to a pseudo-word token s*,
  which is composed into the prompt "a photo of a s* vehicle."
- A multimodal interaction module (1 cross-attention layer, text queries / patch keys+values,
  then 2 self-attention blocks) produces the language-guided image feature.
- At inference the prompt can be the composed one (accurate, runs the text encoder per crop)
  or the fixed "a photo of a vehicle." (simplified, text features are computed once).
"""
import cv2
import numpy as np
import open_clip
import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

PLACEHOLDER = "*"
COMPOSED_PROMPT = f"a photo of a {PLACEHOLDER} vehicle."
SIMPLIFIED_PROMPT = "a photo of a vehicle."


class InversionNetwork(nn.Module):
    """Three-layer MLP (512 hidden) with BN on the output: image embedding -> pseudo token s*."""

    def __init__(self, in_dim, token_dim, hidden_dim=512, token_std=0.02):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, token_dim),
        )
        self.bn = nn.BatchNorm1d(token_dim)
        # Start at the scale of real CLIP word embeddings instead of unit variance
        nn.init.constant_(self.bn.weight, token_std)

    def forward(self, x):
        return self.bn(self.mlp(x))


class MultimodalInteraction(nn.Module):
    """Language-guided cross-attention (Eq. 7) followed by self-attention transformer blocks."""

    def __init__(self, dim=512, num_heads=8, num_self_layers=2, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        layer = nn.TransformerEncoderLayer(dim, num_heads, dim * 4, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.self_blocks = nn.TransformerEncoder(layer, num_self_layers, enable_nested_tensor=False)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, text_tokens, patch_tokens, eot_idx):
        # Z = Softmax(QK^T / sqrt(d)) V: each prompt token gathers the patches that match its semantics
        kv = self.norm_kv(patch_tokens)
        z, _ = self.cross_attn(self.norm_q(text_tokens), kv, kv, need_weights=False)
        z = self.norm_out(self.self_blocks(z))
        return z[:, eot_idx]


class PromptSG(nn.Module):
    def __init__(self, num_classes, model_name="ViT-B-16", pretrained="openai",
                 inference_prompt="simplified"):
        super().__init__()
        assert inference_prompt in ("simplified", "composed")
        self.inference_prompt = inference_prompt
        self.model_name = model_name

        # OpenAI CLIP weights were trained with QuickGELU; force it so loading our own
        # checkpoint (pretrained=None) builds the same network
        clip_model = open_clip.create_model(model_name, pretrained=pretrained, force_quick_gelu=True)
        tokenizer = open_clip.get_tokenizer(model_name)

        self.visual = clip_model.visual

        # Frozen text encoder
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.text_transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.logit_scale = clip_model.logit_scale
        for frozen in (self.token_embedding, self.text_transformer, self.ln_final,
                       self.text_projection, self.positional_embedding, self.logit_scale):
            frozen.requires_grad_(False)

        # Both prompts end at the same EOT position, so the text sequence is cut there:
        # the causal mask means tokens after EOT never influence it
        composed = tokenizer([COMPOSED_PROMPT])
        simplified = tokenizer([SIMPLIFIED_PROMPT])
        placeholder_id = tokenizer([PLACEHOLDER])[0, 1]
        self.placeholder_pos = int((composed[0] == placeholder_id).nonzero()[0])
        self.composed_len = int(composed.argmax(dim=-1)) + 1
        self.simplified_len = int(simplified.argmax(dim=-1)) + 1
        self.register_buffer("composed_tokens", composed[:, :self.composed_len], persistent=False)
        self.register_buffer("simplified_tokens", simplified[:, :self.simplified_len], persistent=False)
        self.register_buffer("attn_mask", clip_model.attn_mask, persistent=False)
        self._simplified_text = None

        embed_dim = self.visual.output_dim
        token_dim = self.token_embedding.weight.shape[1]
        self.inversion = InversionNetwork(embed_dim, token_dim,
                                          token_std=float(self.token_embedding.weight.std()))
        self.interaction = MultimodalInteraction(embed_dim)

        # BNNeck + ID classifiers for the language-guided feature and the CLIP image feature
        self.bn_fused = nn.BatchNorm1d(embed_dim)
        self.bn_image = nn.BatchNorm1d(embed_dim)
        for bn in (self.bn_fused, self.bn_image):
            bn.bias.requires_grad_(False)
        self.classifier_fused = nn.Linear(embed_dim, num_classes, bias=False)
        self.classifier_image = nn.Linear(embed_dim, num_classes, bias=False)
        for classifier in (self.classifier_fused, self.classifier_image):
            nn.init.normal_(classifier.weight, std=0.001)

        self.feature_dim = 2 * embed_dim

    def encode_image(self, images):
        """Returns the pre-projection CLS (768), the projected CLS (512) and projected patch tokens."""
        visual = self.visual
        x = visual.transformer(visual._embeds(images))
        cls, patches = visual._pool(x)
        return cls, cls @ visual.proj, patches @ visual.proj

    def encode_prompt(self, tokens, pseudo_token=None):
        """Runs the frozen text encoder; returns every token projected to the joint space [B, L, D]."""
        length = tokens.shape[1]
        x = self.token_embedding(tokens)
        if pseudo_token is not None:
            p = self.placeholder_pos
            x = torch.cat([x[:, :p], pseudo_token.unsqueeze(1).to(x.dtype), x[:, p + 1:]], dim=1)
        x = x + self.positional_embedding[:length]
        x = self.text_transformer(x, attn_mask=self.attn_mask[:length, :length])
        x = self.ln_final(x)
        if isinstance(self.text_projection, nn.Module):
            return self.text_projection(x)
        return x @ self.text_projection

    def _simplified_prompt(self, batch_size):
        # The text encoder is frozen, so the simplified prompt only has to be encoded once
        if self._simplified_text is None or self._simplified_text.device != self.simplified_tokens.device:
            with torch.no_grad():
                self._simplified_text = self.encode_prompt(self.simplified_tokens)
        return self._simplified_text.expand(batch_size, -1, -1)

    def forward(self, images):
        cls, image_embed, patches = self.encode_image(images)
        batch_size = images.shape[0]

        if self.training or self.inference_prompt == "composed":
            pseudo_token = self.inversion(image_embed)
            text = self.encode_prompt(self.composed_tokens.expand(batch_size, -1), pseudo_token)
            eot = self.composed_len - 1
        else:
            text = self._simplified_prompt(batch_size)
            eot = self.simplified_len - 1

        fused = self.interaction(text, patches, eot)
        fused_bn = self.bn_fused(fused)
        image_bn = self.bn_image(image_embed)

        if self.training:
            return {
                "logits": [self.classifier_fused(fused_bn), self.classifier_image(image_bn)],
                "triplet_features": [fused, image_embed, cls],
                "image_embed": image_embed,
                "text_embed": text[:, eot],
            }
        return torch.cat([fused_bn, image_bn], dim=1)


def build_transforms(input_size=(224, 224), train=False):
    if train:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(input_size, antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop(input_size),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), value="random"),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(input_size, antialias=True),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


class PromptSGDescriptor:
    """Drop-in replacement for VehicleDescriptor: extract_feature(BGR crop) -> L2-normalised tensor."""

    def __init__(self, model_path=None, inference_prompt="simplified", input_size=(224, 224)):
        self.model_path = model_path
        self.model_type = "promptsg"
        self.input_size = input_size

        checkpoint = None
        if model_path:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        num_classes = checkpoint["num_classes"] if checkpoint else 1
        model_name = checkpoint.get("model_name", "ViT-B-16") if checkpoint else "ViT-B-16"

        # A checkpoint holds every CLIP weight, so skip downloading the pretrained ones
        self.model = PromptSG(num_classes, model_name=model_name,
                              pretrained=None if checkpoint else "openai",
                              inference_prompt=inference_prompt)
        if checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Successfully loaded PromptSG weights from {model_path}")
        else:
            print("Warning: no PromptSG checkpoint given, using untrained heads on top of CLIP")

        self.model = self.model.to(device)
        self.model.eval()
        self.feature_dim = self.model.feature_dim
        self.transforms = build_transforms(input_size)
        print(f"Using model type: promptsg ({inference_prompt} prompt) with input size: {input_size} "
              f"and feature dimension: {self.feature_dim}")

    def extract_feature(self, input_img: np.ndarray) -> torch.Tensor:
        # Detector crops are OpenCV BGR; CLIP expects RGB
        img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        img = self.transforms(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = self.model(img).squeeze(0)
        return F.normalize(features, p=2, dim=0)

    def get_model_info(self):
        return {
            'model_type': self.model_type,
            'inference_prompt': self.model.inference_prompt,
            'input_size': self.input_size,
            'feature_dim': self.feature_dim,
            'device': str(device),
            'parameters': sum(p.numel() for p in self.model.parameters())
        }
