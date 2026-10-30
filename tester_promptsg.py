import os
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import VehicleReIDDataset
from utils.compute_metrics import compute_cmc_map
from modules.promptsg import PromptSGDescriptor, build_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract(model, loader):
    features, labels, camera_ids = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Extracting features'):
            features.append(F.normalize(model(batch['image'].to(device)), dim=1).cpu())
            labels.extend(batch['label'].numpy())
            camera_ids.extend(batch['camera_id'].numpy())
    return torch.cat(features), np.array(labels), np.array(camera_ids)


def main():
    parser = argparse.ArgumentParser(description='Evaluate PromptSG on the test split (cross-camera mAP and CMC)')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--image_dir', type=str, default=None, help='Defaults to <data_dir>/images/test')
    parser.add_argument('--model_path', type=str, default='checkpoints_promptsg/best_promptsg_model.pth')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_results', type=str, default=None, help='Path to save the results (JSON)')
    args = parser.parse_args()

    image_dir = args.image_dir or os.path.join(args.data_dir, 'images', 'test')
    dataset = VehicleReIDDataset(args.data_dir, image_dir, split='test', transform=build_transforms())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Test samples: {len(dataset)}, vehicles: {dataset.num_classes}")

    descriptor = PromptSGDescriptor(model_path=args.model_path)
    results = {}
    # Simplified: fixed "a photo of a vehicle." prompt; composed: per-image pseudo-token prompt
    for prompt in ('simplified', 'composed'):
        descriptor.model.inference_prompt = prompt
        mAP, cmc = compute_cmc_map(*extract(descriptor.model, loader), use_cosine=True)
        results[prompt] = {'mAP': float(mAP), **{k: float(v) for k, v in cmc.items()}}
        print(f"[{prompt}] mAP: {mAP:.4f} | " + " | ".join(f"{k}: {v:.4f}" for k, v in cmc.items()))

    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.save_results}")


if __name__ == "__main__":
    main()
