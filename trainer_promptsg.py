import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.dataset import VehicleReIDDataset
from utils.losses import TripletLoss, SupConLoss
from utils.sampler import RandomIdentitySampler
from modules.promptsg import PromptSG, build_transforms
from trainer import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(
    data_dir,
    num_epochs=60,
    batch_size=64,
    num_instances=2,
    visual_lr=5e-6,
    head_lr=5e-5,
    weight_decay=1e-4,
    step_size=20,
    gamma=0.1,
    margin=0.3,
    supcon_weight=0.5,
    model_name='ViT-B-16',
    amp=True,
    save_dir='checkpoints_promptsg'
    ):
    os.makedirs(save_dir, exist_ok=True)

    # Images of both train.txt and val.txt live in <data_dir>/images/train
    image_dir = os.path.join(data_dir, 'images', 'train')
    train_dataset = VehicleReIDDataset(data_dir, image_dir, split='train', transform=build_transforms(train=True))
    val_dataset = VehicleReIDDataset(data_dir, image_dir, split='val', transform=build_transforms())

    # P x K batches; each vehicle only has a cam1 and a cam2 image, hence 2 instances per ID
    # (the paper uses 16 IDs x 4 images for pedestrians)
    train_sampler = RandomIdentitySampler(train_dataset.labels, batch_size, num_instances=num_instances)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              drop_last=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of identities: {train_dataset.num_classes}")

    model = PromptSG(num_classes=train_dataset.num_classes, model_name=model_name).to(device)

    criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_triplet = TripletLoss(margin=margin)
    criterion_supcon = SupConLoss()

    # Pre-trained CLIP visual encoder is fine-tuned gently, the randomly initialised modules faster;
    # the text encoder is frozen and not handed to the optimizer
    visual_params = [p for p in model.visual.parameters() if p.requires_grad]
    visual_ids = {id(p) for p in visual_params}
    head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in visual_ids]
    optimizer = optim.Adam([
        {'params': visual_params, 'lr': visual_lr},
        {'params': head_params, 'lr': head_lr},
    ], weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # fp16 forward pass halves memory and roughly doubles speed on GPUs such as Colab's T4
    amp = amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)

    best_mAP = 0.0
    train_losses = []
    val_mAPs = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                outputs = model(images)

            # Losses in fp32: the contrastive logits are scaled by CLIP's logit scale (100)
            # L = L_Triplet + L_ID + lambda * L_SupCon  (Eq. 10)
            id_loss = sum(criterion_ce(logits.float(), labels) for logits in outputs['logits'])
            triplet_loss = sum(criterion_triplet(feat.float(), labels) for feat in outputs['triplet_features'])
            supcon_loss = criterion_supcon(outputs['image_embed'].float(), outputs['text_embed'].float(), labels,
                                           model.logit_scale.exp())
            loss = id_loss + triplet_loss + supcon_weight * supcon_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'ID': f'{id_loss.item():.4f}',
                'Triplet': f'{triplet_loss.item():.4f}',
                'SupCon': f'{supcon_loss.item():.4f}'
            })

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        if (epoch + 1) % 5 == 0:
            mAP = evaluate(model, val_loader, 'promptsg')
            val_mAPs.append(mAP)
            print(f'Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Val mAP = {mAP:.4f}')

            if mAP > best_mAP:
                best_mAP = mAP
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_mAP': best_mAP,
                    'model_type': 'promptsg',
                    'model_name': model_name,
                    'num_classes': train_dataset.num_classes
                }, os.path.join(save_dir, 'best_promptsg_model.pth'))
                print(f'New best model saved with mAP: {best_mAP:.4f}')

        scheduler.step()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    if val_mAPs:
        plt.subplot(1, 2, 2)
        plt.plot(range(5, len(val_mAPs) * 5 + 1, 5), val_mAPs)
        plt.title('Validation mAP')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'promptsg_training_curves.png'))
    plt.show()

    print(f'Training completed. Best mAP: {best_mAP:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PromptSG Vehicle Re-ID Model')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=60,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--visual_lr', type=float, default=5e-6,
                        help='Learning rate of the CLIP visual encoder')
    parser.add_argument('--head_lr', type=float, default=5e-5,
                        help='Learning rate of the inversion network, interaction module and heads')
    parser.add_argument('--supcon_weight', type=float, default=0.5,
                        help='Weight lambda of the supervised contrastive loss')
    parser.add_argument('--no_amp', action='store_true',
                        help='Train in fp32 instead of mixed precision')
    parser.add_argument('--save_dir', type=str, default='checkpoints_promptsg',
                        help='Directory to save models')

    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        visual_lr=args.visual_lr,
        head_lr=args.head_lr,
        supcon_weight=args.supcon_weight,
        amp=not args.no_amp,
        save_dir=args.save_dir
    )
