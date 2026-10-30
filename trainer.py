import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import json

from utils.dataset import VehicleReIDDataset
from utils.losses import TripletLoss, CenterLoss
from utils.sampler import RandomIdentitySampler
from utils.compute_metrics import compute_mAP
from modules.feature_extraction import OSNet, ResNetIBN, create_vehicle_descriptor
from modules.text_alignment import TextAlignHead, build_caption_bank, text_retrieval_map

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(
    data_dir,
    model_type='osnet',
    num_epochs=120,
    batch_size=32,
    learning_rate=0.0003,
    weight_decay=5e-4,
    step_size=40,
    gamma=0.1,
    margin=0.3,
    save_dir='checkpoints',
    text_labels=None,
    lambda_text=1.0
    ):
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), antialias=True),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        # Hue jitter would change the colors the text captions describe
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0 if text_labels else 0.1),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), antialias=True),
        transforms.CenterCrop((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets and dataloaders
    # Images of both train.txt and val.txt live in <data_dir>/images/train
    image_dir = os.path.join(data_dir, 'images', 'train')
    train_dataset = VehicleReIDDataset(data_dir, image_dir, split='train', transform=train_transform,
                                       text_labels=text_labels)
    val_dataset = VehicleReIDDataset(data_dir, image_dir, split='val', transform=val_transform,
                                     text_labels=text_labels)
    
    # P x K batches (batch_size / 2 vehicles x 2 images each): with plain shuffling almost no
    # batch contains two images of the same vehicle, so the triplet loss is ~0.
    train_sampler = RandomIdentitySampler(train_dataset.labels, batch_size, num_instances=2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                             drop_last=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of identities: {train_dataset.num_classes}")
    
    # Create model
    if model_type == 'osnet':
        model = OSNet(num_classes=train_dataset.num_classes, feature_dim=512)
    elif model_type == 'resnet_ibn':
        model = ResNetIBN(num_classes=train_dataset.num_classes, feature_dim=2048)
    elif model_type == 'efficientnet':
        descriptor = create_vehicle_descriptor('efficientnet', num_classes=train_dataset.num_classes)
        model = descriptor.model
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    # Optional image-text alignment with the frozen CLIP text encoder
    text_head, caption_to_idx, caption_bank, clip_meta = None, None, None, {}
    val_caption_to_idx, val_caption_bank = None, None
    if text_labels:
        if model_type == 'efficientnet':
            raise ValueError('--text_labels is supported for osnet and resnet_ibn only')
        with open(text_labels, 'r') as f:
            meta = json.load(f)['meta']
        clip_meta = {'clip_model': meta['model'], 'clip_pretrained': meta['pretrained']}
        # The loss contrasts against the whole training bank, so val captions stay out of it
        caption_to_idx, caption_bank = build_caption_bank(train_dataset.captions, meta['model'], meta['pretrained'])
        val_caption_to_idx, val_caption_bank = build_caption_bank(val_dataset.captions, meta['model'],
                                                                  meta['pretrained'])
        caption_bank = caption_bank.to(device)
        text_head = TextAlignHead(model.feature_dim, caption_bank.shape[1]).to(device)
        print(f"Text alignment: {len(caption_to_idx)} train / {len(val_caption_to_idx)} val captions, "
              f"lambda_text={lambda_text}")
    
    # Define losses
    criterion_ce = nn.CrossEntropyLoss()
    criterion_triplet = TripletLoss(margin=margin)
    criterion_center = CenterLoss(train_dataset.num_classes, model.feature_dim, use_gpu=torch.cuda.is_available())
    
    # Define optimizer
    params = list(model.parameters()) + (list(text_head.parameters()) if text_head else [])
    optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    optimizer_center = optim.SGD(criterion_center.parameters(), lr=0.5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    # Training loop
    best_mAP = 0.0
    best_text_mAP = 0.0
    train_losses = []
    val_mAPs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_ce_loss = 0.0
        running_triplet_loss = 0.0
        running_center_loss = 0.0
        running_text_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            
            # Forward pass
            if model_type in ['osnet', 'resnet_ibn']:
                features, logits = model(images)
            else:  # efficientnet
                features = model(images)
                logits = features  # For EfficientNet, features are the final output
            
            # Compute losses
            ce_loss = criterion_ce(logits, labels)
            triplet_loss = criterion_triplet(features, labels)
            center_loss = criterion_center(features, labels)
            
            # Total loss
            loss = ce_loss + triplet_loss + 0.0005 * center_loss
            text_loss = torch.zeros((), device=device)
            if text_head is not None:
                caption_ids = torch.tensor([caption_to_idx.get(c, -1) for c in batch['caption']], device=device)
                text_loss = text_head.loss(features, caption_bank, caption_ids)
                loss = loss + lambda_text * text_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update center loss
            for param in criterion_center.parameters():
                param.grad.data *= (1. / 0.0005)
            optimizer_center.step()
            
            # Update running losses
            running_loss += loss.item()
            running_ce_loss += ce_loss.item()
            running_triplet_loss += triplet_loss.item()
            running_center_loss += center_loss.item()
            running_text_loss += text_loss.item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'CE': f'{ce_loss.item():.4f}',
                'Triplet': f'{triplet_loss.item():.4f}',
                'Center': f'{center_loss.item():.4f}',
                **({'Text': f'{text_loss.item():.4f}'} if text_head is not None else {})
            })
        
        # Calculate average training losses
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation phase
        if (epoch + 1) % 5 == 0:  # Validate every 5 epochs
            mAP = evaluate(model, val_loader, model_type)
            val_mAPs.append(mAP)
            
            print(f'Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Val mAP = {mAP:.4f}')
            if text_head is not None:
                text_mAP, r1, r5 = evaluate_text(model, text_head, val_loader, val_caption_to_idx, val_caption_bank)
                print(f'Val text->image mAP = {text_mAP:.4f}, R@1 = {r1:.4f}, R@5 = {r5:.4f}')
                # The image re-ID best and the text retrieval best are usually different epochs
                if text_mAP > best_text_mAP:
                    best_text_mAP = text_mAP
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'best_mAP': mAP,
                        'best_text_mAP': best_text_mAP,
                        'model_type': model_type,
                        **text_checkpoint(text_head, clip_meta)
                    }, os.path.join(save_dir, f'best_{model_type}_text_model.pth'))
                    print(f'New best text model saved with text mAP: {best_text_mAP:.4f}')

            # Save best model
            if mAP > best_mAP:
                best_mAP = mAP
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_mAP': best_mAP,
                    'model_type': model_type,
                    **text_checkpoint(text_head, clip_meta)
                }, os.path.join(save_dir, f'best_{model_type}_model.pth'))
                print(f'New best model saved with mAP: {best_mAP:.4f}')
        
        scheduler.step()
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mAP': mAP if (epoch + 1) % 5 == 0 else 0.0,
                'model_type': model_type,
                **text_checkpoint(text_head, clip_meta)
            }, os.path.join(save_dir, f'{model_type}_epoch_{epoch+1}.pth'))
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    if val_mAPs:
        plt.subplot(1, 2, 2)
        plt.plot(range(4, len(val_mAPs)*5, 5), val_mAPs)
        plt.title('Validation mAP')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_type}_training_curves.png'))
    plt.show()
    
    print(f'Training completed. Best mAP: {best_mAP:.4f}')
    if text_head is not None:
        print(f'Best text->image mAP: {best_text_mAP:.4f} (best_{model_type}_text_model.pth)')

def text_checkpoint(text_head, clip_meta):
    """Extra checkpoint entries for the text alignment head (empty when it is not trained)"""
    if text_head is None:
        return {}
    return {'text_head_state_dict': text_head.state_dict(), **clip_meta}

def evaluate_text(model, text_head, data_loader, caption_to_idx, caption_bank):
    """Text -> image retrieval on the validation set: each caption should rank its vehicles first"""
    model.eval()
    embeds, captions = [], []
    with torch.no_grad():
        for batch in data_loader:
            outputs = model(batch['image'].to(device))
            features = outputs[0] if isinstance(outputs, tuple) else outputs
            embeds.append(text_head(features).cpu())
            captions.extend(batch['caption'])
    bank = caption_bank.cpu()
    return text_retrieval_map(torch.cat(embeds), captions, {c: bank[caption_to_idx[c]] for c in set(captions)})

def evaluate(model, data_loader, model_type):
    """Evaluate model on validation set"""
    model.eval()
    
    all_features = []
    all_labels = []
    all_camera_ids = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Extracting features'):
            images = batch['image'].to(device)
            labels = batch['label']
            camera_ids = batch['camera_id']
            
            outputs = model(images)
            if isinstance(outputs, tuple):
                features = outputs[0]
            else:
                features = outputs
            
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            
            all_features.append(features.cpu())
            all_labels.extend(labels.numpy())
            all_camera_ids.extend(camera_ids.numpy())
    
    all_features = torch.cat(all_features, dim=0)
    all_labels = np.array(all_labels)
    all_camera_ids = np.array(all_camera_ids)
    
    # Compute mAP
    mAP = compute_mAP(all_features, all_labels, all_camera_ids, use_cosine=True)
    
    return mAP


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Vehicle Re-ID Model')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to dataset directory')
    parser.add_argument('--model_type', type=str, default='osnet',
                       choices=['osnet', 'resnet_ibn', 'efficientnet'],
                       help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.0003,
                       help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save models')
    parser.add_argument('--text_labels', type=str, default=None,
                       help='text_labels.json from CLIP_Query.generate_text_labels; enables CLIP text alignment')
    parser.add_argument('--lambda_text', type=float, default=1.0,
                       help='Weight of the image-text alignment loss')
    
    args = parser.parse_args()
    
    
    # Train model
    train(
        data_dir=args.data_dir,
        model_type=args.model_type,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_dir=args.save_dir,
        text_labels=args.text_labels,
        lambda_text=args.lambda_text
    )
