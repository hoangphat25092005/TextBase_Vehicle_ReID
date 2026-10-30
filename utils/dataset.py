import cv2
import json
from pathlib import Path
from torch.utils.data import Dataset

class VehicleReIDDataset(Dataset):
    """
    Dataset class for Vehicle Re-ID training
    """
    
    def __init__(self, data_dir, image_dir, split='train', transform=None, text_labels=None,
                 skip_flagged=True):
        self.data_dir = Path(data_dir)
        self.image_dir = Path(image_dir)
        self.split = split
        self.transform = transform
        
        # Load image list
        split_file = self.data_dir / f'{split}.txt'
        with open(split_file, 'r') as f:
            self.image_list = [line.strip() for line in f.readlines()]
        
        # Extract vehicle keys (vehicle_type, vehicle_id) and camera IDs
        self.vehicle_keys = []  # (vehicle_type, vehicle_id)
        self.camera_ids = []
        
        for img_name in self.image_list:
            # Parse format: vehicleType_vehicleID_cameraID.jpg
            parts = img_name.split('_')
            vehicle_type = parts[0]
            vehicle_id = int(parts[1])
            camera_id = int(parts[2][3:-4])  # Remove 'cam' prefix
            self.vehicle_keys.append((vehicle_type, vehicle_id))
            self.camera_ids.append(camera_id)
        
        # Create label mapping using (vehicle_type, vehicle_id) as key
        unique_keys = sorted(set(self.vehicle_keys))
        self.id_to_label = {key: idx for idx, key in enumerate(unique_keys)}
        self.labels = [self.id_to_label[key] for key in self.vehicle_keys]
        self.num_classes = len(unique_keys)
        
        # Optional captions from CLIP_Query/generate_text_labels.py, keyed by the filename prefix
        # ('car_0088'; a few files are not zero-padded, e.g. 'motorcycle_153').
        # Vehicles flagged needs_review (cameras disagree on the main color, low confidence) get an
        # empty caption when skip_flagged is set, so the text loss ignores them.
        self.captions = None
        if text_labels:
            with open(text_labels, 'r') as f:
                vehicles = json.load(f)['vehicles']
            vehicles_of = [vehicles['_'.join(name.split('_')[:2])] for name in self.image_list]
            self.captions = ['' if skip_flagged and v['needs_review'] else v['caption'] for v in vehicles_of]
            skipped = sum(not c for c in self.captions)
            if skipped:
                print(f"[Dataset] {split}: {skipped}/{len(self.captions)} images have no caption (flagged for review)")
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = self.image_dir / img_name
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        vehicle_type, vehicle_id = self.vehicle_keys[idx]
        camera_id = self.camera_ids[idx]
        
        return {
            'image': image,
            'label': label,
            'vehicle_type': vehicle_type,
            'vehicle_id': vehicle_id,
            'camera_id': camera_id,
            'img_name': img_name,
            **({'caption': self.captions[idx]} if self.captions is not None else {})
        }