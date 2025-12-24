from torch.utils.data import Dataset
import numpy as np
import os
import torch
import json
from datetime import datetime
import rasterio


def collate_fn(batch):
    """
    Collate function for bi-temporal S1+S2 data.
    Args:
        batch (list): list of dictionaries with keys corresponding to modalities
    Returns:
        dict: dictionary with batched tensors
    """
    keys = list(batch[0].keys())
    output = {}
    
    # Handle temporal modalities (s1, s2)
    for key in ["s1", "s2"]:
        if key in keys:
            idx = [x[key] for x in batch]
            # Stack temporal dimension (should be 2 for bi-temporal)
            max_size_0 = max(tensor.size(0) for tensor in idx)
            stacked_tensor = torch.stack([
                torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, 0, 0, max_size_0 - tensor.size(0)))
                for tensor in idx
            ], dim=0)
            output[key] = stacked_tensor.float()
            keys.remove(key)
            
            # Handle dates
            date_key = '_'.join([key, "dates"])
            if date_key in keys:
                idx = [x[date_key] for x in batch]
                max_size_0 = max(tensor.size(0) for tensor in idx)
                stacked_tensor = torch.stack([
                    torch.nn.functional.pad(tensor, (0, max_size_0 - tensor.size(0)))
                    for tensor in idx
                ], dim=0)
                output[date_key] = stacked_tensor.long()
                keys.remove(date_key)
    
    # Handle label if present
    if 'label' in keys:
        output['label'] = torch.stack([x['label'] for x in batch]).float()
        keys.remove('label')
    
    # Handle name if present
    if 'name' in keys:
        output['name'] = [x['name'] for x in batch]
        keys.remove('name')
    
    # Handle any remaining keys
    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            output[key] = torch.stack([x[key] for x in batch]).float()
    
    return output


def prepare_dates(dates_list, reference_date):
    """
    Convert dates to day numbers relative to reference date.
    Args:
        dates_list (list): list of date strings in format YYYYMMDD or datetime objects
        reference_date (datetime): reference date
    Returns:
        torch.Tensor: day numbers
    """
    day_numbers = []
    for date_item in dates_list:
        if isinstance(date_item, str):
            date_obj = datetime.strptime(str(date_item), '%Y%m%d')
        elif isinstance(date_item, int):
            date_obj = datetime.strptime(str(date_item), '%Y%m%d')
        else:
            date_obj = date_item
        days = (date_obj - reference_date).days
        day_numbers.append(days)
    return torch.tensor(day_numbers)


class BiTemporalS1S2(Dataset):
    """
    Bi-temporal Sentinel-1 + Sentinel-2 dataset for self-supervised pretraining.
    
    Expected data structure:
    path/
      ├── train_list.json      # List of sample IDs: ["sample_001", "sample_002", ...]
      ├── val_list.json
      ├── test_list.json
      ├── S1/
      │   └── {sample_id}.tif  # Shape: (4, 256, 256) - 2 dates × 2 bands (VV, VH)
      ├── S2/
      │   └── {sample_id}.tif  # Shape: (26, 256, 256) - 2 dates × 13 bands
      ├── dates/
      │   └── {sample_id}.json  # {"s1": [20200101, 20200615], "s2": [20200105, 20200620]}
      └── labels/  (optional, for segmentation fine-tuning)
          └── {sample_id}.tif  # Shape: (1, 256, 256) - segmentation mask
    
    Args:
        path (str): path to the dataset
        modalities (list): list of modalities to use (e.g., ['s1', 's2'])
        transform: transform to apply to the data
        split (str): train/val/test
        partition (float): proportion of dataset to use
        norm_path (str): path for normalization stats
        temporal_dropout (int): not used for bi-temporal (always 2 dates)
        reference_date (str): reference date for temporal encoding
    """
    
    def __init__(
        self,
        path,
        modalities,
        transform,
        split: str = "train",
        partition: float = 1.0,
        norm_path=None,
        temporal_dropout=0,
        reference_date="2020-01-01",
    ):
        self.path = path
        self.modalities = modalities
        self.transform = transform
        self.split = split
        self.partition = partition
        self.temporal_dropout = temporal_dropout
        self.reference_date = datetime.strptime(reference_date, '%Y-%m-%d')
        
        # Load sample list
        list_file = os.path.join(path, f'{split}_list.json')
        if not os.path.exists(list_file):
            raise FileNotFoundError(f"Sample list file not found: {list_file}")
        
        with open(list_file, 'r') as f:
            self.sample_ids = json.load(f)
        
        print(f"Loaded {len(self.sample_ids)} samples for {split} split")
        
        # Apply partition
        if partition < 1.0:
            n_samples = int(len(self.sample_ids) * partition)
            self.sample_ids = self.sample_ids[:n_samples]
            print(f"Using partition {partition}: {n_samples} samples")
        
        self.collate_fn = collate_fn
        self.norm = {}
        
        # Load or compute normalization stats
        if norm_path is not None:
            for modality in self.modalities:
                norm_file = os.path.join(norm_path, f"NORM_{modality}_patch.json")
                if os.path.exists(norm_file):
                    normvals = json.load(open(norm_file))
                    self.norm[modality] = (
                        torch.tensor(normvals['mean']).float(),
                        torch.tensor(normvals['std']).float(),
                    )
                    print(f"Loaded normalization for {modality}")
                else:
                    print(f"Computing normalization for {modality}...")
                    self.compute_norm_vals(norm_path, modality)
    
    def compute_norm_vals(self, folder, modality):
        """Compute mean and std for normalization."""
        means = []
        stds = []
        
        n_samples = min(len(self.sample_ids), 1000)
        print(f"Computing normalization stats from {n_samples} samples...")
        
        for i in range(n_samples):
            try:
                data = self._load_modality(self.sample_ids[i], modality)
                # Shape: (T, C, H, W) - compute stats over time, height, width
                means.append(data.float().mean(dim=(0, 2, 3)).numpy())
                stds.append(data.float().std(dim=(0, 2, 3)).numpy())
            except Exception as e:
                print(f"Error loading sample {self.sample_ids[i]}: {e}")
                continue
        
        if len(means) == 0:
            raise ValueError(f"Could not compute normalization stats for {modality}")
        
        mean = np.stack(means).mean(axis=0).astype(float)
        std = np.stack(stds).mean(axis=0).astype(float)
        
        norm_vals = dict(mean=list(mean), std=list(std))
        
        os.makedirs(folder, exist_ok=True)
        norm_file = os.path.join(folder, f"NORM_{modality}_patch.json")
        with open(norm_file, "w") as file:
            json.dump(norm_vals, indent=4, fp=file)
        
        self.norm[modality] = (torch.tensor(mean).float(), torch.tensor(std).float())
        print(f"Saved normalization stats to {norm_file}")
    
    def _load_modality(self, sample_id, modality):
        """Load data for a specific modality from GeoTIFF or NumPy."""
        # Try .tif first, then .npy
        tif_file = os.path.join(self.path, modality.upper(), f'{sample_id}.tif')
        npy_file = os.path.join(self.path, modality.upper(), f'{sample_id}.npy')
        
        if os.path.exists(tif_file):
            # Load from GeoTIFF
            with rasterio.open(tif_file) as src:
                data = src.read()  # Shape: (bands, H, W)
                data = torch.from_numpy(data).float()
            
            # Reshape from (T*C, H, W) to (T, C, H, W)
            if modality == 's1':
                # S1: 4 bands = 2 dates × 2 channels (VV, VH)
                # Reshape (4, H, W) → (2, 2, H, W)
                if data.shape[0] != 4:
                    raise ValueError(f"Expected 4 bands for S1 (2 dates × 2 channels), got {data.shape[0]} for {tif_file}")
                data = data.reshape(2, 2, data.shape[1], data.shape[2])
            elif modality == 's2':
                # S2: 26 bands = 2 dates × 13 channels
                # Reshape (26, H, W) → (2, 13, H, W)
                if data.shape[0] != 26:
                    raise ValueError(f"Expected 26 bands for S2 (2 dates × 13 channels), got {data.shape[0]} for {tif_file}")
                data = data.reshape(2, 13, data.shape[1], data.shape[2])
            else:
                # Generic: assume first dimension is T*C, reshape accordingly
                print(f"Warning: Unknown modality {modality}, assuming bi-temporal stacking")
                n_bands = data.shape[0]
                n_channels = n_bands // 2
                data = data.reshape(2, n_channels, data.shape[1], data.shape[2])
                
        elif os.path.exists(npy_file):
            # Load from NumPy - already in correct shape (T*C, H, W) or (T, C, H, W)
            data = torch.from_numpy(np.load(npy_file)).float()
            
            # Check if needs reshaping
            if data.ndim == 3:
                # Shape: (T*C, H, W) - need to reshape
                if modality == 's1':
                    if data.shape[0] != 4:
                        raise ValueError(f"Expected 4 bands for S1, got {data.shape[0]} for {npy_file}")
                    data = data.reshape(2, 2, data.shape[1], data.shape[2])
                elif modality == 's2':
                    if data.shape[0] != 26:
                        raise ValueError(f"Expected 26 bands for S2, got {data.shape[0]} for {npy_file}")
                    data = data.reshape(2, 13, data.shape[1], data.shape[2])
                else:
                    n_bands = data.shape[0]
                    n_channels = n_bands // 2
                    data = data.reshape(2, n_channels, data.shape[1], data.shape[2])
            # If already 4D, assume it's already in correct shape (T, C, H, W)
        else:
            raise FileNotFoundError(f"Data file not found: {tif_file} or {npy_file}")
        
        # Validate final shape
        if data.ndim != 4:
            raise ValueError(f"Expected 4D data (T, C, H, W), got shape {data.shape}")
        if data.shape[0] != 2:
            print(f"Warning: Expected 2 temporal observations, got {data.shape[0]}")
        
        return data
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        Returns:
            dict: Dictionary with modality data, dates, and optionally labels
        """
        sample_id = self.sample_ids[idx]
        output = {'name': sample_id}
        
        # Load dates
        dates_file = os.path.join(self.path, 'dates', f'{sample_id}.json')
        dates_dict = {}
        if os.path.exists(dates_file):
            with open(dates_file, 'r') as f:
                dates_dict = json.load(f)
        
        # Load each modality
        for modality in self.modalities:
            # Load data
            data = self._load_modality(sample_id, modality)
            
            # Normalize if stats available
            if modality in self.norm:
                mean, std = self.norm[modality]
                # Normalize each channel: (T, C, H, W)
                for c in range(data.shape[1]):
                    data[:, c] = (data[:, c] - mean[c]) / (std[c] + 1e-8)
            
            output[modality] = data
            
            # Add dates
            if dates_dict:
                dates = dates_dict.get(modality, dates_dict.get(modality.upper(), []))
                if dates:
                    output[f'{modality}_dates'] = prepare_dates(dates, self.reference_date)
                else:
                    # Create dummy dates if not provided (0 and 180 days from reference)
                    output[f'{modality}_dates'] = torch.tensor([0, 180])
            else:
                # Default dates: t0 and t0+180 days
                output[f'{modality}_dates'] = torch.tensor([0, 180])
        
        # Load label if available (for fine-tuning)
        label_tif = os.path.join(self.path, 'labels', f'{sample_id}.tif')
        label_npy = os.path.join(self.path, 'labels', f'{sample_id}.npy')
        
        if os.path.exists(label_tif):
            with rasterio.open(label_tif) as src:
                label = src.read(1)  # Read first band
                label = torch.from_numpy(label).long()
            output['label'] = label
        elif os.path.exists(label_npy):
            label = torch.from_numpy(np.load(label_npy)).long()
            output['label'] = label
        else:
            # Dummy label for self-supervised pretraining
            output['label'] = torch.zeros(1)
        
        # Apply transforms
        if self.transform is not None:
            output = self.transform(output)
        
        return output
