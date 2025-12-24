# BiTemporal S1+S2 Dataset Configuration

This configuration supports bi-temporal multi-sensor satellite imagery for self-supervised pretraining and semantic segmentation fine-tuning.

## Dataset Specifications

- **Image Size**: 256×256 pixels at 10m/px resolution
- **Temporal**: 2 dates (bi-temporal)
- **Sensors**:
  - Sentinel-1 (S1): 2 bands (VV, VH)
  - Sentinel-2 (S2): 13 bands (all spectral bands)
- **Task**: Self-supervised pretraining → Fine-tuning for segmentation

## Data Structure

Organize your data as follows:

```
data/BiTemporalS1S2/
├── train_list.json          # ["sample_001", "sample_002", ...]
├── val_list.json
├── test_list.json
├── S1/
│   ├── sample_001.npy       # Shape: (2, 2, 256, 256)
│   ├── sample_002.npy       # [time, channels, height, width]
│   └── ...
├── S2/
│   ├── sample_001.npy       # Shape: (2, 13, 256, 256)
│   ├── sample_002.npy
│   └── ...
├── dates/
│   ├── sample_001.json      # {"s1": [20200101, 20200615], "s2": [20200105, 20200620]}
│   ├── sample_002.json
│   └── ...
└── labels/                  # Optional, for fine-tuning
    ├── sample_001.npy       # Shape: (256, 256) - class indices
    ├── sample_002.npy
    └── ...
```

### Data Format Details

**Satellite Data (.npy files)**:
- Shape: `(2, C, 256, 256)` where:
  - 2 = number of temporal observations
  - C = number of channels (2 for S1, 13 for S2)
  - 256×256 = spatial dimensions
- Data type: float32
- Values: Raw or TOA reflectance values

**Sample Lists (.json files)**:
```json
["sample_001", "sample_002", "sample_003"]
```

**Dates (.json files)**:
```json
{
  "s1": [20200101, 20200615],
  "s2": [20200105, 20200620]
}
```
- Format: YYYYMMDD as integers
- Must have 2 dates per sensor

**Labels (.npy files)** - Optional:
- Shape: `(256, 256)`
- Data type: int64
- Values: Class indices [0, num_classes-1]

## Usage

### 1. Self-Supervised Pretraining

Train the model without labels using JEPA loss:

```bash
python src/train.py exp=BiTemporal_AnySat
```

**Key parameters** in `configs/exp/BiTemporal_AnySat.yaml`:
- Model: JEPA (self-supervised)
- Loss: MSE (reconstruction) + MIL-NCE (contrastive)
- Modalities: S1 + S2
- Learning rate: 5e-5
- Max epochs: 300

**What to monitor**:
- `val/loss` (should decrease)
- `val/mse_loss` (reconstruction quality)
- `val/contrastive_loss` (multi-modal alignment)

### 2. Fine-Tuning for Segmentation

After pretraining, fine-tune for dense semantic segmentation:

```bash
python src/train.py exp=BiTemporal_AnySat_FT_SemSeg
```

**Key parameters** in `configs/exp/BiTemporal_AnySat_FT_SemSeg.yaml`:
- Model: SemSeg (dense prediction)
- Pretrained weights: Loaded from pretraining checkpoint
- Modalities: S1 + S2
- Learning rate: 1e-4
- Max epochs: 100

**Before running**, update:
- `dataset.num_classes`: Number of segmentation classes
- `model.network.instance.path`: Path to pretrained checkpoint

**What to monitor**:
- `val/mIoU` (mean Intersection over Union - should increase)
- `val/loss` (should decrease)
- `val/acc` (pixel accuracy)

## Configuration Files

### Dataset Config
[`configs/dataset/BiTemporalS1S2.yaml`](configs/dataset/BiTemporalS1S2.yaml)
- Defines dataset parameters
- Sets image dimensions (256×256)
- Configures data paths and splits

### Experiment Configs
1. [`configs/exp/BiTemporal_AnySat.yaml`](configs/exp/BiTemporal_AnySat.yaml) - Pretraining
2. [`configs/exp/BiTemporal_AnySat_FT_SemSeg.yaml`](configs/exp/BiTemporal_AnySat_FT_SemSeg.yaml) - Fine-tuning

### Python Dataset Class
[`src/data/BiTemporalS1S2.py`](src/data/BiTemporalS1S2.py)
- Handles data loading
- Computes normalization statistics
- Provides collate function for batching

## Customization

### Update Dataset Size
Edit `configs/dataset/BiTemporalS1S2.yaml`:
```yaml
true_len: 10000        # Your training samples
true_len_val: 2000     # Your validation samples
```

### Update Number of Classes (for fine-tuning)
Edit `configs/exp/BiTemporal_AnySat_FT_SemSeg.yaml`:
```yaml
dataset:
  num_classes: 10      # Your number of classes
```

### Adjust Batch Size
Edit `configs/dataset/BiTemporalS1S2.yaml`:
```yaml
global_batch_size: 4   # Adjust based on GPU memory
```

### Change Learning Rate
Override from command line:
```bash
python src/train.py exp=BiTemporal_AnySat model.optimizer.lr=1e-4
```

## Data Preparation Tips

1. **Normalization**: Statistics are computed automatically on first run
2. **Missing Dates**: If dates.json is missing, defaults to [0, 180] days
3. **Image Format**: Ensure data is in (T, C, H, W) format
4. **Memory**: 256×256 images with 2 sensors × 2 dates = ~5MB per sample

## Troubleshooting

**Error: "Sample list file not found"**
- Create train_list.json, val_list.json, test_list.json with sample IDs

**Error: "Data file not found"**
- Check that S1/ and S2/ folders contain .npy files matching sample IDs

**Error: "Expected 4D data"**
- Verify data shape is (2, C, 256, 256)

**Low memory**
- Reduce batch size in dataset config
- Use gradient accumulation

**Model not improving**
- Check val/loss is decreasing
- Verify data normalization
- Ensure labels are in correct format (for fine-tuning)
