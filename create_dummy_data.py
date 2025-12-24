"""
Create dummy bi-temporal flood dataset for testing.
This script creates example .tif files with realistic flood scenarios.
"""
import numpy as np
import rasterio
from rasterio.transform import from_origin
import json
import os

# Configuration
HEIGHT = 256
WIDTH = 256
N_SAMPLES = 5  # Create 5 example samples

def create_s1_data(sample_id, output_dir, has_flood=False):
    """
    Create dummy Sentinel-1 data (VV, VH polarizations).
    Saves pre and post images separately.
    Pre: (2, 256, 256) - VV, VH for date 1
    Post: (2, 256, 256) - VV, VH for date 2
    """
    # Pre-flood (Date 1): Normal backscatter values
    # VV: -15 to -5 dB (water is darker, land is brighter)
    # VH: -25 to -15 dB (cross-pol is generally lower)
    
    np.random.seed(int(sample_id.split('_')[1]))
    
    # Date 1 (Pre-flood): Mixed land and water
    date1_vv = np.random.uniform(-15, -5, (HEIGHT, WIDTH)).astype(np.float32)
    date1_vh = np.random.uniform(-25, -15, (HEIGHT, WIDTH)).astype(np.float32)
    
    # Add some water bodies (rivers) - very low backscatter
    water_mask_pre = np.zeros((HEIGHT, WIDTH), dtype=bool)
    water_mask_pre[100:120, :] = True  # Horizontal river
    date1_vv[water_mask_pre] = np.random.uniform(-25, -20, water_mask_pre.sum())
    date1_vh[water_mask_pre] = np.random.uniform(-35, -30, water_mask_pre.sum())
    
    if has_flood:
        # Date 2 (Post-flood): More water area (flooded regions)
        date2_vv = date1_vv.copy()
        date2_vh = date1_vh.copy()
        
        # Flood extent - water spreads to surrounding areas
        flood_mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
        flood_mask[80:140, 50:200] = True  # Larger flood extent
        flood_mask = flood_mask & ~water_mask_pre  # New flooded areas only
        
        # Flooded areas have very low backscatter
        date2_vv[flood_mask] = np.random.uniform(-23, -18, flood_mask.sum())
        date2_vh[flood_mask] = np.random.uniform(-33, -28, flood_mask.sum())
    else:
        # Date 2 (No flood): Similar to Date 1 with small variations
        date2_vv = date1_vv + np.random.normal(0, 0.5, (HEIGHT, WIDTH)).astype(np.float32)
        date2_vh = date1_vh + np.random.normal(0, 0.5, (HEIGHT, WIDTH)).astype(np.float32)
    
    # Stack pre and post: (2, H, W) each
    data_pre = np.stack([date1_vv, date1_vh], axis=0)
    data_post = np.stack([date2_vv, date2_vh], axis=0)
    
    # Save as GeoTIFF
    transform = from_origin(0, HEIGHT * 10, 10, 10)  # 10m resolution
    
    # Save pre
    filename_pre = os.path.join(output_dir, 'S1', 'pre', f'{sample_id}.tif')
    with rasterio.open(
        filename_pre, 'w',
        driver='GTiff',
        height=HEIGHT,
        width=WIDTH,
        count=2,
        dtype=data_pre.dtype,
        transform=transform
    ) as dst:
        dst.write(data_pre)
    
    # Save post
    filename_post = os.path.join(output_dir, 'S1', 'post', f'{sample_id}.tif')
    with rasterio.open(
        filename_post, 'w',
        driver='GTiff',
        height=HEIGHT,
        width=WIDTH,
        count=2,
        dtype=data_post.dtype,
        transform=transform
    ) as dst:
        dst.write(data_post)
    
    print(f"Created S1: {filename_pre} and {filename_post}")
    return flood_mask if has_flood else None


def create_s2_data(sample_id, output_dir, flood_mask=None):
    """
    Create dummy Sentinel-2 data (13 bands).
    Saves pre and post images separately.
    Pre: (13, 256, 256) - 13 bands for date 1
    Post: (13, 256, 256) - 13 bands for date 2
    """
    np.random.seed(int(sample_id.split('_')[1]) + 1000)
    
    def generate_s2_observation(is_flooded_area=None):
        """Generate one S2 observation (13 bands)."""
        bands = []
        
        # Visible bands (B2-B4): vegetation and water
        for _ in range(3):  # B2, B3, B4
            band = np.random.uniform(500, 2500, (HEIGHT, WIDTH)).astype(np.float32)
            if is_flooded_area is not None:
                # Water has low reflectance in visible
                band[is_flooded_area] = np.random.uniform(200, 800, is_flooded_area.sum())
            bands.append(band)
        
        # Red edge and NIR (B5-B8A): vegetation health
        for _ in range(5):  # B5, B6, B7, B8, B8A
            band = np.random.uniform(1000, 3500, (HEIGHT, WIDTH)).astype(np.float32)
            if is_flooded_area is not None:
                # Water has very low NIR reflectance
                band[is_flooded_area] = np.random.uniform(100, 500, is_flooded_area.sum())
            bands.append(band)
        
        # SWIR bands (B11, B12): water detection
        for _ in range(2):  # B11, B12
            band = np.random.uniform(1500, 3000, (HEIGHT, WIDTH)).astype(np.float32)
            if is_flooded_area is not None:
                # Water absorbs SWIR strongly
                band[is_flooded_area] = np.random.uniform(50, 300, is_flooded_area.sum())
            bands.append(band)
        
        # Additional bands (total 13)
        for _ in range(3):
            band = np.random.uniform(1000, 2500, (HEIGHT, WIDTH)).astype(np.float32)
            bands.append(band)
        
        return np.stack(bands, axis=0)
    
    # Date 1 (Pre-flood): Normal vegetation and water
    date1 = generate_s2_observation(is_flooded_area=None)
    
    # Date 2 (Post-flood): Flooded areas if applicable
    if flood_mask is not None:
        date2 = generate_s2_observation(is_flooded_area=flood_mask)
    else:
        date2 = generate_s2_observation(is_flooded_area=None)
    
    # Save as GeoTIFF
    transform = from_origin(0, HEIGHT * 10, 10, 10)
    
    # Save pre
    filename_pre = os.path.join(output_dir, 'S2', 'pre', f'{sample_id}.tif')
    with rasterio.open(
        filename_pre, 'w',
        driver='GTiff',
        height=HEIGHT,
        width=WIDTH,
        count=13,
        dtype=date1.dtype,
        transform=transform
    ) as dst:
        dst.write(date1)
    
    # Save post
    filename_post = os.path.join(output_dir, 'S2', 'post', f'{sample_id}.tif')
    with rasterio.open(
        filename_post, 'w',
        driver='GTiff',
        height=HEIGHT,
        width=WIDTH,
        count=13,
        dtype=date2.dtype,
        transform=transform
    ) as dst:
        dst.write(date2)
    
    print(f"Created S2: {filename_pre} and {filename_post}")


def create_label(sample_id, output_dir, flood_mask=None):
    """
    Create segmentation label.
    Classes:
    0 = No change
    1 = Flooded area
    """
    if flood_mask is not None:
        label = flood_mask.astype(np.uint8)
    else:
        label = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    
    # Save as GeoTIFF
    transform = from_origin(0, HEIGHT * 10, 10, 10)
    
    filename = os.path.join(output_dir, 'labels', f'{sample_id}.tif')
    with rasterio.open(
        filename, 'w',
        driver='GTiff',
        height=HEIGHT,
        width=WIDTH,
        count=1,
        dtype=label.dtype,
        transform=transform
    ) as dst:
        dst.write(label, 1)
    
    print(f"Created Label: {filename}")


def create_dates_file(sample_id, output_dir, has_flood=False):
    """
    Create dates JSON file.
    Pre-flood: Early season (e.g., May 1)
    Post-flood: Later season (e.g., July 15) if flood, or June 1 if no flood
    """
    if has_flood:
        # Flood scenario: May 1 (pre) → July 15 (post, during flood)
        s1_dates = [20200501, 20200715]
        s2_dates = [20200505, 20200720]  # S2 slightly offset
    else:
        # No flood: May 1 → June 1 (normal monitoring)
        s1_dates = [20200501, 20200601]
        s2_dates = [20200505, 20200605]
    
    dates_dict = {
        "s1": s1_dates,
        "s2": s2_dates
    }
    
    filename = os.path.join(output_dir, 'dates', f'{sample_id}.json')
    with open(filename, 'w') as f:
        json.dump(dates_dict, f, indent=2)
    
    print(f"Created Dates: {filename}")


def main():
    """Create the complete dummy dataset."""
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create necessary directories
    os.makedirs(os.path.join(base_dir, 'S1', 'pre'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'S1', 'post'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'S2', 'pre'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'S2', 'post'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'dates'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'labels'), exist_ok=True)
    
    # Create sample IDs
    train_samples = ['sample_001', 'sample_002', 'sample_003']
    val_samples = ['sample_004']
    test_samples = ['sample_005']
    
    all_samples = train_samples + val_samples + test_samples
    
    # Decide which samples have floods (50% flood, 50% no flood)
    flood_samples = ['sample_001', 'sample_003', 'sample_004']
    
    print("Creating dummy bi-temporal flood dataset...")
    print(f"Image size: {HEIGHT}x{WIDTH} pixels at 10m resolution")
    print(f"Total samples: {len(all_samples)}")
    print(f"Flood samples: {flood_samples}")
    print()
    
    # Create each sample
    for sample_id in all_samples:
        has_flood = sample_id in flood_samples
        print(f"\n{'='*60}")
        print(f"Creating {sample_id} ({'FLOOD' if has_flood else 'NO FLOOD'})")
        print('='*60)
        
        # Create S1 data
        flood_mask = create_s1_data(sample_id, base_dir, has_flood=has_flood)
        
        # Create S2 data
        create_s2_data(sample_id, base_dir, flood_mask=flood_mask)
        
        # Create label
        create_label(sample_id, base_dir, flood_mask=flood_mask)
        
        # Create dates
        create_dates_file(sample_id, base_dir, has_flood=has_flood)
    
    # Create split files
    splits = {
        'train_list.json': train_samples,
        'val_list.json': val_samples,
        'test_list.json': test_samples
    }
    
    print(f"\n{'='*60}")
    print("Creating split files...")
    print('='*60)
    
    for filename, samples in splits.items():
        filepath = os.path.join(base_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(samples, f, indent=2)
        print(f"Created {filename}: {len(samples)} samples")
    
    # Create README
    readme_content = """# Bi-Temporal Flood Dataset - Example

This is a dummy dataset showing the structure for bi-temporal flood detection.

## Dataset Info
- **Samples**: 5 total (3 train, 1 val, 1 test)
- **Image Size**: 256×256 pixels at 10m/px
- **Sensors**: Sentinel-1 (S1) + Sentinel-2 (S2)
- **Temporal**: 2 dates (pre-flood and post-flood)
- **Task**: Flood segmentation

## Flood Samples
- sample_001: ✓ Flood
- sample_002: ✗ No flood
- sample_003: ✓ Flood
- sample_004: ✓ Flood
- sample_005: ✗ No flood

## Data Structure
```
BiTemporalFloodExample/
├── train_list.json          # ["sample_001", "sample_002", "sample_003"]
├── val_list.json            # ["sample_004"]
├── test_list.json           # ["sample_005"]
├── S1/
│   ├── sample_001.tif       # (4, 256, 256) - 2 dates × 2 bands (VV, VH)
│   ├── sample_002.tif
│   └── ...
├── S2/
│   ├── sample_001.tif       # (26, 256, 256) - 2 dates × 13 bands
│   ├── sample_002.tif
│   └── ...
├── dates/
│   ├── sample_001.json      # {"s1": [20200501, 20200715], "s2": [20200505, 20200720]}
│   ├── sample_002.json
│   └── ...
└── labels/
    ├── sample_001.tif       # (1, 256, 256) - Binary flood mask
    ├── sample_002.tif
    └── ...
```

## Band Structure

### S1 (Sentinel-1 SAR)
4 bands total:
1. Date1_VV (Pre-flood)
2. Date1_VH (Pre-flood)
3. Date2_VV (Post-flood)
4. Date2_VH (Post-flood)

### S2 (Sentinel-2 Optical)
26 bands total (2 dates × 13 bands):
- Bands 1-13: Date 1 (Pre-flood)
- Bands 14-26: Date 2 (Post-flood)

Each date has 13 bands: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 + 3 additional

### Labels
Binary segmentation mask:
- 0: No change / Normal area
- 1: Flooded area

## Dates Format
```json
{
  "s1": [20200501, 20200715],  // Pre-flood: May 1, Post-flood: July 15
  "s2": [20200505, 20200720]   // Slightly offset for S2
}
```

## Usage

Train self-supervised model:
```bash
python src/train.py exp=BiTemporal_AnySat data_dir=data/BiTemporalFloodExample
```

Fine-tune for flood segmentation:
```bash
python src/train.py exp=BiTemporal_AnySat_FT_SemSeg data_dir=data/BiTemporalFloodExample
```
"""
    
    readme_path = os.path.join(base_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"\nCreated README.md")
    
    print("\n" + "="*60)
    print("✓ Dataset creation complete!")
    print("="*60)
    print(f"\nDataset location: {base_dir}")
    print("\nNext steps:")
    print("1. Update configs/dataset/BiTemporalS1S2.yaml:")
    print("   - true_len: 3")
    print("   - true_len_val: 1")
    print("2. Run: python src/train.py exp=BiTemporal_AnySat data_dir=data/BiTemporalFloodExample")
    print()


if __name__ == '__main__':
    main()
