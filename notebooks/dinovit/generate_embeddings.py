import os
from typing import Dict
from functools import partial
from tqdm import tqdm
from equimo.io import load_model, download
from datasets import load_dataset
from jax import devices, vmap, device_put, nn
import jax.random as jr
import jax.numpy as jnp
import jax_dataloader as jdl
import equinox as eqx
import augmax
import json
from pathlib import Path
from datetime import datetime

# do not prealocate memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# Set cuda device to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

jdl.manual_seed(6573)
key = jr.PRNGKey(1236)
device0 = devices()[0]
device1 = devices()[1]

# Model configurations
DINOV2_MODELS = {
    "small": "dinov2_vits14_reg",
    "big": "dinov2_vitb14_reg",
    "large": "dinov2_vitl14_reg",
    "giant": "dinov2_vitg14_reg",
}

def setup_directories(base_path: str, dataset_name: str) -> Dict[str, Path]:
    """Create directory structure for storing embeddings."""
    base = Path(base_path)
    base.mkdir(parents=True, exist_ok=True)
    
    # Create dataset directory
    dataset_dir = base / dataset_name.replace('/', '_')
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {
        "dataset_root": dataset_dir
    }
    
    # Create split-specific directories for each model
    splits = ['train', 'test']
    
    for model_name in DINOV2_MODELS.keys():
        model_dir = dataset_dir / f"dinov2_{model_name}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the main feature directories
        forward_dir = model_dir / "forward_features"
        pooled_dir = model_dir / "pooled"
        forward_dir.mkdir(parents=True, exist_ok=True)
        pooled_dir.mkdir(parents=True, exist_ok=True)
        
        # Add split subdirectories
        for split in splits:
            split_forward_dir = forward_dir / split
            split_pooled_dir = pooled_dir / split
            
            split_forward_dir.mkdir(parents=True, exist_ok=True)
            split_pooled_dir.mkdir(parents=True, exist_ok=True)
            
            paths[f"{model_name}_forward_{split}"] = split_forward_dir
            paths[f"{model_name}_pooled_{split}"] = split_pooled_dir
    
    return paths

def get_transform():
    """Get image transformation pipeline."""
    return augmax.Chain(
        augmax.Normalize(),
        augmax.Resize(518, 518),
    )

def setup_data_loader(split: str, batch_size: int, dataset_name: str):
    """Setup data loader for a specific split."""
    ds = load_dataset(dataset_name)
    return jdl.DataLoader(
        ds[split],
        backend='jax',
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

def process_batch(model, batch, key, transform, device_compute, device_store):
    """Process a single batch of images."""
    if batch['image'].ndim != 4:
        return None, None
    
    # Prepare images
    img = device_put(batch['image'], device_compute).astype(jnp.float32) / 255.0
    keys = device_put(jr.split(key, img.shape[0]), device_compute)
    trans_img = jnp.moveaxis(
        vmap(transform)(keys, img),
        -1, -3
    )
    
    # Get features and pooled output
    features = device_put(vmap(model.forward_features)(trans_img, keys), device_store)
    pooled = device_put(vmap(model)(trans_img), device_store)
    
    return features, pooled

def save_batch(features, pooled, batch_idx: int, labels, paths: Dict[str, Path], model_size: str, split: str):
    """Save batch features and metadata."""
    if features is None or pooled is None:
        return
        
    # Save forward features to split-specific directory
    jnp.savez(
        paths[f"{model_size}_forward_{split}"] / f"batch_{batch_idx}.npz",
        x_norm_cls_token=features["x_norm_cls_token"],
        x_norm_reg_tokens=features["x_norm_reg_tokens"],
        labels=labels
    )
    
    # Save pooled features to split-specific directory
    jnp.savez(
        paths[f"{model_size}_pooled_{split}"] / f"batch_{batch_idx}.npz",
        pooled=pooled,
        labels=labels
    )

def generate_embeddings(output_dir: str = "embeddings", dataset_name: str = "tiny-imagenet-200-clean"):
    """Main function to generate and save embeddings."""
    paths = setup_directories(output_dir, dataset_name)
    transform = get_transform()
    
    # Setup metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "models": DINOV2_MODELS,
        "dataset": dataset_name,
        "preprocessing": {
            "resize": 518,
            "normalize": True
        },
        "batch_size": {'small': 64, 'big': 48, 'large': 32, 'giant': 16},
        "seed": 6573,
        "key_seed": 1236
    }
    
    # Process each model
    for model_size, model_id in DINOV2_MODELS.items():
        print(f"\nProcessing {model_size} model...")
        
        # Load model
        model = eqx.nn.inference_mode(device_put(load_model(cls="vit", identifier=model_id), device1))
        
        # Process train and test splits
        for split in ['train', 'test']:
            print(f"\nProcessing {split} split...")
            dataloader = setup_data_loader(split, metadata['batch_size'][model_size], dataset_name)
            
            for batch_idx, batch in enumerate(tqdm(iter(dataloader), total=len(dataloader))):
                features, pooled = process_batch(model, batch, key, transform, device1, device0)
                save_batch(features, pooled, batch_idx, batch['label'], paths, model_size, split)
                
        # Update metadata with model-specific information
        metadata[f"dinov2_{model_size}"] = {
            "feature_dim": model.dim,
            "num_patches": model.num_patches
        }
    
    # Save metadata
    with open(paths["dataset_root"] / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    # generate_embeddings(dataset_name="benjamin-paine/imagenet-1k-256x256")
    generate_embeddings(dataset_name="slegroux/tiny-imagenet-200-clean")

    
