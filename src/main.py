# main.py (for data testing and visualization)
import torch
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np # For spectral display
import os
import logging
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path

from dataset import CottonLeafDataset # Uses the revised CottonLeafDataset
from transforms import get_rgb_transforms, denormalize_image_tensor
from progression import DiseaseProgressionSimulator
from config import (
    ORIGINAL_DATASET_ROOT, AUGMENTED_DATASET_ROOT, IMAGE_SIZE_RGB, IMAGE_SIZE_SPECTRAL,
    BATCH_SIZE, NUM_WORKERS, DEFAULT_STAGE_MAP, VISUALIZATION_DIR
)

logger = logging.getLogger(__name__)

# --- Custom Collate Function for Visualization DataLoader ---
# This collate function is slightly different from the training one,
# as it needs to handle the Dict output from __getitem__ and also stage strings.
def visualization_collate_fn(batch: List[Dict[str, Any]]) \
        -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, List[str], List[int]]:
    """
    Collate function for visualization that handles Dict items.
    If an item has spectral_image as None, a zero tensor of expected shape is used for batching.
    Returns: batched_rgb, batched_spectral, batched_labels, stages_list, ids_list
    """
    # Filter out items where rgb_image is None (due to loading error)
    valid_batch = [item for item in batch if item["rgb_image"] is not None and item["label"] != -1]

    if not valid_batch: # All items in batch failed to load
        logger.warning("Collate function for visualization received an entirely empty or error-filled batch.")
        return torch.empty(0), None, torch.empty(0, dtype=torch.long), [], []

    rgb_images = torch.stack([item["rgb_image"] for item in valid_batch])
    labels = torch.tensor([item["label"] for item in valid_batch], dtype=torch.long)
    stages = [item["stage"] for item in valid_batch]
    ids = [item["id"] for item in valid_batch]
    
    batched_spectral: Optional[torch.Tensor] = None
    has_any_spectral_data = any(item["spectral_image"] is not None for item in valid_batch)

    if has_any_spectral_data:
        spectral_images_processed = []
        ref_spectral_shape = None
        for item in valid_batch: # Find a reference shape
            if item["spectral_image"] is not None:
                ref_spectral_shape = item["spectral_image"].shape
                break
        
        if ref_spectral_shape is None:
            ref_spectral_shape = (1, IMAGE_SIZE_SPECTRAL[0], IMAGE_SIZE_SPECTRAL[1])

        for item in valid_batch:
            if item["spectral_image"] is not None:
                spectral_images_processed.append(item["spectral_image"])
            else:
                spectral_images_processed.append(torch.zeros(ref_spectral_shape, dtype=torch.float32))
        try:
            batched_spectral = torch.stack(spectral_images_processed)
        except RuntimeError as e:
            logger.error(f"RuntimeError stacking spectral images for viz: {e}. Shapes:")
            for i, s_img in enumerate(spectral_images_processed): logger.error(f"  Item {i} shape: {s_img.shape if s_img is not None else 'None'}")
            batched_spectral = None
    
    return rgb_images, batched_spectral, labels, stages, ids


# --- Visualization Functions ---
def plot_dataset_sample(rgb_tensor: torch.Tensor,
                        spectral_tensor: Optional[torch.Tensor],
                        label: int,
                        stage: str,
                        class_names: List[str],
                        sample_id: int,
                        filename_prefix: str = "dataset_sample"):
    num_plots = 1 + (1 if spectral_tensor is not None and spectral_tensor.numel() > 0 else 0)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    if num_plots == 1: axes = [axes] 
    
    title_class_name = "Unknown Class"
    if 0 <= label < len(class_names):
        title_class_name = class_names[label]
    elif label == -1 and stage == "error_loading_rgb":
        title_class_name = "Error Image"
    
    fig.suptitle(f"ID: {sample_id} - Class: {title_class_name} (Label: {label}), Stage: {stage}", fontsize=14)

    # Denormalize RGB for display
    rgb_display = denormalize_image_tensor(rgb_tensor.cpu())
    axes[0].imshow(rgb_display.permute(1, 2, 0).numpy())
    axes[0].set_title("Processed RGB Image")
    axes[0].axis('off')

    if spectral_tensor is not None and spectral_tensor.numel() > 0:
        spectral_display_np = spectral_tensor.cpu().numpy()
        # Handle single vs multi-channel spectral for display
        cmap_to_use = 'viridis' # Default
        if spectral_display_np.shape[0] == 1: # Single channel (e.g., NDVI)
            spectral_img_to_show = spectral_display_np.squeeze()
            cmap_to_use = 'RdYlGn' if np.any(spectral_display_np < 0) else 'viridis' # NDVI like or general grayscale
        elif spectral_display_np.shape[0] >= 3: # Multi-channel, show first 3 as RGB (heuristic)
            spectral_img_to_show = np.transpose(spectral_display_np[:3], (1, 2, 0))
            min_vals = spectral_img_to_show.min(axis=(0,1), keepdims=True)
            max_vals = spectral_img_to_show.max(axis=(0,1), keepdims=True)
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1 # Avoid division by zero
            spectral_img_to_show = (spectral_img_to_show - min_vals) / range_vals
            spectral_img_to_show = np.clip(spectral_img_to_show, 0, 1)
            cmap_to_use = None # Use default RGB rendering
        else: # Other multi-channel cases (e.g. 2 channels), show first
            spectral_img_to_show = spectral_display_np[0]
            
        im = axes[1].imshow(spectral_img_to_show, cmap=cmap_to_use)
        axes[1].set_title("Processed Spectral")
        axes[1].axis('off')
        if cmap_to_use:
             fig.colorbar(im, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)

    safe_filename = f"{filename_prefix}_id{sample_id}_label{label}.png".replace("/", "_").replace(" ", "_")
    save_path = os.path.join(VISUALIZATION_DIR, safe_filename)
    plt.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved dataset sample visualization to {save_path}")


def save_progression_visualization(original_pil_img: Image.Image,
                                   progressed_pil_imgs: Dict[str, Image.Image],
                                   base_filename: str):
    num_images = 1 + len(progressed_pil_imgs)
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    if num_images == 1: axes = [axes]

    axes[0].imshow(original_pil_img)
    axes[0].set_title("Original")
    axes[0].axis('off')

    for i, (stage, p_img) in enumerate(progressed_pil_imgs.items()):
        axes[i + 1].imshow(p_img)
        axes[i + 1].set_title(f"Sim. Stage: {stage}")
        axes[i + 1].axis('off')

    safe_filename = Path(base_filename).name # Use only filename part
    save_path = os.path.join(VISUALIZATION_DIR, f"{safe_filename}_progression_stages.png")
    plt.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved progression visualization to {save_path}")

# --- Testing Functions ---
def demonstrate_progression_simulation(dataset_root: str, sample_idx: int = 0):
    logger.info(f"Demonstrating Disease Progression Simulator using an image from {dataset_root}")
    # Minimal dataset instance for getting an image path
    temp_dataset = CottonLeafDataset(root_dir=dataset_root, transform_rgb=None, apply_progression=False, use_spectral=False)
    if not temp_dataset.images_metadata:
        logger.error("No images found in temp_dataset for progression demo. Exiting.")
        return
    if sample_idx >= len(temp_dataset):
        logger.warning(f"Sample index {sample_idx} out of bounds. Using index 0.")
        sample_idx = 0

    img_path_str = temp_dataset.images_metadata[sample_idx]["rgb_path"]
    try:
        original_pil = Image.open(img_path_str).convert('RGB').resize(IMAGE_SIZE_RGB)
    except Exception as e:
        logger.error(f"Could not open image {img_path_str} for progression demo: {e}")
        return

    simulator = DiseaseProgressionSimulator()
    progressed_stages: Dict[str, Image.Image] = {}
    # Simulate for all stages defined in the simulator
    for stage_key in simulator.stage_effects.keys():
        progressed_stages[stage_key] = simulator.apply(original_pil.copy(), stage_key)

    save_progression_visualization(original_pil, progressed_stages, img_path_str)


def test_dataset_loading_and_visualization(root_dir: str, dataset_name: str, apply_progression: bool = False, use_spectral: bool = True):
    logger.info(f"Testing {dataset_name} from: {root_dir} (Progression: {apply_progression}, Spectral: {use_spectral})")
    
    # Use validation transforms for visualization consistency, or train if you want to see augs
    current_transforms_rgb = get_rgb_transforms(train=False, image_size=IMAGE_SIZE_RGB) 
    
    dataset = CottonLeafDataset(
        root_dir=root_dir,
        transform_rgb=current_transforms_rgb,
        stage_map=DEFAULT_STAGE_MAP,
        apply_progression=apply_progression,
        use_spectral=use_spectral
    )

    if len(dataset) == 0:
        logger.warning(f"Dataset '{dataset_name}' is empty. Skipping test.")
        return

    dataloader = DataLoader(dataset, batch_size=min(BATCH_SIZE, 4), shuffle=True, # Smaller batch for viz
                            num_workers=0, # Easier debugging for visualization scripts
                            collate_fn=visualization_collate_fn)
    try:
        rgb_batch, spectral_batch, labels_batch, stages_batch, ids_batch = next(iter(dataloader))
        
        logger.info(f"Sample Batch from {dataset_name}:")
        logger.info(f"  RGB tensor shape: {rgb_batch.shape if rgb_batch is not None else 'None'}")
        if spectral_batch is not None:
            logger.info(f"  Spectral tensor shape: {spectral_batch.shape}")
        else:
            logger.info(f"  Spectral tensor: None (use_spectral might be False or all items had spectral errors)")
        logger.info(f"  Labels (first few): {labels_batch[:4]}")
        logger.info(f"  Stages (first few): {stages_batch[:4]}")
        logger.info(f"  IDs (first few): {ids_batch[:4]}")

        class_names_for_plot = dataset.classes
        if not class_names_for_plot: # Should not happen if dataset loaded correctly
             class_names_for_plot = [f"Label {i}" for i in range(max(labels_batch.tolist(), default=-1)+1)]

        for i in range(rgb_batch.shape[0]): # Plot each item in the small batch
            plot_dataset_sample(
                rgb_tensor=rgb_batch[i],
                spectral_tensor=spectral_batch[i] if spectral_batch is not None else None,
                label=labels_batch[i].item(),
                stage=stages_batch[i],
                class_names=class_names_for_plot,
                sample_id=ids_batch[i],
                filename_prefix=f"{dataset_name}_sample_item{i}"
            )
        logger.info(f"Successfully tested and visualized samples from {dataset_name}.")

    except StopIteration:
        logger.error(f"DataLoader for {dataset_name} is empty despite dataset having {len(dataset)} items.")
    except Exception as e:
        logger.error(f"Error during batch fetching or processing for {dataset_name}: {e}", exc_info=True)


def run_visualizations():
    # Ensure VISUALIZATION_DIR exists
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    
    logger.info("Starting data loading and progression demonstration...")

    # Check if ORIGINAL_DATASET_ROOT is valid before proceeding
    if not Path(ORIGINAL_DATASET_ROOT).exists() or not any(Path(ORIGINAL_DATASET_ROOT).iterdir()):
        logger.error(f"ORIGINAL_DATASET_ROOT ('{ORIGINAL_DATASET_ROOT}') is not valid or empty. Skipping visualizations that depend on it.")
        return

    demonstrate_progression_simulation(ORIGINAL_DATASET_ROOT, sample_idx=0)

    test_dataset_loading_and_visualization(
        root_dir=ORIGINAL_DATASET_ROOT,
        dataset_name="OriginalDataset_NoProg_NoSpectral",
        apply_progression=False,
        use_spectral=False
    )
    
    test_dataset_loading_and_visualization(
        root_dir=ORIGINAL_DATASET_ROOT,
        dataset_name="OriginalDataset_WithProg_WithSpectral",
        apply_progression=True,
        use_spectral=True
    )
    
    if Path(AUGMENTED_DATASET_ROOT).exists() and any(Path(AUGMENTED_DATASET_ROOT).iterdir()):
        test_dataset_loading_and_visualization(
            root_dir=AUGMENTED_DATASET_ROOT,
            dataset_name="AugmentedDataset_NoProg_WithSpectral",
            apply_progression=False,
            use_spectral=True
        )
    else:
        logger.warning(f"AUGMENTED_DATASET_ROOT ('{AUGMENTED_DATASET_ROOT}') not found or empty. Skipping related test.")
        
    logger.info("Finished all visualization tests.")

if __name__ == "__main__":
    # Setup basic logging for direct script execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    run_visualizations()